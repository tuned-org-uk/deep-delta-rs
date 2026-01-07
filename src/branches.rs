//! Generator branches for Deep Delta Learning parameters.
//!
//! Burn tensor shape conventions (critical):
//! - Tensor rank is part of the type: `Tensor<B, const D: usize, K>`.
//! - Many operations in Burn are rank-preserving unless you explicitly request a new rank
//!   using APIs like `flatten::<D2>(..)`, `reshape::<D2>(..)`, `squeeze::<D2>()`.
//! - `nn::Linear` is rank-preserving: it transforms the last dimension and keeps all leading dims.
//!
//! Therefore, to ensure branches return rank-2 tensors:
//! - Always convert the input state `x: Tensor<B, 3>` into an explicit rank-2 tensor
//!   before passing to any Linear.
//!
//! This module follows Appendix A from the paper (arXiv:2601.00417v1). [file:1]

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

/// Convert X [B, D, V] into a pooled representation [B, D] by averaging over V.
///
/// Implementation detail:
/// - We compute a mean over dim=2 but do not assume the rank changes.
/// - If mean keeps dims, we squeeze the singleton dimension.
///
/// Returned tensor is always `Tensor<B, 2>`.
#[inline]
pub fn pool_v_to_bd<B: Backend>(x: &Tensor<B, 3>) -> Tensor<B, 2> {
    // Try to keep semantics robust: mean_dim might return [B, D, 1] or [B, D] depending on version.
    // We force rank-2 by squeezing dim=2 if it's a singleton.
    let pooled = x.clone().mean_dim(2);

    // If pooled is already rank-2 in your Burn version, this line won't compile.
    // So we implement using `flatten` in a way that always compiles:
    // Strategy: compute sum over V explicitly using elementwise operations is not available here.
    // Instead, we rely on the fact that in current Burn, `mean_dim` returns same rank (keeps dim).
    // Then we squeeze to rank-2.
    let pooled: Tensor<B, 2> = pooled.squeeze::<2>(2);
    pooled
}

/// Convert X [B, D, V] into a pooled representation [B, V] by averaging over D.
#[inline]
pub fn pool_d_to_bv<B: Backend>(x: &Tensor<B, 3>) -> Tensor<B, 2> {
    let pooled = x.clone().mean_dim(1);
    let pooled: Tensor<B, 2> = pooled.squeeze::<2>(1);
    pooled
}

/// L2-normalize vectors along the last dimension for a rank-2 tensor [B, D].
#[inline]
pub fn l2_normalize_bd<B: Backend>(x: Tensor<B, 2>, eps: f32) -> Tensor<B, 2> {
    // norm: [B]
    let norm: Tensor<B, 2> = x.clone().powf_scalar(2.0).sum_dim(1).sqrt().add_scalar(eps);
    // [B] -> [B, 1]
    let norm: Tensor<B, 2> = norm.unsqueeze::<2>();
    x.div(norm)
}

// ---------------------------
// K-Branch
// ---------------------------

#[derive(Config, Debug)]
pub struct KBranchConfig {
    pub d_model: usize,
    pub d_hidden: usize,
    #[config(default = "1e-8")]
    pub eps: f32,
}

/// k(X) generator.
///
/// k̃ = MLP(Pool(X)), k = k̃ / (||k̃|| + eps) [file:1]
#[derive(Module, Debug)]
pub struct KBranch<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    eps: f32,
}

impl KBranchConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> KBranch<B> {
        KBranch {
            linear1: LinearConfig::new(self.d_model, self.d_hidden).init(device),
            linear2: LinearConfig::new(self.d_hidden, self.d_model).init(device),
            eps: self.eps,
        }
    }
}

impl<B: Backend> KBranch<B> {
    /// X: [B, D, V] -> k: [B, D]
    pub fn forward(&self, x: &Tensor<B, 3>) -> Tensor<B, 2> {
        let pooled: Tensor<B, 2> = pool_v_to_bd(x); // [B, D]
        let h: Tensor<B, 2> = self.linear1.forward(pooled);
        let h: Tensor<B, 2> = burn::tensor::activation::gelu(h);
        let k: Tensor<B, 2> = self.linear2.forward(h);
        l2_normalize_bd(k, self.eps)
    }
}

// ---------------------------
// Beta-Branch
// ---------------------------

#[derive(Config, Debug)]
pub struct BetaBranchConfig {
    pub d_model: usize,
    pub d_hidden: usize,
}

/// β(X) generator.
///
/// β(X) = 2 * sigmoid( ... ) in range [0,2]. [file:1]
#[derive(Module, Debug)]
pub struct BetaBranch<B: Backend> {
    w_in: Linear<B>,
    w_out: Linear<B>,
}

impl BetaBranchConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BetaBranch<B> {
        BetaBranch {
            w_in: LinearConfig::new(self.d_model, self.d_hidden).init(device),
            w_out: LinearConfig::new(self.d_hidden, 1).init(device),
        }
    }
}

impl<B: Backend> BetaBranch<B> {
    /// X: [B, D, V] -> beta: [B, 1]
    pub fn forward(&self, x: &Tensor<B, 3>) -> Tensor<B, 2> {
        use burn::tensor::activation::sigmoid;
        let pooled: Tensor<B, 2> = pool_v_to_bd(x); // [B, D]
        let h: Tensor<B, 2> = self.w_in.forward(pooled).tanh();
        let logit: Tensor<B, 2> = self.w_out.forward(h); // [B, 1]
        sigmoid(logit).mul_scalar(2.0)
    }
}

// ---------------------------
// V-Branch
// ---------------------------

#[derive(Config, Debug)]
pub struct VBranchConfig {
    pub d_model: usize,
    pub dv: usize,
    pub d_hidden: usize,
}

/// v(X) generator.
///
/// The paper treats v(X) as a flexible content-update branch. [file:1]
///
/// Default here:
/// - pool over V to get [B, D]
/// - MLP D -> hidden -> dv
#[derive(Module, Debug)]
pub struct VBranch<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
}

impl VBranchConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> VBranch<B> {
        VBranch {
            linear1: LinearConfig::new(self.d_model, self.d_hidden).init(device),
            linear2: LinearConfig::new(self.d_hidden, self.dv).init(device),
        }
    }
}

impl<B: Backend> VBranch<B> {
    /// X: [B, D, V] -> v: [B, V]
    pub fn forward(&self, x: &Tensor<B, 3>) -> Tensor<B, 2> {
        let pooled: Tensor<B, 2> = pool_v_to_bd(x); // [B, D]
        let h: Tensor<B, 2> = self.linear1.forward(pooled);
        let h: Tensor<B, 2> = burn::tensor::activation::gelu(h);
        let v: Tensor<B, 2> = self.linear2.forward(h);
        v
    }
}

// ---------------------------
// Bundle
// ---------------------------

#[derive(Config, Debug)]
pub struct DeltaBranchesConfig {
    pub d_model: usize,
    pub dv: usize,
    pub d_hidden: usize,
    #[config(default = "1e-8")]
    pub eps: f32,
}

#[derive(Module, Debug)]
pub struct DeltaBranches<B: Backend> {
    pub k_branch: KBranch<B>,
    pub beta_branch: BetaBranch<B>,
    pub v_branch: VBranch<B>,
}

impl DeltaBranchesConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DeltaBranches<B> {
        DeltaBranches {
            k_branch: KBranchConfig::new(self.d_model, self.d_hidden)
                .with_eps(self.eps)
                .init(device),
            beta_branch: BetaBranchConfig::new(self.d_model, self.d_hidden).init(device),
            v_branch: VBranchConfig::new(self.d_model, self.dv, self.d_hidden).init(device),
        }
    }
}

impl<B: Backend> DeltaBranches<B> {
    /// X: [B, D, V] -> (k: [B, D], beta: [B, 1], v: [B, V])
    pub fn forward(&self, x: &Tensor<B, 3>) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let k = self.k_branch.forward(x);
        let beta = self.beta_branch.forward(x);
        let v = self.v_branch.forward(x);
        (k, beta, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::backend::AutoBackend;
    type TestBackend = AutoBackend;
    use burn::tensor::Distribution;

    #[test]
    fn test_pool_v_to_bd_shape() {
        let device = Default::default();
        let x =
            Tensor::<TestBackend, 3>::random([2, 8, 4], Distribution::Uniform(-1.0, 1.0), &device);
        let pooled = pool_v_to_bd(&x);
        assert_eq!(pooled.dims(), [2, 8]);
    }

    #[test]
    fn test_k_branch_shape_and_norm() {
        let device = Default::default();
        let cfg = KBranchConfig::new(8, 16);
        let branch = cfg.init::<TestBackend>(&device);

        let x =
            Tensor::<TestBackend, 3>::random([2, 8, 4], Distribution::Uniform(-1.0, 1.0), &device);
        let k = branch.forward(&x);
        assert_eq!(k.dims(), [2, 8]);

        let norms: Tensor<TestBackend, 2> = k.clone().powf_scalar(2.0).sum_dim(1).sqrt();
        let norms: Vec<f32> = norms.into_data().to_vec().unwrap();
        for n in norms {
            assert!((n - 1.0).abs() < 0.05);
        }
    }

    #[test]
    fn test_beta_branch_range() {
        let device = Default::default();
        let cfg = BetaBranchConfig::new(8, 16);
        let branch = cfg.init::<TestBackend>(&device);

        let x =
            Tensor::<TestBackend, 3>::random([2, 8, 4], Distribution::Uniform(-1.0, 1.0), &device);
        let beta = branch.forward(&x);
        assert_eq!(beta.dims(), [2, 1]);

        let vals: Vec<f32> = beta.into_data().to_vec().unwrap();
        for b in vals {
            assert!(b >= 0.0 && b <= 2.0);
        }
    }

    #[test]
    fn test_v_branch_shape() {
        let device = Default::default();
        let cfg = VBranchConfig::new(8, 4, 16);
        let branch = cfg.init::<TestBackend>(&device);

        let x =
            Tensor::<TestBackend, 3>::random([2, 8, 4], Distribution::Uniform(-1.0, 1.0), &device);
        let v = branch.forward(&x);
        assert_eq!(v.dims(), [2, 4]);
    }

    #[test]
    fn test_bundle_shapes() {
        let device = Default::default();
        let cfg = DeltaBranchesConfig::new(8, 4, 16);
        let branches = cfg.init::<TestBackend>(&device);

        let x =
            Tensor::<TestBackend, 3>::random([2, 8, 4], Distribution::Uniform(-1.0, 1.0), &device);
        let (k, beta, v) = branches.forward(&x);

        assert_eq!(k.dims(), [2, 8]);
        assert_eq!(beta.dims(), [2, 1]);
        assert_eq!(v.dims(), [2, 4]);
    }
}
