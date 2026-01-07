//! Neural network modules for Deep Delta Learning
//!
//! Conventions (important in Burn):
//! - `nn::Linear` is rank-preserving: it applies to the last dimension and keeps all leading dims.
//!   If you feed `[B, D, V]` into Linear(D -> H), you'll get `[B, D, H]` (still 3D).
//! - Therefore, branches that *must* return rank-2 tensors (e.g. k: [B, D], beta: [B, 1], v: [B, V])
//!   should **explicitly pool/flatten** before any Linear layers.
//!
//! This module defines:
//! - `DeltaResidual`: a full Delta-Res block using internal generator branches.
//! - `DeltaResidualExternal`: a variant that takes externally-provided (k, beta, v), useful for tauformer.

use burn::prelude::*;

use crate::branches::{DeltaBranches, DeltaBranchesConfig};
use crate::delta::delta_update;

/// Configuration for the DeltaResidual block.
#[derive(Config, Debug)]
pub struct DeltaResidualConfig {
    /// Feature dimension d
    pub d_model: usize,
    /// Value dimension dv
    pub dv: usize,
    /// Hidden dimension used in generator MLPs
    pub d_hidden: usize,
    /// Numerical epsilon used in delta_update and k-normalization
    #[config(default = "1e-8")]
    pub eps: f32,
}

/// Delta Residual block (Delta-Res).
///
/// Implements Eq. (2.5):
/// X_{l+1} = X_l + β(X_l) k(X_l) (v(X_l)^T - k(X_l)^T X_l)
#[derive(Module, Debug)]
pub struct DeltaResidual<B: Backend> {
    branches: DeltaBranches<B>,
    eps: f32,
}

impl DeltaResidualConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DeltaResidual<B> {
        let branches_cfg = DeltaBranchesConfig::new(self.d_model, self.dv, self.d_hidden);
        let branches_cfg = branches_cfg.with_eps(self.eps);

        DeltaResidual {
            branches: branches_cfg.init(device),
            eps: self.eps,
        }
    }
}

impl<B: Backend> DeltaResidual<B> {
    /// Forward for matrix state.
    ///
    /// Signature convention:
    /// - input:  `x` is always `[B, D, V]` (`Tensor<B, 3>`)
    /// - output: `y` is always `[B, D, V]` (`Tensor<B, 3>`)
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let (k, beta, v) = self.branches.forward(&x);
        delta_update(x, k, v, beta, self.eps)
    }

    /// Like `forward`, but also returns the generated parameters (useful for debugging/tests).
    pub fn forward_with_params(
        &self,
        x: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let (k, beta, v) = self.branches.forward(&x);
        let y = delta_update(x, k.clone(), v.clone(), beta.clone(), self.eps);
        (y, k, beta, v)
    }
}

/// Delta residual that takes external (k, beta, v).
#[derive(Module, Debug)]
pub struct DeltaResidualExternal<B: Backend> {
    eps: f32,
    _backend: core::marker::PhantomData<B>, // Required if no other B-related fields exist
}

impl<B: Backend> DeltaResidualExternal<B> {
    pub fn new(eps: f32) -> Self {
        Self {
            eps,
            _backend: core::marker::PhantomData,
        }
    }

    /// Forward with external parameters.
    ///
    /// Shapes:
    /// - x:    [B, D, V]
    /// - k:    [B, D]
    /// - v:    [B, V]
    /// - beta: [B, 1]
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        k: Tensor<B, 2>,
        v: Tensor<B, 2>,
        beta: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        delta_update(x, k, v, beta, self.eps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub type TestBackend = burn_ndarray::NdArray<f32>;
    use burn::tensor::Distribution;

    #[test]
    fn test_delta_residual_shapes() {
        let device = Default::default();

        let cfg = DeltaResidualConfig::new(8, 4, 16);
        let model = cfg.init::<TestBackend>(&device);

        let x =
            Tensor::<TestBackend, 3>::random([2, 8, 4], Distribution::Uniform(-1.0, 1.0), &device);

        let (y, k, beta, v) = model.forward_with_params(x);
        assert_eq!(y.dims(), [2, 8, 4]);
        assert_eq!(k.dims(), [2, 8]);
        assert_eq!(beta.dims(), [2, 1]);
        assert_eq!(v.dims(), [2, 4]);
    }

    #[test]
    fn test_delta_residual_external_shapes() {
        let device = Default::default();

        let block = DeltaResidualExternal::new(1e-8);

        let x =
            Tensor::<TestBackend, 3>::random([2, 8, 4], Distribution::Uniform(-1.0, 1.0), &device);
        let k = Tensor::<TestBackend, 2>::random([2, 8], Distribution::Uniform(-1.0, 1.0), &device);
        let v = Tensor::<TestBackend, 2>::random([2, 4], Distribution::Uniform(-1.0, 1.0), &device);
        let beta =
            Tensor::<TestBackend, 2>::random([2, 1], Distribution::Uniform(0.0, 2.0), &device);

        let y = block.forward(x, k, v, beta);
        assert_eq!(y.dims(), [2, 8, 4]);
    }
}
