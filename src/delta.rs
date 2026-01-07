//! Core Deep Delta Learning operators
//!
//! This module implements the fundamental Delta operator from the paper
//! "Deep Delta Learning" (arXiv:2601.00417v1). [file:1]
//!
//! The Delta operator generalizes residual connections via a rank-1 geometric
//! transformation: X_{l+1} = X_l + β(X_l) k(X_l) (v(X_l)^T - k(X_l)^T X_l)

use burn::prelude::*;

/// Core Delta update for matrix-valued hidden states.
///
/// Implements Equation (2.5) from the paper:
/// X_{l+1} = X_l + β * k * (v^T - k^T X_l)
///
/// # Arguments
/// * `x` - Hidden state matrix of shape [batch, d, dv]
/// * `k` - Reflection direction vector of shape [batch, d]
/// * `v` - Value update vector of shape [batch, dv]
/// * `beta` - Gating scalar of shape [batch, 1]
///
/// # Returns
/// Updated state X_{l+1} of shape [batch, d, dv]
pub fn delta_update<B: Backend>(
    x: Tensor<B, 3>,    // [B, D, V]
    k: Tensor<B, 2>,    // [B, D]
    v: Tensor<B, 2>,    // [B, V]
    beta: Tensor<B, 2>, // [B, 1]
    eps: f32,
) -> Tensor<B, 3> {
    // 1. Normalize k to unit length
    // k: [B, D] -> sum_dim(1) -> [B, 1] (rank preserved)
    let k_squared = k.clone().powf_scalar(2.0);
    let k_norm_val: Tensor<B, 2> = k_squared.sum_dim(1).sqrt().add_scalar(eps);
    // k_norm_val is already [B, 1], so we can directly divide
    let k_normalized: Tensor<B, 2> = k.div(k_norm_val);

    // 2. Project X onto k
    // k_t: [B, 1, D] created from [B, D]
    let k_t: Tensor<B, 3> = k_normalized.clone().unsqueeze_dim::<3>(1);

    // proj: [B, 1, V] = [B, 1, D] x [B, D, V]
    let proj: Tensor<B, 3> = k_t.matmul(x.clone());

    // 3. Compute delta_val = v^T - k^T X
    // v: [B, V] -> [B, 1, V]
    let v_row: Tensor<B, 3> = v.unsqueeze_dim::<3>(1);
    // delta_val: [B, 1, V]
    let delta_val = v_row.sub(proj);

    // 4. Form rank-1 update: k * delta_val
    // k_col: [B, D] -> [B, D, 1]
    let k_col: Tensor<B, 3> = k_normalized.unsqueeze_dim::<3>(2);
    // rank1: [B, D, V] = [B, D, 1] x [B, 1, V]
    let rank1: Tensor<B, 3> = k_col.matmul(delta_val);

    // 5. Gate the update: beta * rank1
    // beta: [B, 1] -> [B, 1, 1] for broadcasting over [B, D, V]
    let beta_expanded: Tensor<B, 3> = beta.unsqueeze_dim::<3>(2); // becomes [B, 1, 1]

    let gated_update = rank1.mul(beta_expanded);

    // 6. Final residual update
    x.add(gated_update)
}

/// Delta update for vector-valued hidden states (dv = 1 case).
///
/// Implements x_{l+1} = x_l + β_l * k_l * (v_l - k_l^T x_l)
pub fn delta_update_vec<B: Backend>(
    x: Tensor<B, 2>,    // [B, D]
    k: Tensor<B, 2>,    // [B, D]
    v: Tensor<B, 2>,    // [B, 1]
    beta: Tensor<B, 2>, // [B, 1]
    eps: f32,
) -> Tensor<B, 2> {
    // 1. Normalize k
    // k: [B, D] -> sum_dim(1) -> [B, 1] (rank preserved)
    let k_squared = k.clone().powf_scalar(2.0);
    let k_norm_val: Tensor<B, 2> = k_squared.sum_dim(1).sqrt().add_scalar(eps);
    // k_norm_val is already [B, 1]
    let k_normalized: Tensor<B, 2> = k.div(k_norm_val);

    // 2. Compute k^T x (scalar projection per batch)
    // Element-wise multiply [B, D] * [B, D] then sum over D -> [B, 1]
    let k_t_x: Tensor<B, 2> = (k_normalized.clone().mul(x.clone())).sum_dim(1);

    // 3. Correction gamma = v - k^T x
    // [B, 1] - [B, 1]
    let gamma: Tensor<B, 2> = v.sub(k_t_x);

    // 4. Gate: beta * gamma
    // [B, 1] * [B, 1]
    let gated_gamma: Tensor<B, 2> = beta.mul(gamma);

    // 5. Update: k * (beta * gamma)
    // [B, D] * [B, 1] (broadcasts)
    let update: Tensor<B, 2> = k_normalized.mul(gated_gamma);

    // 6. Residual
    x.add(update)
}

/// Compute the Delta operator A(X) = I - β * k * k^T
pub fn delta_operator<B: Backend>(
    k: Tensor<B, 2>,    // [B, D]
    beta: Tensor<B, 2>, // [B, 1]
    d: usize,
) -> Tensor<B, 3> {
    // Returns [B, D, D]
    let device = k.device();
    let batch_size = k.dims()[0];

    // 1. Identity: [D, D] -> [1, D, D] -> [B, D, D]
    let identity: Tensor<B, 3> = Tensor::eye(d, &device)
        .unsqueeze_dim::<3>(0)
        .repeat_dim(0, batch_size);

    // 2. Outer Product: k * k^T
    // k_col: [B, D, 1]
    let k_col: Tensor<B, 3> = k.clone().unsqueeze_dim::<3>(2);
    // k_row: [B, 1, D]
    let k_row: Tensor<B, 3> = k.unsqueeze_dim::<3>(1);
    // [B, D, D]
    let k_outer = k_col.matmul(k_row);

    // 3. Beta Scaling
    // beta: [B, 1] -> [B, 1, 1]
    let beta_expanded: Tensor<B, 3> = beta.unsqueeze_dim::<3>(2);

    let scaled_outer = k_outer.mul(beta_expanded);

    // 4. A = I - (beta * k * k^T)
    identity.sub(scaled_outer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::AutoBackend;
    type TestBackend = AutoBackend;

    #[test]
    fn test_delta_update_identity_regime() {
        let device = Default::default();

        // x: [1, 3, 2]
        let x =
            Tensor::<TestBackend, 3>::from_floats([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], &device);
        // k: [1, 3]
        let k = Tensor::<TestBackend, 2>::from_floats([[1.0, 0.0, 0.0]], &device);
        // v: [1, 2]
        let v = Tensor::<TestBackend, 2>::from_floats([[7.0, 8.0]], &device);
        // beta: [1, 1]
        let beta = Tensor::<TestBackend, 2>::from_floats([[0.0001]], &device);

        let result = delta_update(x.clone(), k, v, beta, 1e-8);

        let diff = result.sub(x).abs().sum().into_scalar();
        assert!(diff < 0.01);
    }

    #[test]
    fn test_delta_update_vec_identity() {
        let device = Default::default();
        let x = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0]], &device);
        let k = Tensor::<TestBackend, 2>::from_floats([[1.0, 0.0, 0.0]], &device);
        let v = Tensor::<TestBackend, 2>::from_floats([[5.0]], &device);
        let beta = Tensor::<TestBackend, 2>::from_floats([[0.0]], &device);

        let result = delta_update_vec(x.clone(), k, v, beta, 1e-8);
        let diff = result.sub(x).abs().sum().into_scalar();
        assert!(diff < 1e-6);
    }
}
