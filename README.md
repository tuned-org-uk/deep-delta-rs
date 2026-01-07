# deep-delta-learn

Rust + [Burn](https://burn.dev) implementation of **Deep Delta Learning** (DDL) from the paper *"Deep Delta Learning"* (arXiv:2601.00417v1). [file:1]

This repository provides:
- Core Delta operator (`delta_update`) for matrix-valued states.
- Generator branches for \(k(X)\), \(\beta(X)\), and \(v(X)\).
- A `DeltaResidual` block (Delta-Res) that wraps branches + the Delta update.

## What is Delta-Res?

The Delta-Res update is a rank-1 residual transformation:

- Input state: \(X \in \mathbb{R}^{B \times D \times V}\)
- Parameters from branches: \(k \in \mathbb{R}^{B \times D}\), \(\beta \in \mathbb{R}^{B \times 1}\), \(v \in \mathbb{R}^{B \times V}\)
- Update: \(X_{l+1} = X_l + \beta(X_l) k(X_l) (v(X_l)^T - k(X_l)^T X_l)\) [file:1]

## Shape conventions (Burn)

Burn tracks tensor rank at the type level (`Tensor<B, const D: usize, ...>`), so reductions often behave differently than PyTorch.

Conventions used in this crate:
- Hidden state is always `Tensor<B, 3>` with shape `[B, D, V]`.
- Branch outputs are always rank-2:
  - `k(X)`: `[B, D]`
  - `beta(X)`: `[B, 1]`
  - `v(X)`: `[B, V]`

Important: operations such as `mean_dim` are typically **rank-preserving** (e.g. `[B, D, V] -> [B, D, 1]`), so we explicitly remove singleton dimensions using `squeeze::<2>()` to produce rank-2 tensors. [web:0]

## Build and run

CPU (default):

```bash
cargo run --release
```

WGPU (cross-platform GPU):

```bash
cargo run --release --features wgpu
```

CUDA (NVIDIA GPU):

```bash
cargo run --release --features cuda
```

Run tests:

```bash
cargo test
```

## Repository layout

- `src/delta.rs`: core Delta operators.
- `src/branches.rs`: generator branches for \(k, \beta, v\).
- `src/nn.rs`: `DeltaResidual` and helper blocks.
- `src/backend.rs`: backend selection helper for Burn 0.18.
- `src/main.rs`: simple smoke test binary.

## License

Apache-2.0.
