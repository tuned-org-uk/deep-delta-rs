# deep-delta-learn

Rust + [Burn](https://burn.dev) implementation of **Deep Delta Learning** (DDL) from the paper *"Deep Delta Learning"* (arXiv:2601.00417v1).

This repository provides:
- Core Delta operator (`delta_update`) for matrix-valued states.
- Generator branches for $(k(X)\), \(\beta(X))$, and $(v(X))$.
- A `DeltaResidual` block (Delta-Res) that wraps branches + the Delta update.

## What is Delta-Res?

The Delta-Res update is a rank-1 residual transformation:

- Input state: $(X \in \mathbb{R}^{B \times D \times V})$
- Parameters from branches: $(k \in \mathbb{R}^{B \times D}\), \(\beta \in \mathbb{R}^{B \times 1}\), \(v \in \mathbb{R}^{B \times V})$
- Update: $(X_{l+1} = X_l + \beta(X_l) k(X_l) (v(X_l)^T - k(X_l)^T X_l))$

## Shape conventions (Burn)

Burn tracks tensor rank at the type level (`Tensor<B, const D: usize, ...>`), so reductions behave differently than in PyTorch.

**Critical conventions in this crate:**
- Hidden state is always `Tensor<B, 3>` with shape `[B, D, V]`.
- Branch outputs are always rank-2: `k` is `[B, D]`, `beta` is `[B, 1]`, `v` is `[B, V]`.

**Rank-preserving reductions:**
Operations like `mean_dim` and `sum_dim` are **rank-preserving** in Burn (e.g., `sum_dim(1)` on `[B, D]` yields `[B, 1]`, not `[B]`).
- In `branches.rs` (pooling), we use `squeeze::<2>()` to explicitly drop singleton dimensions and return rank-2 tensors.
- In `delta.rs`, we leverage the preserved rank (e.g., `[B, 1]`) for correct broadcasting without needing extra `unsqueeze` calls.

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

- `src/delta.rs`: core Delta operators (Eq 2.5).
- `src/branches.rs`: generator branches for $k, \beta, v$.
- `src/nn.rs`: `DeltaResidual` and helper blocks.
- `src/backend.rs`: backend selection helper for Burn 0.18.
- `src/main.rs`: simple smoke test binary.
