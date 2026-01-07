## Three Generator Branches

### **KBranch** - Reflection Direction

Implements Equation (A.1): `k̃_MLP = MLP(Pool(X))` then L2-normalize[^1]

- Pools across value dimension (mean over dv)
- Two-layer MLP: Linear → GELU → Linear
- Automatic L2 normalization to satisfy unit-vector assumption
- Output: `[batch, d]`


### **BetaBranch** - Gating Scalar

Implements Equation (A.2): `β(X) = 2 · σ(w_β^T tanh(W_in Pool(X)))`[^1]

- Pools across value dimension
- Linear → Tanh → Linear → Sigmoid → Scale by 2
- Output constrained to  for identity/projection/reflection regimes
- Output: `[batch, 1]`


### **VBranch** - Value Vector

Generates the residual value vector v ∈ ℝ^{dv}[^1]

- Pools across **feature** dimension (different from k and β)
- Two-layer MLP architecture
- Output: `[batch, dv]`


## Unified Interface

**DeltaBranches** combines all three for convenience:

- Single `forward()` call returns `(k, beta, v)` tuple
- Uses burn's `Config`/`Module` pattern for clean initialization
- Compatible with tauformer's AutoBackend system[^2]


## Test Coverage

Four unit tests verify:

1. **K normalization**: Verifies ||k||₂ ≈ 1 [^1]
2. **Beta range**: Ensures β ∈[^1]
3. **V shape**: Validates output dimensions
4. **Integration**: Tests all branches together

The implementation follows burn conventions exactly like tauformer, ready to integrate with the `delta.rs` core operators ![^3]
