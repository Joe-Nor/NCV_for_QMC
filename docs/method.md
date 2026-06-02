# Method Description

## Overview

This repository implements a neural network-based control variate method for mitigating the sign problem in quantum Monte Carlo (QMC) simulations. The approach uses autoregressive Transformer models to learn probability distributions over operator string configurations, which are then used to construct control variates that reduce the variance of sign-problem-affected observables.

## Background: The Sign Problem

In quantum Monte Carlo simulations of fermionic systems or frustrated magnets, the partition function involves a sum over configurations with both positive and negative weights:

```
Z = Σ_C w(C)
```

where `w(C)` can be negative. The sign problem arises when the average sign:

```
⟨sign⟩ = |Σ_C w(C)| / Σ_C |w(C)|
```

becomes exponentially small with system size or inverse temperature β. This leads to exponentially large statistical errors in observables.

## Control Variate Method

### Basic Principle

For an observable O, we construct a control variate f such that:

```
O_CV = O - α(f - ⟨f⟩)
```

where α is chosen to minimize Var(O_CV). The optimal coefficient is:

```
α* = Cov(O, f) / Var(f)
```

The variance reduction factor is:

```
Var(O_CV) / Var(O) = 1 - ρ²
```

where ρ is the correlation coefficient between O and f.

### Neural Network Control Variates

We use Transformer models to learn probability distributions over operator string configurations. For a configuration C represented as a sequence of operators, we model:

```
p_θ(C) = ∏_t p_θ(o_t | o_1, ..., o_{t-1})
```

where o_t is the operator at position t.

## Model Architecture

### Autoregressive Transformer

The model consists of:

1. **Embedding Layer**: Maps discrete operator types to continuous vectors
   - Operator type embedding
   - Position encoding
   - Parity prefix embedding (for symmetry)

2. **Transformer Blocks**: Stack of self-attention layers
   - Multi-head self-attention
   - Feed-forward networks
   - Layer normalization
   - Residual connections

3. **Output Head**: Predicts next operator probability
   - Linear projection to operator vocabulary
   - Softmax activation

### Parity Prefix Embedding

To incorporate symmetry information, we use parity prefix embeddings that encode:
- Current parity state (even/odd)
- Cumulative operator statistics
- Spatial symmetry information

This allows the model to respect physical symmetries and improve sample efficiency.

## Training Procedure

### Data Generation

1. **MCMC Sampling**: Run RSSE (Resummation-based Stochastic Series Expansion) sampler
   - Generate operator string configurations
   - Record weights and observables
   - Save to binary format

2. **Data Preprocessing**:
   - Convert operator strings to integer sequences
   - Compute parity prefixes
   - Create training/validation splits

### Model Training

We train two model families, and each family has separate even- and odd-parity
models. A complete energy estimate therefore uses four checkpoints:

1. **Numerator models**: `q_num,even(C)` and `q_num,odd(C)`
   - Trained on configurations in the corresponding parity sector
   - Use an observable-weighted loss; for the energy estimator this is the
     `n_h`-weighted objective, so the learned distributions approximate the
     numerator magnitude within each parity sector
   - Used to construct the numerator control variate for ⟨H·sign⟩

2. **Denominator models**: `q_den,even(C)` and `q_den,odd(C)`
   - Trained on configurations in the corresponding parity sector
   - Use the absolute-weight sampling distribution without the extra `n_h`
     numerator weight
   - Used to construct the denominator control variate for ⟨sign⟩

The even and odd models are combined as a signed difference. For a sampling
density `f(C) ∝ |w(C)|`, the zero-mean control-variate function has the form:

```
h(C) = (q_even(C) - q_odd(C)) / f(C)
```

because `E_f[h] = Σ_C (q_even(C) - q_odd(C)) = 0` when both parity models are
normalized. The implementation applies this construction separately to the
numerator and denominator model families.

Training uses:
- **Loss**: Negative log-likelihood
- **Optimizer**: AdamW with learning rate scheduling
- **Regularization**: Weight decay, dropout
- **Batch size**: Typically 32-128
- **Epochs**: Until convergence (monitored on validation set)

## Control Variate Construction

### For Energy Estimation

The ground state energy is computed as:

```
E = ⟨H·sign⟩ / ⟨sign⟩
```

We construct parity-difference control variates for both numerator and
denominator. Let `f(C) ∝ |w(C)|` be the sampling density. The implementation
uses:

```
h_num(C) = (q_num,even(C) - q_num,odd(C)) / f(C)
h_den(C) = (q_den,even(C) - q_den,odd(C)) / f(C)
```

and then forms:

```
A_CV(C) = H(C)·sign(C) - c_num h_num(C)
B_CV(C) = sign(C) - c_den h_den(C)
E_CV = ⟨A_CV⟩ / ⟨B_CV⟩
```

The coefficients `c_num` and `c_den` are estimated on training data. The joint
estimator used in `compute_energy_jackknife_Cov.py` chooses them to reduce the
variance of the final ratio, rather than fitting the numerator and denominator
completely independently.

### Jackknife Estimation

To properly propagate uncertainties through the ratio E = ⟨H·sign⟩/⟨sign⟩, we use jackknife resampling:

1. Divide data into N blocks
2. For each block i, compute:
   ```
   E_i = ⟨H·sign⟩_{-i,CV} / ⟨sign⟩_{-i,CV}
   ```
   where subscript -i means "excluding block i"
3. Estimate variance:
   ```
   Var(E) = (N-1)/N Σ_i (E_i - Ē)²
   ```

## Variance Reduction Analysis

The effectiveness of control variates is measured by:

1. **Variance Reduction Factor (VRF)**:
   ```
   VRF = Var(O_CV) / Var(O)
   ```
   Values < 1 indicate variance reduction

2. **Effective Sample Size**:
   ```
   N_eff = N / (1 + 2τ)
   ```
   where τ is the integrated autocorrelation time

3. **Correlation Coefficient**:
   ```
   ρ = Cov(O, f) / √(Var(O)·Var(f))
   ```
   Higher |ρ| indicates better control variates

## Implementation Details

### NH Window Selection

For configurations with varying operator string lengths, we use an "NH window" approach:
- Select a contiguous window of operators
- Window size chosen to balance information content and model capacity
- Multiple windows can be used for longer strings

### Data Augmentation

To improve generalization, we apply:
- Spatial symmetry transformations (rotations, reflections)
- Temporal translations (cyclic shifts of operator strings)
- Parity flips (when applicable)

### Numerical Stability

To avoid numerical issues:
- Log-probabilities are computed in log-space
- Gradient clipping during training
- Careful handling of very small/large weights

## References

For more details on the theoretical foundation and algorithmic implementation,
please refer to the accompanying paper:
[Neural Autoregressive Control Variates for the Quantum Monte Carlo Sign Problem](https://arxiv.org/abs/2605.26814).

## Next Steps

- [Data Format Specification](data_format.md)
- [Associated arXiv Paper](https://arxiv.org/abs/2605.26814)
