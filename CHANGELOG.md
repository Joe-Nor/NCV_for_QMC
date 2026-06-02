# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-05-18

### Added
- Initial public release
- Transformer-based control variate models for RSSE
- Fortran MCMC sampler with O(1) loop counting
- Autoregressive models with parity prefix embeddings
- Jackknife-based variance reduction estimation
- Support for 2×2 and 3×1 lattice geometries
- Documentation and command templates for data generation, training, and evaluation
- Guidance for archiving paper-scale data and checkpoints outside git

### Features
- **Models**
  - Numerator model (even parity configurations)
  - Denominator model (all configurations)
  - Configurable architecture (d_model, n_heads, n_layers)
  - Parity prefix embeddings for symmetry
  - Data augmentation support

- **Training**
  - AdamW optimizer with learning rate scheduling
  - Gradient clipping and weight decay
  - Validation monitoring and early stopping
  - Checkpoint saving and resuming
  - Mixed precision training support (FP16)

- **Evaluation**
  - Jackknife resampling for uncertainty quantification
  - Variance reduction factor computation
  - Correlation analysis
  - Autocorrelation time estimation

- **Data Processing**
  - Binary MCMC data reader
  - Data preprocessing and splitting
  - Parity prefix computation (Fortran library)
  - Spatial symmetry transformations

- **Visualization**
  - Energy vs temperature plots
  - Sign vs temperature plots
  - Variance reduction analysis
  - Training curve visualization
  - Log-probability vs log-frequency correlation

### Documentation
- Installation guide
- Method description
- Data format specification
- Reproducibility guide
- API documentation
- Example scripts
- Contributing guidelines

### Performance
- GPU training: ~2-4 hours per model (RTX 3090)
- Variance reduction: 60-80% for tested systems
- Inference: ~5 minutes for 100k samples (GPU)

## [Unreleased]

### Planned
- Support for larger lattice sizes (4×4, 6×6)
- Additional lattice geometries (honeycomb, kagome)
- Distributed training support
- Model compression techniques
- Web interface for visualization
- Integration with other QMC codes

### Known Issues
- Memory usage can be high for very long operator strings
- Training convergence can be slow for low temperatures
- Limited support for non-square lattices

## Version History

### Version Numbering

- **Major version** (X.0.0): Incompatible API changes
- **Minor version** (0.X.0): New features, backward compatible
- **Patch version** (0.0.X): Bug fixes, backward compatible

### Release Notes

#### [1.0.0] - 2026-05-18
First public release accompanying the arXiv paper:
"Neural Autoregressive Control Variates for the Quantum Monte Carlo Sign Problem"

Key achievements:
- Demonstrated 60-80% variance reduction on test systems
- Validated against exact diagonalization results
- Provided source code for the RSSE sampler, neural control-variate models, and jackknife evaluation
- Prepared the repository for external data and checkpoint archival

---

## How to Update This Changelog

When making changes:

1. Add entries under `[Unreleased]` section
2. Use categories: Added, Changed, Deprecated, Removed, Fixed, Security
3. Keep entries concise and user-focused
4. Link to relevant issues/PRs when applicable

Example:
```markdown
### Added
- New feature X for doing Y (#123)

### Fixed
- Bug in Z that caused W (#456)
```

When releasing a new version:
1. Move `[Unreleased]` entries to new version section
2. Add release date
3. Update version links at bottom
4. Create git tag: `git tag -a v1.0.0 -m "Release v1.0.0"`
