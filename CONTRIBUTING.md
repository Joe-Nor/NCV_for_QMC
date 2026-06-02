# Contributing to RSSE Control Variates

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request:

1. Check if the issue already exists in the [issue tracker](https://github.com/Joe-Nor/NCV_for_QMC/issues)
2. If not, create a new issue with:
   - Clear description of the problem or feature
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - System information (OS, Python version, PyTorch version)
   - Relevant code snippets or error messages

### Submitting Changes

1. **Fork the repository**
   ```bash
   git clone https://github.com/Joe-Nor/NCV_for_QMC.git
   cd NCV_for_QMC
   ```

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

3. **Make your changes**
   - Write clear, documented code
   - Follow the existing code style
   - Add tests if applicable
   - Update documentation as needed

4. **Test your changes**
   ```bash
   # Run tests
   pytest test/

   # Check code style
   black python/ --check
   flake8 python/
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Clear description of changes"
   ```

6. **Push and create a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

   Then create a pull request on GitHub with:
   - Description of changes
   - Related issue numbers
   - Any breaking changes
   - Testing performed

## Code Style

### Python

- Follow [PEP 8](https://pep8.org/)
- Use [Black](https://black.readthedocs.io/) for formatting (line length: 100)
- Use type hints where appropriate
- Write docstrings for public functions/classes

Example:
```python
def compute_control_variate(
    observable: np.ndarray,
    control: np.ndarray,
    weights: np.ndarray
) -> tuple[float, float]:
    """
    Compute control variate estimate.

    Args:
        observable: Observable values
        control: Control variate values
        weights: Sample weights

    Returns:
        (cv_estimate, optimal_coefficient)
    """
    # Implementation
    pass
```

### Fortran

- Use modern Fortran (90+)
- Include comments for complex algorithms
- Use meaningful variable names
- Avoid goto statements

### Documentation

- Update README.md if adding new features
- Add docstrings to new functions
- Update relevant documentation in `docs/`
- Include examples for new functionality

## Testing

### Running Tests

```bash
# Run all tests
pytest test/

# Run specific test file
pytest test/test_parity_lib.py

# Run with coverage
pytest test/ --cov=python --cov-report=html
```

### Writing Tests

- Place tests in `test/` directory
- Name test files `test_*.py`
- Use descriptive test names
- Test edge cases and error conditions

Example:
```python
def test_parity_computation():
    """Test parity prefix computation."""
    operators = np.array([1, 2, 1, 2])
    expected_parity = np.array([0, 1, 0, 1])
    
    result = compute_parity_prefix(operators)
    
    np.testing.assert_array_equal(result, expected_parity)
```

## Development Setup

### Install Development Dependencies

```bash
pip install -e ".[dev]"
```

This installs:
- pytest (testing)
- pytest-cov (coverage)
- black (formatting)
- flake8 (linting)
- mypy (type checking)

### Pre-commit Checks

Before committing, run:

```bash
# Format code
black python/

# Check style
flake8 python/

# Run tests
pytest test/

# Type check (optional)
mypy python/
```

## Project Structure

```
rsse-control-variates/
├── python/              # Python package
│   ├── train/          # Training and CV evaluation
│   ├── analysis/       # Analysis scripts
│   └── __init__.py
├── src/                # Fortran libraries
├── fortran/            # MCMC sampler
├── test/               # Tests
├── examples/           # Example scripts
├── docs/               # Documentation
├── scripts/            # Utility scripts
└── README.md
```

## Areas for Contribution

### High Priority

- [ ] Improve documentation with more examples
- [ ] Add unit tests for core functions
- [ ] Optimize model inference speed
- [ ] Support for larger lattice sizes
- [ ] Better error messages and validation

### Medium Priority

- [ ] Add visualization tools
- [ ] Implement additional control variate methods
- [ ] Support for different lattice geometries
- [ ] Hyperparameter tuning utilities
- [ ] Distributed training support

### Low Priority

- [ ] Web interface for results visualization
- [ ] Integration with other QMC codes
- [ ] Automated hyperparameter search
- [ ] Model compression techniques

## Questions?

If you have questions about contributing:

1. Check existing documentation
2. Search closed issues
3. Open a new issue with the "question" label
4. Contact the maintainers

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Acknowledge contributions

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Publishing others' private information
- Other unprofessional conduct

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be acknowledged in:
- CONTRIBUTORS.md file
- Release notes
- Paper acknowledgments (for significant contributions)

Thank you for contributing to RSSE Control Variates!
