# Contributing to MLOps Healthcare Platform

Thank you for your interest in contributing to FDA-compliant MLOps tooling!

## Getting Started

```bash
git clone https://github.com/cmangun/mlops-healthcare-platform.git
cd mlops-healthcare-platform
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
pytest
```

## How to Contribute

### Pull Request Process
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes with tests
4. Run `pytest` and `black . && ruff check .`
5. Commit using conventional commits: `feat:`, `fix:`, `docs:`
6. Open a Pull Request

### Code Style
- Black formatter (line length 88)
- Ruff linter
- Type hints required
- Google-style docstrings

## Areas We Need Help
- [ ] Additional validation protocol templates
- [ ] Integration with other ML frameworks (TensorFlow, JAX)
- [ ] Documentation and examples
- [ ] Performance benchmarks

## Questions?
Open an issue or connect on [LinkedIn](https://linkedin.com/in/christophermangun).

## License
MIT License
