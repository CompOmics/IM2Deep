# Contributing to IM2Deep

We welcome contributions to IM2Deep! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment
4. Create a feature branch
5. Make your changes
6. Run tests
7. Submit a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/IM2Deep.git
cd IM2Deep

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .[dev,test]
```

## Code Standards

### Style Guide
- Follow PEP 8
- Use Black for code formatting: `black im2deep/`
- Use isort for imports: `isort im2deep/`
- Maximum line length: 99 characters

### Documentation
- Use NumPy-style docstrings
- Include type hints
- Provide examples in docstrings
- Update documentation for new features

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Add tests for new functionality
4. Update documentation
5. Ensure all tests pass
6. Update CHANGELOG.md
7. Submit pull request

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help others learn and grow

## Questions?

Feel free to open an issue for questions or discussion!
