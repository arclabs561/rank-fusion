# Documentation Inspiration & Best Practices

References from well-known Rust and Python projects for documentation practices.

## Rust Projects

### Tokio (`tokio-rs/tokio`)
- **Doctest coverage**: All examples in doc comments are tested
- **Separate doc CI job**: Isolates doc issues from code
- **Comprehensive examples**: Real-world async patterns
- **Link**: https://github.com/tokio-rs/tokio

### Serde (`serde-rs/serde`)
- **Extensive inline examples**: Every public function has working examples
- **API documentation**: Clear, concise function signatures
- **Link**: https://github.com/serde-rs/serde

### PyO3 (`PyO3/pyo3`)
- **Python bindings documentation**: Excellent examples for PyO3 usage
- **Type stub validation**: Ensures `.pyi` files match implementation
- **Multi-language docs**: Clear separation between Rust and Python APIs
- **Link**: https://github.com/PyO3/pyo3

## Python Projects

### Pydantic (`pydantic/pydantic`)
- **Type validation examples**: Clear examples of validation patterns
- **API reference**: Comprehensive, well-organized
- **Link**: https://github.com/pydantic/pydantic

### FastAPI (`tiangolo/fastapi`)
- **Tutorial-style docs**: Step-by-step guides
- **Code examples**: All examples are runnable
- **Link**: https://github.com/tiangolo/fastapi

## Key Practices to Adopt

1. **Test all doc examples**: `cargo test --doc` ensures examples stay valid
2. **Link validation**: Prevents broken links in production
3. **Type stub checking**: Python bindings match Rust API
4. **Separate doc CI**: Isolates doc issues from code issues
5. **Real-world examples**: Show actual usage patterns, not toy examples

## Tools Used

- **`cargo test --doc`**: Tests Rust documentation examples
- **`markdown-link-check`**: Validates markdown links
- **`mypy`**: Validates Python type stubs
- **`cargo doc`**: Generates and validates Rust documentation

