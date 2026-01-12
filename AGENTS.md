# ROMA-DSPy Development Guide for Agentic Coding Agents

This file provides essential guidelines and commands for agentic coding agents working in the ROMA-DSPy repository.

## Build/Lint/Test Commands

### Environment Setup
```bash
# Use virtual environment (.venv) or uv run for all commands
uv pip install -e ".[dev]"  # Install with dev dependencies

# Alternative: Just use uv run for all commands
uv run pytest tests/
```

### Testing Commands
```bash
# Run all tests
uv run pytest tests/
just test

# Run specific test file
uv run pytest tests/unit/test_toolkits.py
just test-file test_toolkits.py

# Run single test
uv run pytest tests/unit/test_toolkits.py::TestBaseToolkit::test_toolkit_initialization -v

# Run tests with coverage
uv run pytest tests/ --cov=src/roma_dspy --cov-report=html --cov-report=term
just test-coverage

# Run unit tests only
uv run pytest tests/unit/ -v
just test-unit

# Run integration tests only
uv run pytest tests/integration/ -v
just test-integration

# Run tests in verbose mode
uv run pytest tests/ -v
just test-verbose
```

### Linting and Formatting
```bash
# Run linting (ruff)
uv run ruff check src/ tests/
just lint

# Format code (ruff format)
uv run ruff format src/ tests/
just format

# Type checking (mypy)
uv run mypy src/roma_dspy
just typecheck

# Run all pre-commit checks
just pre-commit  # Runs format, lint, typecheck, test
```

### Build and Package
```bash
# Build package
uv run python -m build
just build

# Clean cache and build artifacts
just clean
```

## Code Style Guidelines

### Import Style
- Use `from __future__ import annotations` at the top of all Python files
- Group imports in this order: stdlib, third-party, local imports
- Use `isort`-compatible grouping (ruff format handles this automatically)
- Avoid wildcard imports (`from module import *`)
- Use TYPE_CHECKING for imports that are only needed for type hints:

```python
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import dspy
from loguru import logger

if TYPE_CHECKING:
    from roma_dspy.config.schemas.agents import AgentConfig
```

### Type Annotations
- Use strict type annotations for all function signatures
- Prefer `Union[str, int]` over `str | int` for compatibility with Python 3.12+
- Use `Optional[T]` for nullable types
- Use `Literal` and `Enum` for constrained values
- Always annotate class attributes and return types

```python
def process_data(
    input_data: Dict[str, Any],
    config: Optional[Config] = None,
    *,
    force: bool = False,
) -> Result[str]:
    """Process input data with optional configuration."""
    pass
```

### Naming Conventions
- **Classes**: `PascalCase` (e.g., `BaseModule`, `ToolkitManager`)
- **Functions/Methods**: `snake_case` (e.g., `process_data`, `get_toolkit`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_RETRIES`, `DEFAULT_TIMEOUT`)
- **Private methods**: Prefix with underscore (`_internal_method`)
- **Type aliases**: `PascalCase` with descriptive names (e.g., `PredictionStrategy`)

### Error Handling
- Use structured error handling with specific exception types
- Create custom exception classes for domain-specific errors
- Use `loguru` for logging with appropriate levels
- Include context in error messages but avoid sensitive data

```python
from loguru import logger

class ToolkitError(Exception):
    """Base exception for toolkit-related errors."""
    pass

def execute_tool(tool_name: str, args: Dict[str, Any]) -> Any:
    try:
        return toolkit_manager.execute(tool_name, args)
    except ToolNotFoundError as e:
        logger.error(f"Tool not found: {tool_name}")
        raise ToolkitError(f"Unknown tool: {tool_name}") from e
```

### Code Organization
- Follow the existing module structure: `src/roma_dspy/`
- Use `BaseModule` as the base class for all DSPy modules
- Implement both sync (`forward`) and async (`aforward`) methods
- Use Pydantic models for configuration and data validation
- Keep classes focused on single responsibilities

### Configuration and Settings
- Use Pydantic models for configuration validation
- Support both dictionary and environment-based configuration
- Use OmegaConf for layered configuration management
- Provide sensible defaults for all configuration options

```python
from pydantic import Field
from pydantic_settings import BaseSettings

class AgentConfig(BaseSettings):
    """Configuration for ROMA agents."""
    
    name: str = Field(default="default", description="Agent name")
    max_depth: int = Field(default=5, ge=1, le=10)
    timeout: int = Field(default=300, ge=1)
    
    class Config:
        env_prefix = "ROMA_"
```

### Testing Guidelines
- Use descriptive test method names that explain what is being tested
- Follow the AAA pattern: Arrange, Act, Assert
- Use fixtures for common test setup
- Mock external dependencies using `unittest.mock`
- Test both success and failure scenarios

```python
class TestToolkitManager:
    """Test toolkit management functionality."""
    
    def test_toolkit_registration_success(self):
        """Test that toolkits are registered correctly."""
        # Arrange
        manager = ToolkitManager()
        toolkit = MockToolkit(enabled=True)
        
        # Act
        manager.register_toolkit(toolkit)
        
        # Assert
        assert manager.has_toolkit("MockToolkit")
        assert manager.get_toolkit("MockToolkit") is toolkit
```

### Documentation
- Use docstrings for all public classes, methods, and functions
- Follow Google-style docstring format
- Include type hints in docstrings
- Document configuration options and their defaults
- Use examples for complex functionality

```python
def solve(
    task: str,
    profile: str = "general",
    max_depth: int = 5,
) -> str:
    """Solve a task using ROMA-DSPy framework.
    
    Args:
        task: The task description to solve
        profile: Configuration profile name (default: "general")
        max_depth: Maximum recursion depth for task decomposition
        
    Returns:
        The solution result as a string
        
    Raises:
        ValueError: If task is empty or profile is invalid
        
    Example:
        >>> solve("What is 2+2?")
        '4'
    """
    pass
```

### Performance Guidelines
- Use async/await for I/O-bound operations
- Implement caching for expensive operations
- Use connection pooling for database operations
- Monitor memory usage for large datasets
- Use DSPy caching for LLM calls

### Security Guidelines
- Never commit API keys or secrets to the repository
- Use environment variables for sensitive configuration
- Validate all user inputs and configuration
- Sanitize outputs before logging
- Follow principle of least privilege for toolkit permissions

## Repository Structure Notes

### Key Directories
- `src/roma_dspy/core/`: Core framework modules (BaseModule, signatures)
- `src/roma_dspy/tools/`: Toolkit implementations
- `src/roma_dspy/config/`: Configuration management
- `src/roma_dspy/api/`: FastAPI REST API
- `tests/unit/`: Unit tests
- `tests/integration/`: Integration tests
- `config/profiles/`: Configuration profiles

### Important Files
- `pyproject.toml`: Project configuration and dependencies
- `justfile`: Development command shortcuts
- `conftest.py`: Pytest configuration and fixtures
- `.env.example`: Environment variable template

### Development Workflow
1. Always run `just pre-commit` before committing
2. Use `uv run` for all Python commands to ensure isolation
3. Write tests for new functionality
4. Update documentation for API changes
5. Follow semantic versioning for releases

## Common Patterns

### Module Implementation
```python
from __future__ import annotations

import dspy
from typing import Any, Optional

from roma_dspy.core.modules.base_module import BaseModule
from roma_dspy.types import PredictionStrategy

class CustomModule(BaseModule):
    """Custom ROMA module implementing specific functionality."""
    
    def __init__(
        self,
        prediction_strategy: Union[PredictionStrategy, str] = PredictionStrategy.CHAIN_OF_THOUGHT,
        *,
        lm: Optional[dspy.LM] = None,
        model: Optional[str] = None,
        model_config: Optional[Mapping[str, Any]] = None,
        tools: Optional[Union[Sequence[Any], TMapping[str, Any]]] = None,
        **strategy_kwargs: Any,
    ) -> None:
        super().__init__(
            prediction_strategy=prediction_strategy,
            lm=lm,
            model=model,
            model_config=model_config,
            tools=tools,
            **strategy_kwargs,
        )
    
    def forward(self, input_data: str) -> Any:
        """Process input data using the configured prediction strategy."""
        # Implementation here
        pass
    
    async def aforward(self, input_data: str) -> Any:
        """Async version of forward method."""
        # Implementation here
        pass
```

### Toolkit Implementation
```python
from __future__ import annotations

from roma_dspy.tools.base.base import BaseToolkit
from roma_dspy.config.schemas.toolkit import ToolkitConfig

class CustomToolkit(BaseToolkit):
    """Custom toolkit implementing specific tools."""
    
    def _setup_dependencies(self):
        """Setup external dependencies."""
        pass
    
    def _initialize_tools(self):
        """Initialize toolkit tools."""
        pass
    
    def custom_tool(self, param: str) -> str:
        """Custom tool implementation."""
        return f"Processed: {param}"
```

This guide provides the essential information for working effectively with the ROMA-DSPy codebase. Always refer to existing code examples and tests for specific implementation details.