# Local Interpreter and Code Execution in ROMA-DSPy

**Date**: January 12, 2026  
**Analysis**: Comprehensive review of code execution capabilities and setup options in ROMA-DSPy

## Executive Summary

**ROMA-DSPy provides easy-to-use real code execution capabilities** through multiple toolkits, ranging from secure cloud-based sandboxes to local subprocess execution. The framework is designed to enable agents to execute Python code and shell commands safely and efficiently.

## Available Code Execution Methods

### 1. E2B Toolkit (Recommended - Cloud-based Sandbox)

**Location**: `src/roma_dspy/tools/core/e2b.py`

#### Overview
The E2B Toolkit provides secure sandboxed code execution using E2B's cloud infrastructure. It's the **recommended approach** for production use cases.

#### Key Features
- ✅ **Secure sandboxed execution** (isolated cloud environment)
- ✅ **Async support** (non-blocking, event-loop safe)
- ✅ **Auto-cleanup** (24-hour limit handling with preemptive restart)
- ✅ **Package installation** via `run_command()`
- ✅ **File persistence** (optional S3 integration)
- ✅ **Production ready** (designed for real workloads)
- ✅ **Zero event loop blocking** (uses AsyncSandbox)
- ✅ **Comprehensive logging** (execution artifacts tracked)

#### Setup Requirements
```bash
# 1. Install dependencies
uv pip install roma-dspy[e2b]

# 2. Set API key
export E2B_API_KEY="your_e2b_api_key"

# 3. Configure in ROMA (optional)
# In config/profiles/*.yaml:
agents:
  executor:
    toolkits:
      - class_name: "E2BToolkit"
        enabled: true
        toolkit_config:
          timeout: 600  # 10 minutes
          template: "roma-dspy-sandbox-dev"
          max_lifetime_hours: 23.5
          auto_reinitialize: true
```

#### API Methods
```python
async def run_python_code(self, code: str) -> str:
    """Execute Python code in secure E2B sandbox."""
    
async def run_command(self, command: str, timeout_seconds: int = 60) -> str:
    """Execute shell commands in sandbox environment."""
```

#### Usage Examples
```python
from roma_dspy.tools.core.e2b import E2BToolkit

# Initialize toolkit
toolkit = E2BToolkit(api_key="your_e2b_api_key")

# Basic Python execution
result = await toolkit.run_python_code("""
import pandas as pd
import numpy as np

# Create and analyze data
data = pd.DataFrame({
    'values': np.random.normal(0, 1, 1000)
})
print(f"Mean: {data.values.mean():.3f}")
print(f"Std: {data.values.std():.3f}")
""")

# Install packages and use them
await toolkit.run_command("pip install matplotlib scikit-learn")

analysis_code = """
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load and analyze data
data = load_iris()
model = RandomForestClassifier()
model.fit(data.data, data.target)

# Create visualization
importances = model.feature_importances_
plt.bar(range(len(importances)), importances)
plt.title('Feature Importances')
plt.savefig('/tmp/feature_importance.png')
print('Analysis complete - plot saved')
"""

result = await toolkit.run_python_code(analysis_code)
```

#### Response Format
```json
{
    "success": true,
    "results": ["Mean: 0.012", "Std: 1.023", "Analysis complete..."],
    "stdout": ["Execution output lines"],
    "stderr": ["Error lines if any"],
    "error": null,
    "sandbox_id": "sandbox-abc123"
}
```

### 2. Subprocess Toolkit (Local Execution)

**Location**: `src/roma_dspy/tools/terminal/subprocess_toolkit.py`

#### Overview
Pure subprocess-based execution with no external dependencies. **Designed for trusted environments only** due to security implications.

#### Security Warning
⚠️ **CRITICAL SECURITY WARNING** ⚠️
This toolkit executes arbitrary shell commands and Python code with the same privileges as the Python process.

**Security Risks:**
- **Command Injection**: Shell metacharacters (`;`, `&&`, `|`, `$`) enable arbitrary command execution
- **Code Execution**: `execute_python()` runs arbitrary Python code  
- **Privilege Escalation**: Commands run with process privileges (potentially root in containers)
- **Environment Leakage**: All environment variables (including secrets) are passed to subprocesses
- **Resource Exhaustion**: No built-in limits on CPU, memory, or process count

#### Safe Usage Guidelines
- ✅ Only use with trusted input or in sandboxed environments
- ✅ Validate and sanitize all input before passing to commands
- ✅ Run with minimal required privileges
- ✅ Use resource limits (containers, cgroups) to prevent DoS
- ✅ Filter environment variables to prevent secret leakage

#### Setup Requirements
```bash
# No additional dependencies needed
# Available by default in ROMA-DSPy
```

#### API Methods
```python
async def execute_python(self, code: str) -> str:
    """Execute Python code in local subprocess."""
    
async def execute_command(self, command: str, timeout_seconds: int = 60) -> str:
    """Execute shell command in local subprocess."""
```

#### Usage Examples
```python
from roma_dspy.tools.terminal.subprocess_toolkit import SubprocessTerminalToolkit
from roma_dspy.core.storage.file_storage import FileStorage

# ⚠️ Only use with trusted input!
toolkit = SubprocessTerminalToolkit(file_storage=file_storage)

# Basic Python execution
result = await toolkit.execute_python("print('Hello local execution!')")

# Package management
await toolkit.execute_command("pip install requests numpy pandas")

# Data analysis
analysis_code = """
import pandas as pd
import numpy as np

# Create sample data
np.random.seed(42)
data = pd.DataFrame({
    'x': np.random.randn(1000),
    'y': np.random.randn(1000)
})

# Calculate correlation
correlation = data['x'].corr(data['y'])
print(f"Correlation coefficient: {correlation:.4f}")
"""

result = await toolkit.execute_python(analysis_code)
print(f"Analysis result: {result}")
```

#### Best Practices
```python
# ✅ CORRECT: Separate package installation and code execution
await toolkit.execute_command("pip install requests numpy")
if toolkit.last_returncode != 0:
    raise RuntimeError(f"Package installation failed")

# ✅ THEN: Execute your logic
result = await toolkit.execute_python("""
import requests
import numpy as np
# Your analysis code here
""")

# ❌ WRONG: Don't mix package installation with code execution
result = await toolkit.execute_python("""
import subprocess
subprocess.check_call(['pip', 'install', 'requests'])  # Will fail!
""")

# ❌ DANGEROUS: Never pass unsanitized user input
user_input = request.get_param("file")  # Could be "; rm -rf /"
result = await toolkit.execute_command(f"cat {user_input}")  # COMMAND INJECTION!
```

### 3. Terminal Toolkit (Tmux-based)

**Location**: `src/roma_dspy/tools/terminal/toolkit.py`

#### Overview
Terminal interaction toolkit designed for **Terminal-Bench evaluation framework**. Provides tmux-based terminal sessions with comprehensive FileStorage integration.

#### Use Cases
- **Benchmark evaluation** (Terminal-Bench framework)
- **Interactive terminal sessions**
- **Structured command logging**
- **Screenshot capture** (optional)

#### Setup Requirements
```python
# Requires terminal-bench for full functionality
pip install terminal-bench

# Falls back to built-in tmux implementation
```

## Security Comparison

| Aspect | E2B Toolkit | Subprocess Toolkit | Terminal Toolkit |
|---------|---------------|-------------------|------------------|
| **Security Level** | ✅ High (isolated sandbox) | ❌ Low (same process) | ⚠️ Medium (tmux isolation) |
| **Isolation** | ✅ Cloud-based isolation | ❌ No isolation | ⚠️ Process isolation |
| **Setup Complexity** | ⭐⭐ Medium | ⭐ Easy | ⭐⭐⭐ Complex |
| **Performance** | ⭐⭐⭐ Good (network latency) | ⭐⭐⭐⭐ Excellent (local) | ⭐⭐⭐ Good (local) |
| **Dependencies** | E2B API key | None | terminal-bench (optional) |
| **Best For** | Production, untrusted code | Trusted environments, development | Terminal-Bench evaluation |

## Integration with ROMA Agents

### Executor Configuration
```python
from roma_dspy import Executor, E2BToolkit

# Add code execution to agent
executor = Executor(
    lm=dspy.LM("openrouter/openai/gpt-4o-mini"),
    tools=[E2BToolkit()],  # Agent can execute code
    prediction_strategy="react",  # Agent decides when to use tools
    context_defaults={"track_usage": True},
)

# Agent can now execute code automatically
task = "Analyze this financial dataset and create predictive model"
result = executor.forward(task)
# Agent will use E2B toolkit when code execution is needed
```

### Tool Registration Pattern
```python
# In agent configuration profiles
agents:
  executor:
    toolkits:
      - class_name: "E2BToolkit"
        enabled: true
        toolkit_config:
          api_key: "${E2B_API_KEY}"
          timeout: 600
          
      - class_name: "FileToolkit"
        enabled: true
        
      - class_name: "CalculatorToolkit"  
        enabled: true
```

## Advanced Usage Patterns

### Data Science Workflow
```python
async def data_analysis_pipeline():
    toolkit = E2BToolkit()
    
    # Step 1: Install required packages
    packages = [
        "pip install pandas numpy scikit-learn matplotlib seaborn",
        "pip install yfinance",  # Financial data
    ]
    
    for cmd in packages:
        result = await toolkit.run_command(cmd)
        if not json.loads(result)["success"]:
            raise RuntimeError(f"Package installation failed: {cmd}")
    
    # Step 2: Data acquisition and analysis
    analysis_code = """
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Get stock data
data = yf.download('AAPL', start='2020-01-01', end='2023-12-31')
data['Returns'] = data['Close'].pct_change()
data.dropna(inplace=True)

# Feature engineering
data['MA_10'] = data['Close'].rolling(10).mean()
data['Volatility'] = data['Returns'].rolling(20).std()

# Prepare features
features = ['MA_10', 'Volatility', 'Volume']
X = data[features].fillna(0)
y = (data['Returns'] > 0).astype(int)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate and save results
predictions = model.predict(X_test)
accuracy = (predictions > 0.5).mean()
print(f"Direction accuracy: {accuracy:.3f}")

# Create visualization
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Price')
plt.plot(data.index, data['MA_10'], label='MA-10')
plt.title('Stock Price Analysis')
plt.legend()
plt.savefig('/tmp/stock_analysis.png')
print('Analysis complete - visualization saved')
"""
    
    # Step 3: Execute analysis
    result = await toolkit.run_python_code(analysis_code)
    return json.loads(result)
```

### Machine Learning Training
```python
async def ml_training_pipeline():
    toolkit = E2BToolkit()
    
    # Install ML libraries
    await toolkit.run_command("pip install torch torchvision")
    
    # Training script
    training_code = """
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Generate dummy data
X = torch.randn(1000, 784)
y = torch.randint(0, 10, (1000,))
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = 0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

print("Training complete!")
"""
    
    result = await toolkit.run_python_code(training_code)
    return result
```

## Error Handling and Logging

### E2B Error Handling
```python
async def safe_code_execution(code: str, max_retries: int = 3):
    toolkit = E2BToolkit()
    
    for attempt in range(max_retries):
        try:
            result = await toolkit.run_python_code(code)
            parsed = json.loads(result)
            
            if parsed["success"]:
                return parsed["results"]
            else:
                error = parsed.get("error", "Unknown error")
                print(f"Attempt {attempt+1} failed: {error}")
                
        except Exception as e:
            print(f"Attempt {attempt+1} exception: {e}")
            
        if attempt < max_retries - 1:
            # Wait before retry
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    raise RuntimeError(f"Code execution failed after {max_retries} attempts")
```

### Subprocess Error Handling
```python
async def safe_subprocess_execution(code: str):
    toolkit = SubprocessTerminalToolkit(file_storage=file_storage)
    
    try:
        result = await toolkit.execute_python(code)
        
        if toolkit.last_returncode != 0:
            error_msg = f"Execution failed with return code: {toolkit.last_returncode}"
            print(error_msg)
            raise RuntimeError(error_msg)
            
        return result
        
    except Exception as e:
        print(f"Subprocess execution exception: {e}")
        raise
```

## Performance Considerations

### E2B Toolkit Performance
- **Network latency**: ~100-500ms round trip
- **Sandbox startup**: ~2-5 seconds for cold start
- **Memory limits**: Configurable (default ~2GB)
- **CPU limits**: Configurable (default ~2 cores)
- **Timeout handling**: Automatic with retry

### Subprocess Toolkit Performance
- **No network overhead**: Direct local execution
- **Instant startup**: No sandbox initialization
- **Full resources**: Access to full machine capabilities
- **Risk considerations**: Security responsibilities

## Best Practices Summary

### For E2B Toolkit (Production)
1. **Use environment variables** for API keys
2. **Set reasonable timeouts** for your workload
3. **Monitor sandbox usage** to optimize resource allocation
4. **Implement retry logic** for network issues
5. **Log execution results** for debugging

### For Subprocess Toolkit (Development)
1. **Never use with untrusted input**
2. **Validate and sanitize** all commands
3. **Use resource limits** (containers, cgroups)
4. **Filter environment variables** to prevent secret leakage
5. **Consider E2B** for any production use case

### For Agent Integration
1. **Configure appropriate tools** for your use case
2. **Use ReAct/CodeAct strategies** to enable tool usage
3. **Monitor tool execution** costs and performance
4. **Implement error handling** for tool failures
5. **Log all tool interactions** for observability

## Conclusion

ROMA-DSPy provides **comprehensive code execution capabilities** that are easy to integrate into agent workflows. The E2B Toolkit offers the best balance of security and functionality for production use cases, while the Subprocess Toolkit provides maximum performance for trusted development environments.

**Key Recommendations:**
- **Production**: Use E2B Toolkit for secure, scalable execution
- **Development**: Use Subprocess Toolkit with proper security measures  
- **Evaluation**: Use Terminal Toolkit for Terminal-Bench compatibility
- **Always**: Implement proper error handling and logging

The framework is designed to make code execution a first-class capability for AI agents, with production-ready security and observability features.

---

**Files Analyzed**:
- `src/roma_dspy/tools/core/e2b.py`
- `src/roma_dspy/tools/terminal/subprocess_toolkit.py`
- `src/roma_dspy/tools/terminal/toolkit.py`
- `src/roma_dspy/utils/lazy_imports.py`
- `pyproject.toml` (dependencies)

**Analysis Date**: January 12, 2026