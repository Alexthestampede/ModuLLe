# Adding Custom Tools to ModuLLe

ModuLLe's tool calling framework makes it easy to add new capabilities that LLMs can use autonomously.

## Quick Start

### 1. Create Your Tool

Inherit from `BaseTool` and implement 4 methods:

```python
from modulle.tools.base import BaseTool
from typing import Dict, Any

class MyTool(BaseTool):
    def get_name(self) -> str:
        """Tool name (used by LLM)."""
        return "my_tool"

    def get_description(self) -> str:
        """What the tool does (helps LLM decide when to use it)."""
        return "Does something useful when needed"

    def get_parameters(self) -> Dict[str, Any]:
        """JSON Schema for parameters."""
        return {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "Input to process"
                }
            },
            "required": ["input"]
        }

    def execute(self, input: str) -> str:
        """Execute the tool (return string for LLM)."""
        return f"Processed: {input}"
```

### 2. Register and Use

```python
from modulle.tools import ToolRegistry

registry = ToolRegistry()
registry.register(MyTool())

# LLM can now use your tool!
response = client.chat_with_tools(
    model="PickMeAsDefault",
    messages=messages,
    tools=registry.to_ollama_format()
)
```

## Real Examples

### Calculator Tool

```python
class CalculatorTool(BaseTool):
    def get_name(self) -> str:
        return "calculator"

    def get_description(self) -> str:
        return "Perform mathematical calculations"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression (e.g., '2 + 2', 'sqrt(16)')"
                }
            },
            "required": ["expression"]
        }

    def execute(self, expression: str) -> str:
        import math
        safe_dict = {'sqrt': math.sqrt, 'pi': math.pi}
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"Result: {result}"
```

### Weather API Tool

```python
class WeatherTool(BaseTool):
    def __init__(self, api_key):
        self.api_key = api_key

    def get_name(self) -> str:
        return "get_weather"

    def get_description(self) -> str:
        return "Get current weather for a location"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name (e.g., 'London', 'New York')"
                }
            },
            "required": ["location"]
        }

    def execute(self, location: str) -> str:
        import requests
        url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {'q': location, 'appid': self.api_key}
        response = requests.get(url, params=params)
        data = response.json()
        return f"Weather in {location}: {data['main']['temp']}°C"
```

### File Reader Tool (with Security)

```python
class FileReadTool(BaseTool):
    def __init__(self, allowed_dirs):
        self.allowed_dirs = allowed_dirs  # Restrict to safe directories

    def get_name(self) -> str:
        return "read_file"

    def get_description(self) -> str:
        return "Read contents of a text file"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to file to read"
                }
            },
            "required": ["file_path"]
        }

    def execute(self, file_path: str) -> str:
        import os
        # Security check
        abs_path = os.path.abspath(file_path)
        if not any(abs_path.startswith(d) for d in self.allowed_dirs):
            return f"Error: Access denied to {file_path}"

        with open(file_path, 'r') as f:
            return f.read()
```

## Best Practices

### 1. Good Descriptions

The description helps the LLM decide when to use your tool:

❌ Bad:
```python
def get_description(self) -> str:
    return "Does stuff"
```

✅ Good:
```python
def get_description(self) -> str:
    return (
        "Calculate mathematical expressions. Use this when you need to "
        "compute numbers, solve equations, or perform math operations. "
        "Supports arithmetic, trigonometry, and common math functions."
    )
```

### 2. Clear Parameter Descriptions

```python
def get_parameters(self) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query. Be specific and use relevant keywords."
                # Good: Tells LLM HOW to use the parameter
            }
        },
        "required": ["query"]
    }
```

### 3. Return Useful Information

```python
def execute(self, query: str) -> str:
    results = search(query)

    # ❌ Bad: Too little info
    return f"Found {len(results)} results"

    # ✅ Good: Detailed, actionable
    formatted = f"Search results for '{query}':\n\n"
    for i, r in enumerate(results, 1):
        formatted += f"{i}. {r['title']}\n"
        formatted += f"   URL: {r['url']}\n"
        formatted += f"   {r['snippet']}\n\n"
    return formatted
```

### 4. Security Considerations

```python
class ShellCommandTool(BaseTool):
    def __init__(self, allowed_commands=['ls', 'pwd', 'date']):
        self.allowed_commands = allowed_commands  # Whitelist only

    def execute(self, command: str) -> str:
        # Always validate before executing!
        if not any(command.startswith(allowed) for allowed in self.allowed_commands):
            return f"Error: Command not allowed"

        # ... safe execution
```

### 5. Error Handling

```python
def execute(self, url: str) -> str:
    try:
        result = fetch_data(url)
        return result
    except Exception as e:
        # Return error as string - LLM will see it and can retry
        return f"Error fetching {url}: {str(e)}"
```

## Tool Ideas

Here are some useful tools you could add:

- **File Operations**: read, write, list directory, search files
- **System Info**: disk space, CPU usage, memory, running processes
- **Database**: query SQL databases, NoSQL stores
- **APIs**: weather, news, stocks, translations, maps
- **Code Execution**: run Python, evaluate expressions
- **Image Processing**: resize, convert, analyze images
- **Email**: send emails, check inbox
- **Calendar**: check schedule, add events
- **Notes/Memory**: store and retrieve information across conversations
- **Version Control**: git operations (status, commit, log)

## Multi-Tool Agents

Combine multiple tools for powerful agents:

```python
registry = ToolRegistry()

# Research agent
registry.register(SearchWebTool(web))
registry.register(FetchPageTool(web))
registry.register(CalculatorTool())
registry.register(MemoryTool())

# Now LLM can:
# 1. Search for information
# 2. Fetch and read pages
# 3. Do calculations on data
# 4. Remember findings for later
```

## Provider Compatibility

All tools work with all providers:
- ✅ Ollama - `registry.to_ollama_format()`
- ✅ OpenAI - `registry.to_openai_format()`
- ✅ Claude - `registry.to_claude_format()`
- ✅ Gemini - `registry.to_gemini_format()`
- ✅ LM Studio - `registry.to_openai_format()`

The `BaseTool` class automatically converts your tool to the correct format!

## Testing Your Tool

```python
# Test tool directly
tool = MyTool()
result = tool.execute(input="test")
print(result)

# Test with LLM
registry = ToolRegistry()
registry.register(tool)

messages = [{
    "role": "user",
    "content": "Use my_tool to process 'hello world'"
}]

response = client.chat_with_tools(
    model="PickMeAsDefault",
    messages=messages,
    tools=registry.to_ollama_format()
)
```

## Complete Example

See `examples/custom_tools_examples.py` for working examples of:
- Calculator
- File operations
- Shell commands
- Weather API
- Database queries
- Memory/note-taking

## Need Help?

- Check existing tools: `modulle/web/tools.py`
- See examples: `examples/custom_tools_examples.py`
- Read BaseTool docs: `modulle/tools/base.py`
