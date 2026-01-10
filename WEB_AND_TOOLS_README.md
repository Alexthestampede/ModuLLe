# Web Access & Tool Calling - Complete Feature Set

This document summarizes the web access and tool calling capabilities added to ModuLLe.

## What Was Added

### 1. Web Access Module (`modulle/web/`)

Direct methods for web operations:
- **WebAccessor** - Main class for web access
- **WebFetcher** - HTML fetching with BeautifulSoup
- **HTMLConverter** - Clean conversion to text/markdown
- **Search** - DuckDuckGo integration (free, no API key)

**Example:**
```python
from modulle import create_ai_client
from modulle.web import WebAccessor

web = WebAccessor()
_, processor, _ = create_ai_client('ollama', text_model='cogito:3b')

# Search
results = web.search_web("Python async", max_results=5)

# Fetch
content = web.fetch_page(results[0]['url'], format='markdown')

# Analyze with AI
summary = processor.generate(prompt=f"Summarize: {content[:3000]}")
```

### 2. Tool Calling Framework (`modulle/tools/`)

Generic framework for LLM tool use:
- **BaseTool** - Abstract class for all tools
- **ToolRegistry** - Manages and executes tools
- **Provider converters** - Automatic format conversion

**Supported Providers:**
- ✅ Ollama
- ✅ OpenAI
- ✅ Claude
- ✅ Gemini
- ✅ LM Studio

### 3. Web Tools (`modulle/web/tools.py`)

Pre-built tools for autonomous web access:
- **SearchWebTool** - LLM can search the web
- **FetchPageTool** - LLM can fetch and read pages

**Example:**
```python
from modulle.web import WebAccessor
from modulle.web.tools import SearchWebTool, FetchPageTool
from modulle.tools import ToolRegistry

web = WebAccessor()
registry = ToolRegistry()
registry.register(SearchWebTool(web))
registry.register(FetchPageTool(web))

# LLM now autonomously uses tools!
response = client.chat_with_tools(
    model="PickMeAsDefault",
    messages=[{"role": "user", "content": "Research Python async"}],
    tools=registry.to_ollama_format()
)
```

### 4. Provider Tool Calling Support

All 5 providers now support tool calling:
- `modulle/providers/ollama/client.py` - `chat_with_tools()`
- `modulle/providers/openai/client.py` - `chat_with_tools()`
- `modulle/providers/claude/client.py` - `chat_with_tools()`
- `modulle/providers/gemini/client.py` - `chat_with_tools()`
- `modulle/providers/lm_studio/client.py` - `chat_with_tools()`

## Examples

### Direct Method (You Control)
`examples/web_research.py` - You explicitly call web methods

```bash
python examples/web_research.py
```

### Tool Calling (LLM Controls)
`examples/autonomous_web_agent.py` - LLM decides when to search/fetch

```bash
python examples/autonomous_web_agent.py
```

## Documentation

### User Guides
- **[docs/CUSTOM_TOOLS.md](docs/CUSTOM_TOOLS.md)** - How to create custom tools
- **[docs/ANDROID.md](docs/ANDROID.md)** - Running on Android devices

### What You Can Build

**Research Agents:**
```python
# Agent that researches topics and provides sourced answers
messages = [{"role": "user", "content": "Compare Python vs Rust"}]
# LLM autonomously: searches → fetches → analyzes → answers
```

**Custom Tools:**
```python
# Calculator, file operations, APIs, databases, etc.
class MyTool(BaseTool):
    def execute(self, **kwargs) -> str:
        # Your tool logic here
        return result
```

**Multi-Tool Agents:**
```python
registry.register(SearchWebTool(web))
registry.register(FetchPageTool(web))
registry.register(CalculatorTool())
registry.register(WeatherTool(api_key))
# LLM can use any combination of tools as needed
```

## Architecture

```
┌─────────────────────────────────────────┐
│         Your Application                │
│    "Research quantum computing"         │
└────────────────┬────────────────────────┘
                 │
        ┌────────▼─────────┐
        │  ToolRegistry    │
        │  - SearchWeb     │
        │  - FetchPage     │
        │  - Calculator    │
        │  - Weather       │
        │  - Custom...     │
        └────────┬─────────┘
                 │
    Provider-specific formatting
                 │
    ┌────────────┼────────────┐
    ▼            ▼            ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│  Ollama  │ │  OpenAI  │ │  Claude  │
│  Gemini  │ │LM Studio │ │  (etc.)  │
└──────────┘ └──────────┘ └──────────┘
    │            │            │
    └────────────┼────────────┘
                 │
        ┌────────▼─────────┐
        │  LLM Autonomy    │
        │  - Decides tools │
        │  - Executes      │
        │  - Synthesizes   │
        └──────────────────┘
```

## Key Features

### 1. Provider Agnostic
Same code works with all providers:
```python
# Just change these two lines!
PROVIDER = 'openai'  # or 'claude', 'gemini', 'ollama', 'lm_studio'
MODEL = 'gpt-4o-mini'
```

### 2. Extensible
Easy to add new tools:
```python
class MyTool(BaseTool):
    # Implement 4 methods, done!
    def get_name(self) -> str: ...
    def get_description(self) -> str: ...
    def get_parameters(self) -> Dict: ...
    def execute(self, **kwargs) -> str: ...
```

### 3. Autonomous
LLM decides when/how to use tools:
- Can search multiple times
- Can fetch multiple pages
- Adapts based on results
- Synthesizes final answer

### 4. Production Ready
- Full error handling
- Comprehensive logging
- Type hints throughout
- Security considerations
- Rate limiting support

### 5. Platform Support
- ✅ Linux
- ✅ macOS
- ✅ Windows
- ✅ Android (via Termux)

## Quick Start

### 1. Install
```bash
pip install -e .
```

### 2. Setup API Keys (for cloud providers)
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GEMINI_API_KEY="your-key"
```

### 3. Run Examples
```bash
# Direct method
python examples/web_research.py

# Autonomous agent
python examples/autonomous_web_agent.py
```

### 4. Create Custom Tool
```python
from modulle.tools.base import BaseTool

class WeatherTool(BaseTool):
    def get_name(self) -> str:
        return "get_weather"

    def get_description(self) -> str:
        return "Get current weather for a location"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }

    def execute(self, location: str) -> str:
        # Your API call here
        return f"Weather in {location}: ..."

# Register and use
registry.register(WeatherTool())
```

## Real-World Use Cases

### Research Assistant
```python
# Ask any question
# LLM searches web → fetches sources → synthesizes answer
```

### Data Analysis
```python
# Combine web data with calculations
registry.register(SearchWebTool(web))
registry.register(CalculatorTool())
# LLM can fetch data and compute statistics
```

### API Integration
```python
# Connect to any API
registry.register(WeatherTool())
registry.register(StockPriceTool())
registry.register(NewsAPITool())
# LLM coordinates multiple API calls
```

### System Automation
```python
# File operations, shell commands, etc.
registry.register(FileReadTool())
registry.register(ShellCommandTool())
# LLM can automate system tasks
```

## Performance Tips

### For Mobile/Android
- Use lightweight models (gpt-4o-mini, gemini-1.5-flash)
- Limit search results (max_results=3)
- Truncate fetched content
- Use WiFi instead of mobile data

### For Desktop
- Use faster models for better quality
- Increase max_results for thoroughness
- Enable caching (if implemented)
- Use local models (Ollama) for privacy

## Security Notes

### Web Tools
- Content is truncated to prevent context overflow
- Only HTTPS connections are used
- Rate limiting respected

### Custom Tools
- Always validate inputs
- Use whitelists for dangerous operations
- Sanitize user-provided data
- Log all tool executions

### API Keys
- Never commit to version control
- Use environment variables
- Restrict permissions on config files
- Consider using secrets management

## What's Next?

Potential future enhancements:
- Content caching for repeated fetches
- SerpAPI integration for production search
- JavaScript rendering with Playwright
- PDF and document parsing
- Image analysis tools
- Video transcription tools
- Database query tools
- Email/calendar integration

## Credits

Built with:
- ModuLLe - AI provider abstraction
- BeautifulSoup4 - HTML parsing
- html2text - Markdown conversion
- ddgs (DuckDuckGo Search) - Free web search
- requests - HTTP client

## License

Same as ModuLLe - MIT License

---

**Questions?** Check the docs:
- [CUSTOM_TOOLS.md](docs/CUSTOM_TOOLS.md) - Creating tools
- [ANDROID.md](docs/ANDROID.md) - Android deployment
- [examples/](examples/) - Working code examples
