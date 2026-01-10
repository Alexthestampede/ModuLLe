# Running ModuLLe on Android

ModuLLe can run on Android devices! This guide shows you how.

## Requirements

- Android device (phone or tablet)
- [Termux](https://f-droid.org/en/packages/com.termux/) app (install from F-Droid)
- Internet connection (for cloud providers or package installation)

## Installation

### 1. Install Termux

Download Termux from [F-Droid](https://f-droid.org/) (not Google Play - that version is outdated).

### 2. Setup Python Environment

```bash
# Update packages
pkg update && pkg upgrade

# Install Python and dependencies
pkg install python python-pip git

# Install build tools (needed for some dependencies)
pkg install clang make binutils

# Optional: Install rust if needed for some packages
pkg install rust
```

### 3. Install ModuLLe

```bash
# Clone the repository
git clone https://github.com/Alexthestampede/ModuLLe.git
cd ModuLLe

# Install ModuLLe
pip install -e .
```

### Alternative: Install from PyPI (when available)

```bash
pip install modulle
```

## Potential Issues and Solutions

### Issue 1: lxml Fails to Install

**Problem**: lxml requires compilation and might fail.

**Solution**: ModuLLe will automatically fall back to `html.parser`:

```python
# This is already handled in modulle/web/fetcher.py
try:
    soup = BeautifulSoup(response.content, 'lxml')
except:
    soup = BeautifulSoup(response.content, 'html.parser')
```

If you really want lxml:
```bash
pkg install libxml2 libxslt
pip install lxml
```

### Issue 2: Network Permissions

Termux should have network access by default. If you encounter issues:

```bash
# In Termux settings
termux-setup-storage

# Grant storage permission when prompted
```

### Issue 3: Storage Access

To access files outside Termux:

```bash
# Setup storage (run once)
termux-setup-storage

# Access your phone's storage
cd ~/storage/shared
```

Files will be at:
- `~/storage/shared/` - Main storage
- `~/storage/downloads/` - Download folder
- `~/storage/dcim/` - Camera photos

## Provider Support on Android

### Cloud Providers (Best Option)

✅ **OpenAI, Claude, Gemini** work perfectly on Android:

```bash
# Set API keys
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GEMINI_API_KEY="your-key"

# Run examples
python examples/autonomous_web_agent.py
```

Change the provider in the script:
```python
PROVIDER = 'openai'  # or 'claude', 'gemini'
MODEL = 'gpt-4o-mini'
```

### Local Providers

#### Ollama on Android

Ollama has experimental Android support!

**Option 1: Run Ollama directly on Android** (experimental)
- Check: https://github.com/ollama/ollama/issues/2463
- Some users have built ARM versions
- Requires more setup

**Option 2: Connect to Ollama on PC** (easier)
```bash
# On your PC, allow external connections
OLLAMA_HOST=0.0.0.0:11434 ollama serve

# On Android, set the PC's IP
export OLLAMA_BASE_URL="http://192.168.1.100:11434"

python examples/web_research.py
```

#### LM Studio
- No native Android version
- Can connect to LM Studio running on PC (same as Ollama option 2)

```bash
export LM_STUDIO_BASE_URL="http://192.168.1.100:1234"
```

## Example Usage on Android

### 1. Simple Web Research

```bash
cd ~/ModuLLe

# Edit the example to use a cloud provider
nano examples/web_research.py

# Change these lines:
PROVIDER = 'openai'
TEXT_MODEL = 'gpt-4o-mini'

# Run it
python examples/web_research.py
```

### 2. Autonomous Web Agent

```bash
# Export your API key
export OPENAI_API_KEY="your-key-here"

# Run autonomous agent
python examples/autonomous_web_agent.py
```

The agent will:
- Search the web
- Fetch pages
- Synthesize information
- Answer your question

All running on your Android device!

### 3. Custom Script

Create a simple script:

```python
#!/data/data/com.termux/files/usr/bin/python

from modulle import create_ai_client
from modulle.web import WebAccessor

# Use cloud provider
_, processor, _ = create_ai_client(
    'openai',
    text_model='gpt-4o-mini'
)

web = WebAccessor()

# Search and summarize
query = "Latest Android development news"
results = web.search_web(query, max_results=3)

content = web.fetch_page(results[0]['url'])

summary = processor.generate(
    prompt=f"Summarize: {content[:2000]}",
    temperature=0.3
)

print(summary)
```

Make it executable:
```bash
chmod +x my_script.py
./my_script.py
```

## Performance Tips

### 1. Use Lightweight Models

For cloud providers, use smaller/cheaper models:
- OpenAI: `gpt-4o-mini` instead of `gpt-4o`
- Gemini: `gemini-1.5-flash` instead of `gemini-1.5-pro`
- Claude: Consider usage costs

### 2. Limit Web Fetching

```python
# Don't fetch too many pages at once
results = web.search_web(query, max_results=3)  # Not 10
content = web.fetch_page(url)[:5000]  # Truncate large pages
```

### 3. Use WiFi

Mobile data works but can be expensive. Use WiFi when possible.

### 4. Background Tasks

Termux supports background execution:

```bash
# Install Termux:Boot (from F-Droid)
# Then setup background tasks
```

## Storage Considerations

### Config Files

Store API keys in Termux home:

```bash
# Create config file
cat > ~/.modulle_config.sh << 'EOF'
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GEMINI_API_KEY="your-key"
EOF

# Load in .bashrc
echo "source ~/.modulle_config.sh" >> ~/.bashrc
```

### Data Storage

```bash
# Store outputs in shared storage
output_dir=~/storage/shared/ModuLLe
mkdir -p $output_dir

# Save results
python examples/web_research.py > $output_dir/results.txt
```

## Keyboard Shortcuts in Termux

- **Ctrl+C**: Stop current command
- **Volume Down + C**: Ctrl key
- **Volume Down + Q**: Close session
- **Swipe from left**: Show keyboard with special keys

## Security Notes

### 1. Protect API Keys

```bash
# Never commit API keys to git
# Use environment variables or config files

# Set restrictive permissions
chmod 600 ~/.modulle_config.sh
```

### 2. Network Security

```bash
# Only connect to trusted networks
# Use HTTPS for all API calls (ModuLLe does this by default)
```

### 3. File Permissions

```bash
# Termux has its own isolated storage
# Apps can't access Termux files without permission
```

## Troubleshooting

### "Module not found" errors

```bash
# Reinstall ModuLLe
cd ~/ModuLLe
pip install -e . --force-reinstall
```

### Slow performance

```bash
# Use a lighter model
# Reduce max_results in searches
# Truncate fetched content
```

### Out of memory

```bash
# Close other apps
# Use streaming where possible
# Process smaller chunks of data
```

### Connection errors

```bash
# Check internet connection
ping google.com

# Check API keys
echo $OPENAI_API_KEY

# Try with curl
curl https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"
```

## Example: Mobile Research Assistant

Here's a complete example of a research assistant for Android:

```python
#!/data/data/com.termux/files/usr/bin/python
"""
Mobile Research Assistant

Usage:
    ./research.py "What is quantum computing?"
"""

import sys
from modulle import create_ai_client
from modulle.web import WebAccessor

def research(question):
    """Research a question using web search and AI."""

    print(f"📱 Researching: {question}\n")

    # Setup (using OpenAI for reliability on mobile)
    web = WebAccessor()
    _, processor, _ = create_ai_client('openai', text_model='gpt-4o-mini')

    # Search
    print("🔍 Searching...")
    results = web.search_web(question, max_results=3)

    if not results:
        print("No results found")
        return

    # Fetch top result
    print(f"📄 Reading: {results[0]['title']}")
    content = web.fetch_page(results[0]['url'])

    if not content:
        print("Failed to fetch content")
        return

    # Analyze
    print("🤔 Analyzing...\n")
    summary = processor.generate(
        prompt=f"Based on this article, answer: {question}\n\nArticle:\n{content[:3000]}",
        system_prompt="You are a helpful research assistant. Provide clear, concise answers.",
        temperature=0.3
    )

    print("📝 Answer:")
    print("=" * 50)
    print(summary)
    print("=" * 50)
    print(f"\n📚 Source: {results[0]['url']}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: ./research.py 'your question'")
        sys.exit(1)

    question = ' '.join(sys.argv[1:])
    research(question)
```

Save as `research.py`, make executable, and use:

```bash
chmod +x research.py
./research.py "What are the benefits of Python asyncio?"
```

## Conclusion

ModuLLe works great on Android via Termux! The cloud providers (OpenAI, Claude, Gemini) work best since they don't require local compute. You can build powerful AI-assisted tools that run right on your phone or tablet.

## Useful Links

- Termux: https://f-droid.org/en/packages/com.termux/
- Termux Wiki: https://wiki.termux.com/
- ModuLLe GitHub: https://github.com/Alexthestampede/ModuLLe
- Ollama Android: https://github.com/ollama/ollama/issues/2463
