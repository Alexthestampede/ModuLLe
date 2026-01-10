#!/usr/bin/env python3
"""
Web Research Example for ModuLLe

This example demonstrates how to use ModuLLe's web access module to:
1. Search the web for information
2. Fetch page content
3. Use AI to analyze and summarize the content

The example works with any ModuLLe provider (Ollama, OpenAI, Gemini, Claude, LM Studio).
"""

import sys
import os
from modulle import create_ai_client
from modulle.web import WebAccessor


def print_separator():
    """Print a visual separator."""
    print("\n" + "=" * 80 + "\n")


def main():
    """Run web research example."""
    print("ModuLLe Web Research Example")
    print_separator()

    # Configuration - Change these to match your setup
    PROVIDER = 'ollama'  # Options: 'ollama', 'openai', 'gemini', 'claude', 'lm_studio'
    TEXT_MODEL = 'llama2'  # Change based on your provider

    # For cloud providers, you'll need API keys:
    # export OPENAI_API_KEY="your-key"
    # export GEMINI_API_KEY="your-key"
    # export ANTHROPIC_API_KEY="your-key"

    # Initialize web accessor
    print(f"Initializing web accessor...")
    web = WebAccessor()

    # Initialize AI client
    print(f"Connecting to {PROVIDER} with model {TEXT_MODEL}...")
    try:
        _, processor, _ = create_ai_client(PROVIDER, text_model=TEXT_MODEL)
        print("✓ Connected successfully")
    except Exception as e:
        print(f"✗ Failed to connect to AI provider: {e}")
        print("\nMake sure your provider is running and configured correctly.")
        print("For Ollama: ollama serve")
        print("For cloud providers: Set appropriate API key environment variables")
        return 1

    print_separator()

    # Step 1: Search the web
    query = "Python async programming best practices"
    print(f"Searching for: '{query}'")
    print("This may take a few seconds...")

    try:
        results = web.search_web(query, max_results=5)
    except Exception as e:
        print(f"✗ Search failed: {e}")
        return 1

    if not results:
        print("✗ No results found")
        return 1

    print(f"✓ Found {len(results)} results:\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']}")
        print(f"   URL: {result['url']}")
        print(f"   Snippet: {result['snippet'][:100]}...")
        print()

    print_separator()

    # Step 2: Fetch content from top result
    top_result = results[0]
    print(f"Fetching content from: {top_result['title']}")
    print(f"URL: {top_result['url']}")

    try:
        content = web.fetch_page(top_result['url'], format='text')
    except Exception as e:
        print(f"✗ Failed to fetch page: {e}")
        return 1

    if not content:
        print("✗ Failed to retrieve content")
        return 1

    # Truncate content for LLM context
    max_content = 3000
    if len(content) > max_content:
        content = content[:max_content]
        print(f"✓ Retrieved content (truncated to {max_content} characters)")
    else:
        print(f"✓ Retrieved content ({len(content)} characters)")

    print_separator()

    # Step 3: Use AI to analyze the content
    print("Asking AI to summarize the content...")
    print("This may take a few seconds...\n")

    system_prompt = """You are a technical content analyst. Your task is to:
1. Summarize the main points of the article
2. Extract key takeaways
3. Present information clearly and concisely"""

    user_prompt = f"""Please analyze this article about Python async programming:

Title: {top_result['title']}
URL: {top_result['url']}

Content:
{content}

Provide a clear summary with the main points and key takeaways."""

    try:
        summary = processor.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.3
        )
    except Exception as e:
        print(f"✗ AI analysis failed: {e}")
        return 1

    if not summary:
        print("✗ Failed to generate summary")
        return 1

    print("AI Summary:")
    print("-" * 80)
    print(summary)
    print("-" * 80)

    print_separator()

    # Bonus: Demonstrate search_and_fetch convenience method
    print("Bonus: Using search_and_fetch() convenience method")
    print("This searches and fetches multiple pages in one call...\n")

    try:
        fetched_pages = web.search_and_fetch(
            query="Python asyncio tutorial",
            num_pages=2,
            format='text',
            max_content_length=1500
        )
    except Exception as e:
        print(f"✗ search_and_fetch failed: {e}")
        return 1

    print(f"✓ Fetched {len(fetched_pages)} pages:")
    for i, page in enumerate(fetched_pages, 1):
        print(f"\n{i}. {page['title']}")
        print(f"   URL: {page['url']}")
        print(f"   Content length: {len(page['content'])} characters")

    print_separator()
    print("✓ Example completed successfully!")
    print("\nNext steps:")
    print("- Try different search queries")
    print("- Experiment with different AI providers")
    print("- Use 'markdown' format for better structured content")
    print("- Build your own research applications on top of ModuLLe")

    return 0


if __name__ == '__main__':
    sys.exit(main())
