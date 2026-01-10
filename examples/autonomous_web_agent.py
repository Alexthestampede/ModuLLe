#!/usr/bin/env python3
"""
Autonomous Web Agent Example for ModuLLe

This example demonstrates tool calling where the LLM autonomously decides
when to search the web and fetch pages. Unlike the direct method approach
in web_research.py, here the LLM controls the research workflow.

The agent can:
1. Search the web when it needs information
2. Fetch pages to read full content
3. Synthesize information from multiple sources
4. Decide when it has enough information to answer

This works with all ModuLLe providers that support tool calling:
- Ollama (PickMeAsDefault and other tool-capable models)
- OpenAI (gpt-4o, gpt-4o-mini, etc.)
- Claude (claude-3-5-sonnet-20241022, etc.)
- Gemini (gemini-1.5-flash, gemini-1.5-pro, etc.)
- LM Studio (depends on loaded model)
"""

import sys
from modulle.web import WebAccessor
from modulle.web.tools import SearchWebTool, FetchPageTool
from modulle.tools import ToolRegistry


def print_separator(char="="):
    """Print a visual separator."""
    print("\n" + char * 80 + "\n")


def get_provider_client(provider, model):
    """
    Get the appropriate client for the specified provider.

    Args:
        provider: Provider name ('ollama', 'openai', 'claude', 'gemini', 'lm_studio')
        model: Model name to use

    Returns:
        Tuple of (client, tool_format_method)
    """
    if provider == 'ollama':
        from modulle.providers.ollama.client import OllamaClient
        client = OllamaClient()
        return client, 'to_ollama_format'

    elif provider == 'openai':
        from modulle.providers.openai.client import OpenAIClient
        client = OpenAIClient()
        return client, 'to_openai_format'

    elif provider == 'claude':
        from modulle.providers.claude.client import ClaudeClient
        client = ClaudeClient()
        return client, 'to_claude_format'

    elif provider == 'gemini':
        from modulle.providers.gemini.client import GeminiClient
        client = GeminiClient()
        return client, 'to_gemini_format'

    elif provider == 'lm_studio':
        from modulle.providers.lm_studio.client import LMStudioClient
        client = LMStudioClient()
        return client, 'to_openai_format'  # LM Studio uses OpenAI format

    else:
        raise ValueError(f"Unknown provider: {provider}")


def main():
    """Run autonomous web agent example."""
    print("ModuLLe Autonomous Web Agent")
    print("The LLM will decide when to search and fetch pages")
    print_separator()

    # Configuration - Change these to match your setup
    PROVIDER = 'ollama'  # Options: 'ollama', 'openai', 'claude', 'gemini', 'lm_studio'

    # Model selection per provider
    MODELS = {
        'ollama': 'PickMeAsDefault',
        'openai': 'gpt-4o-mini',
        'claude': 'claude-3-5-sonnet-20241022',
        'gemini': 'gemini-1.5-flash',
        'lm_studio': 'local-model'  # Whatever model you have loaded
    }

    MODEL = MODELS.get(PROVIDER, 'PickMeAsDefault')
    MAX_ITERATIONS = 10  # Max agent iterations to prevent infinite loops

    # Initialize components
    print("Setting up autonomous agent...")
    print(f"Provider: {PROVIDER}")
    print(f"Model: {MODEL}")

    try:
        client, tool_format = get_provider_client(PROVIDER, MODEL)
    except Exception as e:
        print(f"✗ Failed to initialize {PROVIDER} client: {e}")
        print("\nMake sure:")
        print("- Provider is running (for Ollama/LM Studio)")
        print("- API keys are set (for OpenAI/Claude/Gemini)")
        return 1

    web = WebAccessor()

    # Create and register tools
    registry = ToolRegistry()
    registry.register(SearchWebTool(web))
    registry.register(FetchPageTool(web))

    print(f"✓ Registered {len(registry)} tools:")
    for tool_name in registry.list_tools():
        print(f"  - {tool_name}")

    print(f"✓ Using model: {MODEL}")
    print_separator()

    # Initial conversation
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI research assistant with access to web search and page fetching tools. "
                "When asked a question, use these tools to gather current information from the web. "
                "Search for relevant information, fetch pages to read them, and synthesize what you learn. "
                "Once you have enough information, provide a comprehensive answer with citations. "
                "Be thorough but efficient - don't fetch more pages than needed."
            )
        },
        {
            "role": "user",
            "content": (
                "What are the key differences between Python's asyncio and threading? "
                "Search for recent articles and provide a detailed comparison."
            )
        }
    ]

    print("User Question:")
    print(f"  {messages[1]['content']}")
    print_separator()

    # Agent loop - LLM decides what to do
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"--- Agent Iteration {iteration} ---\n")

        # Get LLM response (may include tool calls)
        # Use the appropriate tool format for this provider
        tools_formatted = getattr(registry, tool_format)()

        response = client.chat_with_tools(
            model=MODEL,
            messages=messages,
            tools=tools_formatted,
            temperature=0.7
        )

        if response['finish_reason'] == 'error':
            print("✗ Error communicating with LLM")
            return 1

        # Check if LLM wants to use tools
        if response['tool_calls']:
            print(f"Agent is calling {len(response['tool_calls'])} tool(s):\n")

            # Execute each tool call
            for tool_call in response['tool_calls']:
                tool_name = tool_call['name']
                tool_args = tool_call['arguments']

                print(f"  🔧 {tool_name}:")
                if tool_name == 'search_web':
                    print(f"     Query: {tool_args.get('query', 'N/A')}")
                elif tool_name == 'fetch_page':
                    print(f"     URL: {tool_args.get('url', 'N/A')}")

                # Execute the tool
                try:
                    result = registry.execute(tool_name, **tool_args)
                    print(f"     ✓ Success ({len(result)} chars returned)")

                    # Add assistant message with tool call
                    messages.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": tool_call['id'],
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": tool_args
                            }
                        }]
                    })

                    # Add tool result message
                    # Different providers need different formats
                    tool_result_msg = {
                        "role": "tool",
                        "content": result,
                        "name": tool_name
                    }

                    # Claude needs tool_use_id
                    if PROVIDER == 'claude':
                        tool_result_msg['tool_use_id'] = tool_call['id']

                    messages.append(tool_result_msg)

                except Exception as e:
                    print(f"     ✗ Error: {e}")
                    # Still add the error to messages so LLM knows what happened
                    messages.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": tool_call['id'],
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": tool_args
                            }
                        }]
                    })
                    messages.append({
                        "role": "tool",
                        "content": f"Error: {str(e)}",
                        "name": tool_name
                    })

            print()

        else:
            # LLM provided final answer (no more tool calls)
            print("Agent has completed research and is providing final answer:\n")
            print_separator("-")
            print(response['content'])
            print_separator("-")

            print(f"\n✓ Research completed in {iteration} iteration(s)")
            print(f"✓ Total messages in conversation: {len(messages)}")

            # Show what tools were used
            tool_calls_made = [
                msg for msg in messages
                if msg.get('role') == 'tool'
            ]
            if tool_calls_made:
                print(f"✓ Tools called: {len(tool_calls_made)} time(s)")

                # Count by tool type
                search_count = sum(1 for m in tool_calls_made if m.get('name') == 'search_web')
                fetch_count = sum(1 for m in tool_calls_made if m.get('name') == 'fetch_page')

                if search_count:
                    print(f"  - search_web: {search_count} time(s)")
                if fetch_count:
                    print(f"  - fetch_page: {fetch_count} time(s)")

            return 0

    # Max iterations reached
    print(f"\n⚠ Warning: Reached maximum iterations ({MAX_ITERATIONS})")
    print("The agent may not have completed its research.")

    if response['content']:
        print("\nPartial answer:")
        print_separator("-")
        print(response['content'])
        print_separator("-")

    return 1


if __name__ == '__main__':
    print("\nNote: This example requires a model with tool calling support.")
    print("Using: PickMeAsDefault\n")

    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
        sys.exit(130)
