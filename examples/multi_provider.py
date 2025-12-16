#!/usr/bin/env python3
"""
Multi-provider example for ModuLLe.

This example demonstrates how to easily switch between different AI providers
using the same code.
"""

from modulle import create_ai_client

# Sample text to summarize
SAMPLE_TEXT = """
The Internet of Things (IoT) is revolutionizing how we interact with our
environment. Connected devices in homes, cars, and cities collect and share
data to improve efficiency and convenience. Smart thermostats learn user
preferences, wearable devices monitor health metrics, and industrial sensors
optimize manufacturing processes. As 5G networks expand, IoT applications
will become even more sophisticated and widespread.
"""


def test_provider(provider_name, **kwargs):
    """Test a specific provider."""
    print(f"\nTesting {provider_name.upper()} provider")
    print("-" * 50)

    try:
        # Create client for the provider
        client, text_processor, vision_processor = create_ai_client(
            provider=provider_name,
            **kwargs
        )
        print(f"✓ {provider_name.capitalize()} client created")

        # Check health
        if client.health_check():
            print(f"✓ {provider_name.capitalize()} server is healthy")
        else:
            print(f"✗ {provider_name.capitalize()} server not responding")
            return

        # Generate summary
        print("Generating summary...")
        result = text_processor.generate_summary(
            text=SAMPLE_TEXT,
            language="English",
            max_length=150
        )

        if result:
            print(f"Title: {result['title']}")
            print(f"Summary: {result['summary'][:100]}...")
            print(f"Clickbait: {result['is_clickbait']}")
        else:
            print("Failed to generate summary")

    except Exception as e:
        print(f"✗ Error with {provider_name}: {e}")


def main():
    print("ModuLLe Multi-Provider Example")
    print("=" * 50)
    print("\nThis example shows how to use the same code with different providers.")
    print("Make sure to configure your API keys and local servers before running.")

    # Test Ollama (local)
    test_provider(
        'ollama',
        text_model='llama2',
        base_url='http://localhost:11434'
    )

    # Test LM Studio (local)
    # Uncomment if you have LM Studio running
    # test_provider(
    #     'lm_studio',
    #     text_model='local-model',
    #     base_url='http://localhost:1234'
    # )

    # Test OpenAI (cloud)
    # Uncomment and add your API key
    # test_provider(
    #     'openai',
    #     text_model='gpt-4o-mini',
    #     api_key='your-api-key-here'
    # )

    # Test Gemini (cloud)
    # Uncomment and add your API key
    # test_provider(
    #     'gemini',
    #     text_model='gemini-1.5-flash',
    #     api_key='your-api-key-here'
    # )

    # Test Claude (cloud)
    # Uncomment and add your API key
    # test_provider(
    #     'claude',
    #     text_model='claude-3-5-haiku-20241022',
    #     api_key='your-api-key-here'
    # )

    print("\n" + "=" * 50)
    print("Example completed!")


if __name__ == "__main__":
    main()
