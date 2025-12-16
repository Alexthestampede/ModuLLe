#!/usr/bin/env python3
"""
Basic usage example for ModuLLe AI provider abstraction.

This example demonstrates how to use ModuLLe's generic text generation API.
ModuLLe provides simple, generic methods - you build your application logic
by crafting appropriate prompts.
"""

from modulle.providers.ollama.text_processor import OllamaTextProcessor
from modulle.providers.ollama.client import OllamaClient

# Sample text to work with
SAMPLE_TEXT = """
Artificial Intelligence (AI) has become an integral part of modern technology,
transforming industries from healthcare to finance. Machine learning algorithms
can now process vast amounts of data to identify patterns and make predictions
with remarkable accuracy. Deep learning, a subset of machine learning, has
enabled breakthroughs in computer vision, natural language processing, and
speech recognition.
"""


def main():
    print("ModuLLe Basic Usage Example")
    print("=" * 70)
    print("\nModuLLe provides GENERIC methods: generate() and chat()")
    print("You build application logic by crafting prompts!")
    print()

    # Create Ollama client (make sure Ollama is running)
    print("1. Creating Ollama client...")
    print("-" * 70)
    try:
        client = OllamaClient(base_url='http://localhost:11434')
        text_processor = OllamaTextProcessor(
            model='llama2',  # Change to your installed model
            base_url='http://localhost:11434'
        )
        print("   ✓ Client created successfully")
    except Exception as e:
        print(f"   ✗ Error creating client: {e}")
        print("   Make sure Ollama is running: ollama serve")
        return

    # Check server health
    print("\n2. Checking server health...")
    print("-" * 70)
    if client.health_check():
        print("   ✓ Ollama server is running")
    else:
        print("   ✗ Ollama server is not responding")
        return

    # List available models
    print("\n3. Listing available models...")
    print("-" * 70)
    models = client.list_models()
    if models:
        print(f"   Available models: {', '.join(models[:5])}")
        if len(models) > 5:
            print(f"   ... and {len(models) - 5} more")
    else:
        print("   No models found or error occurred")

    # Example 1: Text Summarization
    print("\n4. Example: Text Summarization")
    print("-" * 70)
    print("   Using generic generate() method with a summarization prompt")

    summary_result = text_processor.generate(
        prompt=f"Summarize this text in 2-3 sentences:\n\n{SAMPLE_TEXT}",
        system_prompt="You are a helpful assistant that creates concise summaries.",
        temperature=0.3
    )

    if summary_result:
        print(f"\n   Summary:\n   {summary_result}")
    else:
        print("   ✗ Failed to generate summary")

    # Example 2: Translation
    print("\n5. Example: Translation")
    print("-" * 70)
    print("   Using generic generate() method with a translation prompt")

    translation_result = text_processor.generate(
        prompt=f"Translate this text to Spanish:\n\n{SAMPLE_TEXT[:200]}",
        system_prompt="You are a professional translator.",
        temperature=0.2
    )

    if translation_result:
        print(f"\n   Translation:\n   {translation_result[:200]}...")
    else:
        print("   ✗ Failed to translate")

    # Example 3: Q&A
    print("\n6. Example: Question Answering")
    print("-" * 70)
    print("   Using generic generate() method with a Q&A prompt")

    qa_result = text_processor.generate(
        prompt=f"Based on this text, what industries has AI transformed?\n\nText: {SAMPLE_TEXT}",
        system_prompt="You are a helpful assistant that answers questions based on provided text.",
        temperature=0.1
    )

    if qa_result:
        print(f"\n   Answer:\n   {qa_result}")
    else:
        print("   ✗ Failed to answer question")

    # Example 4: Chat (multi-turn conversation)
    print("\n7. Example: Multi-turn Chat")
    print("-" * 70)
    print("   Using generic chat() method for conversation")

    chat_result = text_processor.chat(
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "Machine learning is a subset of AI that enables systems to learn from data."},
            {"role": "user", "content": "Can you give me an example?"}
        ],
        temperature=0.5
    )

    if chat_result:
        print(f"\n   Response:\n   {chat_result}")
    else:
        print("   ✗ Failed to chat")

    print("\n" + "=" * 70)
    print("\nKey Takeaways:")
    print("1. ModuLLe provides only generate() and chat() - simple and generic")
    print("2. You build ANY application logic by crafting prompts")
    print("3. Works with ANY provider: Ollama, OpenAI, Claude, Gemini, LM Studio")
    print("4. See article_summarizer.py for a complete application example")
    print()


if __name__ == "__main__":
    main()
