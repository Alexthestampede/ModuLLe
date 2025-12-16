#!/usr/bin/env python3
"""
Multi-language example for ModuLLe.

This example demonstrates how to generate summaries in different languages.
"""

from modulle import create_ai_client

# Sample article about technology (in English)
ARTICLE_TEXT = """
Cloud computing has transformed the way businesses operate by providing
on-demand access to computing resources over the internet. Companies can
now scale their infrastructure dynamically, paying only for what they use.
This flexibility has enabled startups to compete with established enterprises
and has accelerated innovation across industries. Major cloud providers like
AWS, Azure, and Google Cloud offer a wide range of services from basic storage
to advanced machine learning capabilities.
"""


def generate_in_language(text_processor, language):
    """Generate summary in specified language."""
    print(f"\nGenerating summary in {language}...")
    print("-" * 50)

    result = text_processor.generate_summary(
        text=ARTICLE_TEXT,
        language=language,
        max_length=200
    )

    if result:
        print(f"Title: {result['title']}")
        print(f"Summary: {result['summary']}")
        print(f"Clickbait: {result['is_clickbait']}")
    else:
        print(f"Failed to generate summary in {language}")


def main():
    print("ModuLLe Multi-Language Example")
    print("=" * 50)

    # Create AI client (using Ollama as example)
    print("\nCreating AI client...")
    try:
        client, text_processor, vision_processor = create_ai_client(
            provider='ollama',
            text_model='llama2',  # Make sure this model supports multiple languages
            base_url='http://localhost:11434'
        )
        print("✓ Client created successfully")
    except Exception as e:
        print(f"✗ Error: {e}")
        return

    # Test different languages
    languages = [
        "English",
        "Spanish",
        "French",
        "German",
        "Italian",
        "Portuguese",
        "Japanese",
        "Chinese"
    ]

    print("\nGenerating summaries in multiple languages...")
    for language in languages:
        try:
            generate_in_language(text_processor, language)
        except Exception as e:
            print(f"Error with {language}: {e}")

    print("\n" + "=" * 50)
    print("Example completed!")
    print("\nNote: Language quality depends on the model's training.")
    print("Some models may work better with certain languages.")


if __name__ == "__main__":
    main()
