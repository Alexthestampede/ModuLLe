#!/usr/bin/env python3
"""
Article Summarizer Example - Building Application Logic on ModuLLe

This example demonstrates how to build application-specific functionality
(article summarization, clickbait detection, title generation) ON TOP of
ModuLLe's generic AI provider abstraction.

Key points:
- ModuLLe provides ONLY generic `generate()` and `chat()` methods
- Application logic is built by crafting specific prompts
- This approach keeps ModuLLe clean and reusable for ANY application
"""

from typing import Optional, Dict, Any
from modulle.providers.ollama.text_processor import OllamaTextProcessor
from modulle.providers.openai.text_processor import OpenAITextProcessor


class ArticleSummarizer:
    """
    Article summarization built on top of ModuLLe's generic API.

    This class shows how to build domain-specific logic (article processing)
    by composing prompts and using the generic generate() method.
    """

    def __init__(self, text_processor):
        """
        Initialize with any ModuLLe text processor.

        Args:
            text_processor: Any BaseTextProcessor implementation
                           (Ollama, OpenAI, Claude, Gemini, LM Studio)
        """
        self.processor = text_processor

        # Application-specific configuration
        self.clickbait_authors = []  # Hardcoded clickbait author list
        self.summary_temperature = 0.3  # Lower temp for factual summaries
        self.title_temperature = 0.2    # Very low temp for concise titles

    def detect_clickbait(self, title: str, text: str) -> bool:
        """
        Detect if an article is clickbait using AI.

        This uses the generic `generate()` method with a specific prompt.

        Args:
            title: Article title
            text: Article text excerpt

        Returns:
            True if clickbait detected, False otherwise
        """
        if not title or not text:
            return False

        system_prompt = (
            "You are a clickbait detection expert. "
            "Analyze the article title and excerpt to determine if it is clickbait. "
            "Clickbait indicators include: "
            "- Sensationalized or exaggerated headlines "
            "- Misleading titles that don't match the content "
            "- Emotional manipulation tactics "
            "- 'You won't believe...', 'This one trick...', 'Shocking...' type language "
            "- Withholding key information to force clicks "
            "Respond with ONLY 'yes' if it is clickbait, or 'no' if it is not."
        )

        user_prompt = f"Title: {title}\n\nExcerpt: {text[:1000]}\n\nIs this clickbait?"

        response = self.processor.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.1  # Very low temperature for consistent detection
        )

        if response and 'yes' in response.lower():
            print(f"   Clickbait detected: {title[:50]}...")
            return True

        return False

    def generate_summary(
        self,
        text: str,
        title: Optional[str] = None,
        author: Optional[str] = None,
        language: str = "English",
        max_length: int = 500
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a summary of an article.

        This method orchestrates multiple calls to the generic API:
        1. Check for hardcoded clickbait authors
        2. Use AI to detect clickbait
        3. Generate summary with appropriate prompt
        4. Generate title from summary

        Args:
            text: Article text to summarize
            title: Article title (for clickbait detection)
            author: Article author (for clickbait checking)
            language: Language for the summary
            max_length: Maximum summary length

        Returns:
            Dict with summary, title, and clickbait info, or None on error
        """
        if not text or not text.strip():
            print("   Empty text provided")
            return None

        # Check for clickbait using hardcoded author list
        is_clickbait_author = author in self.clickbait_authors if author else False

        # Check for clickbait using AI
        is_clickbait_ai = False
        if title:
            is_clickbait_ai = self.detect_clickbait(title, text)

        # Combine both detection methods
        is_clickbait = is_clickbait_author or is_clickbait_ai

        # Determine detection source
        if is_clickbait_author and is_clickbait_ai:
            detection_method = "both"
        elif is_clickbait_author:
            detection_method = "author"
        elif is_clickbait_ai:
            detection_method = "ai"
        else:
            detection_method = None

        # Choose appropriate system prompt based on clickbait detection
        if is_clickbait:
            system_prompt = (
                "This article shows signs of clickbait or sensationalism. "
                "Provide an objective, factual summary that strips away dramatic language "
                "and focuses on verifiable facts only. "
                "If no substantial facts exist, state 'Clickbait article with no substantial content.' "
                "Maintain a neutral, skeptical tone."
            )
        else:
            system_prompt = (
                "You are a professional news summarizer. "
                "Provide clear, concise, and objective summaries. "
                "Focus on the key facts, main points, and important details. "
                "Maintain a neutral, professional tone. "
                "Keep summaries between 100-300 words."
            )

        user_prompt = f"IMPORTANT: You MUST respond in {language}. Summarize the following article:\n\n{text[:10000]}"

        # Generate summary using generic API
        summary = self.processor.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=self.summary_temperature
        )

        if not summary:
            print("   Failed to generate summary")
            return None

        # Truncate if too long
        if len(summary) > max_length:
            summary = summary[:max_length].rsplit('.', 1)[0] + '.'

        # Generate title from summary
        generated_title = self.generate_title(summary, language)

        return {
            'summary': summary,
            'title': generated_title,
            'is_clickbait': is_clickbait,
            'clickbait_detected_by': detection_method
        }

    def generate_title(self, summary: str, language: str = "English") -> str:
        """
        Generate a concise title from a summary.

        Args:
            summary: Article summary
            language: Language for the title

        Returns:
            Generated title string
        """
        system_prompt = (
            "You are a professional headline writer. "
            "Generate a clear, concise, and informative headline (max 80 characters) "
            "based on the provided summary. "
            "Do not use clickbait language or sensationalism."
        )

        user_prompt = f"IMPORTANT: You MUST respond in {language}. Generate a headline for this summary:\n\n{summary}"

        title = self.processor.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=self.title_temperature
        )

        if not title:
            return "Article Summary"

        # Clean up and truncate title
        title = title.strip().strip('"\'')
        if len(title) > 80:
            title = title[:77] + "..."

        return title

    def summarize_article(self, article_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Summarize an article from structured data.

        Args:
            article_data: Dict with 'text', 'title', 'author', 'url' fields

        Returns:
            Dict with summary results and original metadata, or None on error
        """
        text = article_data.get('text', '')
        author = article_data.get('author')
        original_title = article_data.get('title', '')
        url = article_data.get('url', '')

        print(f"   Summarizing: {original_title[:50]}...")

        result = self.generate_summary(text, title=original_title, author=author)

        if result:
            result['original_title'] = original_title
            result['url'] = url
            result['author'] = author

        return result


def main():
    """
    Demonstrate building application logic on top of ModuLLe's generic API.
    """
    print("ModuLLe Article Summarizer Example")
    print("=" * 70)
    print("\nThis example shows how to build article summarization USING ModuLLe")
    print("rather than having ModuLLe provide article-specific methods.")
    print()

    # Sample article
    article = {
        'title': "The Rise of Artificial Intelligence in Modern Technology",
        'text': """
            Artificial Intelligence (AI) has become an integral part of modern technology,
            transforming industries from healthcare to finance. Machine learning algorithms
            can now process vast amounts of data to identify patterns and make predictions
            with remarkable accuracy. Deep learning, a subset of machine learning, has
            enabled breakthroughs in computer vision, natural language processing, and
            speech recognition. As AI continues to evolve, it promises to revolutionize
            how we work, communicate, and solve complex problems. However, experts also
            warn about the ethical implications and the need for responsible AI development.
        """,
        'author': "Tech Reporter",
        'url': "https://example.com/ai-article"
    }

    # Example 1: Using Ollama (local)
    print("\n1. Using Ollama (local model)")
    print("-" * 70)
    try:
        ollama_processor = OllamaTextProcessor(
            model='llama2',  # or any model you have installed
        )

        summarizer = ArticleSummarizer(ollama_processor)

        result = summarizer.summarize_article(article)

        if result:
            print(f"\n   Generated Title: {result['title']}")
            print(f"   Summary: {result['summary'][:200]}...")
            print(f"   Clickbait: {result['is_clickbait']}")
        else:
            print("   ✗ Failed to summarize")

    except Exception as e:
        print(f"   ✗ Error: {e}")
        print("   (Make sure Ollama is running with: ollama serve)")

    # Example 2: Using OpenAI (cloud)
    print("\n\n2. Using OpenAI (cloud API)")
    print("-" * 70)
    try:
        openai_processor = OpenAITextProcessor(
            model='gpt-4o-mini',
            # api_key='your-key-here'  # or set OPENAI_API_KEY env var
        )

        summarizer = ArticleSummarizer(openai_processor)

        result = summarizer.summarize_article(article)

        if result:
            print(f"\n   Generated Title: {result['title']}")
            print(f"   Summary: {result['summary'][:200]}...")
            print(f"   Clickbait: {result['is_clickbait']}")
        else:
            print("   ✗ Failed to summarize")

    except Exception as e:
        print(f"   ✗ Error: {e}")
        print("   (Set OPENAI_API_KEY environment variable)")

    print("\n" + "=" * 70)
    print("\nKey Takeaways:")
    print("1. ModuLLe provides ONLY generic generate() and chat() methods")
    print("2. Article logic is YOUR application code, not part of ModuLLe")
    print("3. Same ArticleSummarizer works with ANY provider (Ollama, OpenAI, etc.)")
    print("4. You can build ANY domain logic this way: chatbots, code analysis, etc.")
    print()


if __name__ == "__main__":
    main()
