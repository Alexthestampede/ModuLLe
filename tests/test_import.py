"""
Basic import tests for ModuLLe package.
"""
import pytest


def test_import_modulle():
    """Test that the main package can be imported."""
    import modulle
    assert modulle.__version__ == "0.1.0"


def test_import_base_classes():
    """Test that base classes can be imported."""
    from modulle import BaseAIClient, BaseTextProcessor
    assert BaseAIClient is not None
    assert BaseTextProcessor is not None


def test_import_factory():
    """Test that factory function can be imported."""
    from modulle import create_ai_client
    assert create_ai_client is not None


def test_import_config():
    """Test that config module can be imported."""
    from modulle import config
    assert hasattr(config, 'OLLAMA_BASE_URL')
    assert hasattr(config, 'OPENAI_TEXT_MODEL')


def test_import_providers():
    """Test that provider modules exist."""
    from modulle.providers import ollama
    from modulle.providers import lm_studio
    from modulle.providers import openai
    from modulle.providers import gemini
    from modulle.providers import claude

    assert ollama is not None
    assert lm_studio is not None
    assert openai is not None
    assert gemini is not None
    assert claude is not None


def test_import_utils():
    """Test that utility modules can be imported."""
    from modulle.utils import get_logger, create_session
    assert get_logger is not None
    assert create_session is not None


def test_factory_error_handling():
    """Test that factory raises appropriate errors."""
    from modulle import create_ai_client

    # Test with invalid provider
    with pytest.raises(ValueError) as exc_info:
        create_ai_client(provider='invalid_provider')
    assert 'Unknown AI provider' in str(exc_info.value)


def test_base_classes_are_abstract():
    """Test that base classes cannot be instantiated directly."""
    from modulle.base import BaseAIClient, BaseTextProcessor

    with pytest.raises(TypeError):
        BaseAIClient()

    with pytest.raises(TypeError):
        BaseTextProcessor()
