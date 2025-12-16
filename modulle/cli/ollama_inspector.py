"""
Ollama Model Inspector
Inspects Ollama server models for capabilities like vision support and context size.
"""
import json
import urllib.request
import urllib.error
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ModelCapabilities:
    """Data class to store model capabilities and metadata."""
    name: str
    size: str
    parameters: str
    context_window: int
    actual_context: Optional[int]
    vision_capable: bool
    tool_calling: bool
    family: str
    format: str
    quantization: str
    embedding_length: Optional[int] = None
    error: Optional[str] = None


class OllamaInspector:
    """Inspector for Ollama server models."""

    def __init__(self, server_url: str = "http://localhost:11434"):
        """Initialize the inspector with server URL."""
        self.server_url = server_url.rstrip('/')

    def _make_request(self, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """Make HTTP request to Ollama server."""
        url = f"{self.server_url}{endpoint}"

        try:
            if data:
                # POST request with JSON data
                req_data = json.dumps(data).encode('utf-8')
                req = urllib.request.Request(
                    url,
                    data=req_data,
                    headers={'Content-Type': 'application/json'}
                )
            else:
                # GET request
                req = urllib.request.Request(url)

            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode('utf-8'))

        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to connect to {url}: {e.reason}")
        except urllib.error.HTTPError as e:
            raise ConnectionError(f"HTTP error {e.code} when accessing {url}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response from {url}: {e}")

    def test_connection(self) -> bool:
        """Test if the Ollama server is accessible."""
        try:
            self._make_request("/api/tags")
            return True
        except (ConnectionError, ValueError):
            return False

    def list_models(self) -> List[str]:
        """List all available models on the server."""
        response = self._make_request("/api/tags")
        models = response.get("models", [])
        return [model["name"] for model in models]

    def get_model_info(self, model_name: str) -> Dict:
        """Get detailed information about a specific model."""
        return self._make_request("/api/show", {"name": model_name})

    def _extract_context_window(self, model_info: Dict) -> int:
        """Extract context window size from model_info."""
        info = model_info.get("model_info", {})

        # Search for any key ending with .context_length
        for key, value in info.items():
            if key.endswith(".context_length"):
                try:
                    return int(value)
                except (ValueError, TypeError):
                    pass

        # Fallback: check common locations
        if "context_length" in info:
            try:
                return int(info["context_length"])
            except (ValueError, TypeError):
                pass

        return 0  # Unknown

    def _extract_num_ctx(self, model_info: Dict) -> Optional[int]:
        """Extract num_ctx parameter from modelfile."""
        modelfile = model_info.get("modelfile", "")

        if not modelfile:
            return None

        # Parse modelfile line by line looking for PARAMETER num_ctx
        for line in modelfile.split('\n'):
            line = line.strip()
            if line.upper().startswith("PARAMETER") and "num_ctx" in line.lower():
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        return int(parts[2])
                    except (ValueError, IndexError):
                        pass

        return None

    def _check_vision_capability(self, model_info: Dict) -> bool:
        """Check if model has vision capabilities."""
        details = model_info.get("details", {})
        family = details.get("family", "").lower()

        # Check for vision-language (vl) in family name
        if "vl" in family or "vision" in family:
            return True

        # Check model_info for vision-related fields
        info = model_info.get("model_info", {})

        # Check for CLIP/vision projector parameters
        for key in info.keys():
            key_lower = key.lower()
            if "vision" in key_lower or "clip" in key_lower or "projector" in key_lower:
                return True

        # Check projector info in model details
        projector = model_info.get("projector_info", {})
        if projector:
            return True

        return False

    def _format_size(self, size_bytes: int) -> str:
        """Format size in bytes to human-readable format."""
        if size_bytes == 0:
            return "Unknown"

        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"

    def _extract_parameters(self, model_info: Dict) -> str:
        """Extract parameter count from model info."""
        details = model_info.get("details", {})

        # Check parameter_size field
        param_size = details.get("parameter_size", "")
        if param_size:
            return param_size

        # Check parameters field
        parameters = details.get("parameters", "")
        if parameters:
            return parameters

        # Try to extract from model_info
        info = model_info.get("model_info", {})

        # Look for general.parameter_count
        param_count = info.get("general.parameter_count")
        if param_count:
            try:
                count = int(param_count)
                if count >= 1_000_000_000:
                    return f"{count / 1_000_000_000:.1f}B"
                elif count >= 1_000_000:
                    return f"{count / 1_000_000:.1f}M"
                else:
                    return str(count)
            except (ValueError, TypeError):
                pass

        return "Unknown"

    def inspect_model(self, model_name: str) -> ModelCapabilities:
        """Inspect a single model and return its capabilities."""
        try:
            info = self.get_model_info(model_name)

            details = info.get("details", {})

            # Extract all capabilities
            context_window = self._extract_context_window(info)
            actual_context = self._extract_num_ctx(info)
            vision_capable = self._check_vision_capability(info)

            # Extract metadata
            size_bytes = info.get("size", 0)
            size = self._format_size(size_bytes)
            parameters = self._extract_parameters(info)
            family = details.get("family", "Unknown")
            format_type = details.get("format", "Unknown")
            quantization = details.get("quantization_level", "Unknown")

            return ModelCapabilities(
                name=model_name,
                size=size,
                parameters=parameters,
                context_window=context_window,
                actual_context=actual_context,
                vision_capable=vision_capable,
                tool_calling=False,  # Not needed for our use case
                family=family,
                format=format_type,
                quantization=quantization,
                embedding_length=None
            )

        except Exception as e:
            return ModelCapabilities(
                name=model_name,
                size="Error",
                parameters="Error",
                context_window=0,
                actual_context=None,
                vision_capable=False,
                tool_calling=False,
                family="Error",
                format="Error",
                quantization="Error",
                error=str(e)
            )

    def inspect_all_models(self) -> List[ModelCapabilities]:
        """Inspect all models on the server."""
        models = self.list_models()
        return [self.inspect_model(model) for model in models]

    def get_vision_models(self, min_context: int = 4096) -> List[ModelCapabilities]:
        """
        Get all vision-capable models with context >= min_context.

        Args:
            min_context: Minimum context window size (default 4096)

        Returns:
            List of ModelCapabilities for vision models
        """
        all_models = self.inspect_all_models()
        return [
            m for m in all_models
            if m.vision_capable and not m.error and
            (m.actual_context or m.context_window or 0) >= min_context
        ]

    def get_text_models(self, min_context: int = 4096) -> List[ModelCapabilities]:
        """
        Get all text models with context >= min_context.

        Args:
            min_context: Minimum context window size (default 4096)

        Returns:
            List of ModelCapabilities for text models
        """
        all_models = self.inspect_all_models()
        return [
            m for m in all_models
            if not m.error and
            (m.actual_context or m.context_window or 0) >= min_context
        ]


def main():
    """Main entry point for ollama inspector CLI."""
    import sys

    print("=" * 70)
    print("  Ollama Model Inspector")
    print("=" * 70)
    print()

    # Get server URL from args or use default
    server_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:11434"

    print(f"Connecting to Ollama server: {server_url}")
    print()

    inspector = OllamaInspector(server_url)

    # Test connection
    if not inspector.test_connection():
        print("ERROR: Failed to connect to Ollama server!")
        print(f"Please ensure Ollama is running at {server_url}")
        sys.exit(1)

    print("Connection successful!")
    print()

    # Get all models
    models = inspector.inspect_all_models()

    if not models:
        print("No models found on server.")
        print("Pull a model with: ollama pull llama2")
        sys.exit(0)

    print(f"Found {len(models)} model(s):\n")

    # Display models
    for i, model in enumerate(models, 1):
        print(f"[{i}] {model.name}")

        if model.error:
            print(f"    ERROR: {model.error}")
        else:
            print(f"    Parameters: {model.parameters}, Size: {model.size}")
            print(f"    Family: {model.family}, Format: {model.format}")
            print(f"    Context Window: {model.context_window}")
            if model.actual_context:
                print(f"    Configured Context: {model.actual_context}")
            print(f"    Vision Capable: {'Yes' if model.vision_capable else 'No'}")

        print()

    # Show summary
    vision_models = [m for m in models if m.vision_capable and not m.error]
    text_models = [m for m in models if not m.error]

    print("Summary:")
    print(f"  Total models: {len(models)}")
    print(f"  Text models: {len(text_models)}")
    print(f"  Vision models: {len(vision_models)}")
    print()


if __name__ == '__main__':
    main()
