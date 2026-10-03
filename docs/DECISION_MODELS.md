# Decision Models (clef / clef-flash)

ModuLLe supports Ollama **decision models** via the `system_one()` method on
`OllamaClient`. Decision models (e.g. [clef-flash](https://ollama.com/library/clef-flash),
a 9B model from Cloudflare) judge a `state` against typed questions and score
every option in a single non-autoregressive pass — fast enough for
latency-critical decisions like routing, gating agent steps, and classification.

**Requirements:**
- Ollama **>= 0.35.1** (decision models use the separate `/v1/systemone` endpoint)
- A decision model pulled locally: `ollama pull clef-flash`

**Important:** decision models do *not* work through the normal
`generate()` / `chat()` / `analyze_image()` API. Use `system_one()` directly:

```python
from modulle.providers.ollama import OllamaClient

client = OllamaClient(base_url="http://192.168.2.150:11434")

result = client.system_one(
    model="clef-flash",
    state="Checkout has been failing for every customer for the last hour.",
    questions={
        "urgent": {"type": "noul", "instructions": "Is this support request urgent?"},
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this request?",
            "criteria": {
                "billing": "Payments, invoices, and refunds",
                "technical": "Outages, errors, and configuration",
                "sales": "Plans and upgrades",
            },
        },
        "severity": {
            "type": "score",
            "instructions": "How severe is the customer impact?",
            "criteria": ["No impact", "Minor", "Major", "Critical"],
        },
    },
)
```

## Question types

| Type | `criteria` | Answer fields |
|------|-----------|---------------|
| `noul` | Optional `{"true": "...", "false": "..."}` | `noul` (probability the answer is true) |
| `choice` | Option → description mapping (2–26 options, `null` = name describes itself) | `choice`, `probabilities`, `confidence` |
| `score` | List of level descriptions, lowest first (2–26 levels, numbered from 0) | `score` (probability-weighted level), `legend`, `probabilities`, `confidence` |

Return value is the full response payload (`None` on error):
`result["answers"]["team"]["choice"]`, `result["answers"]["urgent"]["noul"]`, etc.

## Extras

- **Images:** pass `images=[...]` (base64-encoded PNG/JPEG/WebP, shared by all
  questions) to decide over screenshots, receipts, or photos. Multimodal input
  is supported by clef-flash; URLs and data URLs are not.
- **Structured state:** `state` can be a JSON-serializable object/array, not
  just a string.
- **Ties:** if two `choice` options tie, the answer follows the model's option
  order — put your preferred option first.
- 1–64 questions per request; `confidence` measures probability concentration,
  not correctness.

## Example

See `examples/decision_model.py`:

```bash
OLLAMA_BASE_URL=http://your-server:11434 python3 examples/decision_model.py
```