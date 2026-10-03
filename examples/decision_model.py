#!/usr/bin/env python3
"""
Decision model example for ModuLLe (Ollama only).

Decision models like clef / clef-flash answer typed questions about a state
in a single fast forward pass, using Ollama's /v1/systemone endpoint.
Requires Ollama >= 0.35.1 and a decision model (ollama pull clef-flash).

Run against your server:
    OLLAMA_BASE_URL=http://192.168.2.150:11434 python3 examples/decision_model.py
"""

import os

from modulle.config import OLLAMA_BASE_URL
from modulle.providers.ollama import OllamaClient

SERVER = os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
MODEL = "clef-flash"

client = OllamaClient(base_url=SERVER)

if not client.health_check():
    raise SystemExit(f"Ollama server not reachable at {SERVER}")

print(f"Judging with {MODEL} on {SERVER}\n" + "-" * 60)

result = client.system_one(
    model=MODEL,
    state="Checkout has been failing for every customer for the last hour.",
    questions={
        "urgent": {
            "type": "noul",
            "instructions": "Is this support request urgent?",
        },
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

if result is None:
    raise SystemExit(
        "Request failed. Check that Ollama >= 0.35.1 is running and "
        f"'{MODEL}' is pulled (ollama pull {MODEL})."
    )

answers = result["answers"]
print(f"urgent   (noul):  {answers['urgent']['noul']:.3f} true")
print(f"team     (choice): {answers['team']['choice']}")
print(f"severity (score):  {answers['severity']['score']:.2f} / 3")

print("\nRaw usage:", result.get("usage"))
