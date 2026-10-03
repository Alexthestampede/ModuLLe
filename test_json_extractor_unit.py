#!/usr/bin/env python3
"""Unit tests for modulle.utils.json_extractor (upstream import style)."""
import json
from modulle.utils.json_extractor import extract_json

THINK_OPEN = "<" + "think" + ">"
THINK_CLOSE = "<" + "/" + "think" + ">"

GOOD = {
    "title": "Dynamic Island Now Shows Three Live Activities",
    "summary": "Apple updated the iPhone 18 Pro Dynamic Island.",
    "is_clickbait": False,
    "is_ad": False,
}

def main():
    failures = 0
    def check(name, actual, expected):
        nonlocal failures
        ok = actual == expected
        if not ok: failures += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")

    check("bare JSON", extract_json(json.dumps(GOOD)), GOOD)
    check("fenced", extract_json("```json\n" + json.dumps(GOOD) + "\n```"), GOOD)
    check("preamble+trailer", extract_json("Sure!\n" + json.dumps(GOOD) + "\nLet me know!"), GOOD)
    check("tagged reasoning", extract_json(THINK_OPEN + "t" + THINK_CLOSE + json.dumps(GOOD)), GOOD)
    check("no JSON", extract_json("I cannot comply."), None)
    check("malformed then valid", extract_json("{broken json\n" + json.dumps(GOOD)), GOOD)
    check("None", extract_json(None), None)
    print("FAILURES:", failures)
    raise SystemExit(1 if failures else 0)

main()
