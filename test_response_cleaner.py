#!/usr/bin/env python3
"""
Test response_cleaner with real thinking-model output patterns.

Patterns taken from logs/thehorror.html (Qwen 3.8 27B) and typical
Qwen3/DeepSeek-R1/Nemotron output formats.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modulle.utils.response_cleaner import (
    clean_response,
    strip_think_tags,
    looks_like_reasoning,
    parse_yes_no,
)

# Construct tags via chr() so literal tags can never be mangled by tooling
THINK_OPEN = "<" + "think" + ">"
THINK_CLOSE = "<" + "/" + "think" + ">"


def main():
    failures = 0

    def check(name, actual, expected):
        nonlocal failures
        status = "PASS" if actual == expected else "FAIL"
        if actual != expected:
            failures += 1
        print(f"[{status}] {name}")
        if actual != expected:
            print(f"  expected: {expected!r}")
            print(f"  actual:   {actual!r}")

    # 1. The exact reasoning leak from thehorror.html (Qwen 3.8, untagged)
    t1 = (
        "We need answer user's request in English. Need summarize article, "
        "objective factual, strip sensationalism. Article: Apple has updated "
        "iPhone 18 Pro Dynamic Island to be more useful with up to three Live "
        "Activities shown simultaneously."
    )
    check("pure untagged reasoning -> None", clean_response(t1), None)
    check("pure untagged reasoning -> looks_like_reasoning", looks_like_reasoning(t1), True)

    # 2. Reasoning followed by separated answer (double newline)
    t2 = (
        "We need answer user: \"IMPORTANT: You MUST respond in English. Generate a headline\"\n\n"
        "Dynamic Island Now Shows Three Live Activities at Once"
    )
    check("reasoning + answer -> salvaged tail", clean_response(t2),
          "Dynamic Island Now Shows Three Live Activities at Once")

    # 3. Properly tagged think block (Qwen3/DeepSeek-R1/GLM format)
    t3 = (
        THINK_OPEN + "Let me analyze this article about Apple."
        + THINK_CLOSE
        + " Apple announced that the iPhone 18 Pro Dynamic Island "
        "now shows up to three Live Activities at once."
    )
    check("tagged think block -> stripped", clean_response(t3),
          "Apple announced that the iPhone 18 Pro Dynamic Island now shows up to three Live Activities at once.")

    # 4. Unclosed think tag (truncated output): everything after the tag is
    #    treated as thinking, so nothing usable remains -> None (skip article)
    t4 = (
        THINK_OPEN + "The article discusses Apple and battery life."
        + " Apple announced that the iPhone 18 Pro Dynamic Island "
        "now shows three Live Activities."
    )
    check("unclosed think tag -> None (nothing usable)", clean_response(t4), None)

    # 5. Empty think block
    t5 = THINK_OPEN + THINK_CLOSE + "Summary text here."
    check("empty think block -> stripped", clean_response(t5), "Summary text here.")

    # 6. Normal response passes through untouched
    normal = "Apple updated the Dynamic Island to display three Live Activities simultaneously."
    check("normal response untouched", clean_response(normal), normal)

    # 7. Empty / None
    check("None -> None", clean_response(None), None)
    check("empty -> None", clean_response(""), None)
    check("whitespace -> None", clean_response("   \n  "), None)

    # 8. Pure reasoning inside think tags -> only tag stripped, content evaluated
    t8 = THINK_OPEN + "Okay, let's see. Need summarize." + THINK_CLOSE + "Apple did X and Y."
    check("tagged reasoning -> answer kept", clean_response(t8), "Apple did X and Y.")

    # 9. yes/no parsing - the critical detection fix
    check("bare yes", parse_yes_no("yes"), True)
    check("bare no", parse_yes_no("no"), False)
    check("YES uppercase", parse_yes_no("YES"), True)
    check("yes with punctuation", parse_yes_no("\"Yes.\""), True)
    check("tagged yes",
          parse_yes_no(THINK_OPEN + "checking the title..." + THINK_CLOSE + "yes"), True)

    # 10. THE thehorror failure mode: reasoning text that merely MENTIONS yes
    reasoning_yes = (
        "We need answer user's question. Need analyze if article is clickbait. "
        "The title says 'You won't believe this' but content is factual, "
        "so the answer should be no in this case."
    )
    check("reasoning mentioning yes/no -> None (not auto-True)",
          parse_yes_no(reasoning_yes), None)

    long_ambiguous = (
        "I would say yes but honestly the article is not that great and the "
        "no case is also plausible, so yes maybe"
    )
    check("long ambiguous reasoning -> None", parse_yes_no(long_ambiguous), None)

    # 11. Chatty-model sentence verdicts (Ornith style, from live test)
    check("sentence verdict 'No, ...'", parse_yes_no(
        "No, the article is straightforward factual reporting about Apple."), False)
    check("sentence verdict 'Yes, ...'", parse_yes_no(
        "Yes, this article uses sensationalized language."), True)
    check("negation phrasing", parse_yes_no(
        "It is not clickbait; the title matches the content."), False)
    check("meta-commentary 'The user wants' -> reasoning", looks_like_reasoning(
        "The user wants me to summarize this article about Apple's iPhone."), True)

    # 12. Trailing narration (from live Ornith test: ends with "Let me write...")
    live_ornith = (
        "Key facts:\n- Apple updated iPhone 18 Pro Dynamic Island\n"
        "- Can now show up to three Live Activities simultaneously\n\n"
        "Let me write a professional summary."
    )
    check("trailing 'Let me' narration stripped", clean_response(live_ornith),
          "Key facts:\n- Apple updated iPhone 18 Pro Dynamic Island\n"
          "- Can now show up to three Live Activities simultaneously")

    # 13. Strip only (no None conversion)
    check("strip_think_tags keeps content",
          strip_think_tags(THINK_OPEN + "x" + THINK_CLOSE + "result only"),
          "result only")

    # 14. THE Disenchanted leak: closing tag with no opener — model reasoned
    #    as plain text and only emitted </think> at the end of its thinking.
    t14 = "Reasoning about the user's image. I should respond warmly." \
          + THINK_CLOSE + "\n\nThis is the final answer."
    check("lone closing tag -> prefix cut, answer kept", clean_response(t14),
          "This is the final answer.")

    # 15. Tag variants: attributes and inner whitespace
    t15 = "<think foo=1>x< /think >Answer."
    check("tag with attributes/whitespace -> stripped",
          strip_think_tags(t15), "Answer.")

    # 16. HTML-escaped tag pair (renderers escape angle brackets)
    t16 = "&lt;think&gt;r&lt;/think&gt;Answer here."
    check("escaped think pair -> stripped", strip_think_tags(t16), "Answer here.")

    # 17. HTML-escaped lone closer
    t17 = "reasoning &lt;/think&gt;\n\nAnswer."
    check("escaped lone closer -> prefix cut", clean_response(t17), "Answer.")

    print()
    if failures:
        print(f"{failures} test(s) FAILED")
        raise SystemExit(1)
    print("All tests passed!")


if __name__ == "__main__":
    main()