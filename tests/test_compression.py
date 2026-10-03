"""Tests for the context compression engine (offline, mocked summarizer)."""

import pytest

from modulle.context import (
    ContextCompressor,
    estimate_tokens,
    estimate_messages_tokens,
    DEFAULT_COMPRESSION_PROMPT,
)


def msg(role, text):
    return {'role': role, 'content': text}


def make_history(n_pairs, filler_len=400):
    """Create n user/assistant pairs of realistic-size text."""
    out = [msg('system', 'You are helpful.')]
    for i in range(n_pairs):
        out.append(msg('user', f"Question {i}: " + "word " * filler_len))
        out.append(msg('assistant', f"Answer {i}: " + "word " * filler_len))
    return out


class TestEstimates:
    def test_estimate_tokens(self):
        assert estimate_tokens('') == 0
        assert estimate_tokens('abcd' * 10) == 10

    def test_estimate_messages_tokens(self):
        msgs = [msg('system', 'x' * 40), msg('user', 'y' * 40)]
        assert estimate_messages_tokens(msgs) == 10 + 10 + 8  # +4 overhead each


class TestCompressorInit:
    def test_bad_threshold(self):
        with pytest.raises(ValueError):
            ContextCompressor(threshold_pct=0)
        with pytest.raises(ValueError):
            ContextCompressor(threshold_pct=101)

    def test_bad_max_tokens(self):
        with pytest.raises(ValueError):
            ContextCompressor(max_tokens=100)

    def test_defaults(self):
        c = ContextCompressor()
        assert c.threshold_pct == 70
        assert DEFAULT_COMPRESSION_PROMPT in c.summary_prompt


class TestShouldCompress:
    def test_below_threshold_no_compress(self):
        c = ContextCompressor(summarizer=lambda p: 'summary', max_tokens=8192,
                              threshold_pct=70)
        msgs = make_history(2, filler_len=50)
        assert c.should_compress(msgs) is False
        assert c.maybe_compress(msgs) == msgs

    def test_at_threshold_triggers(self):
        c = ContextCompressor(summarizer=lambda p: 'summary', max_tokens=8192,
                              threshold_pct=70)
        # 8192 * 0.7 = ~5734 tokens -> need that much text
        msgs = make_history(8, filler_len=450)
        assert c.should_compress(msgs) is True


class TestMaybeCompress:
    def _compressor(self):
        calls = []

        def fake_summarizer(prompt):
            calls.append(prompt)
            return "The user asked about topic A and B."

        return ContextCompressor(summarizer=fake_summarizer,
                                 max_tokens=4096, threshold_pct=70,
                                 keep_recent=4), calls

    def test_small_history_untouched_even_if_forced(self):
        c, _ = self._compressor()
        msgs = [msg('system', 'sys'), msg('user', 'hi')]
        assert c.maybe_compress(msgs, force=True) == msgs

    def test_compress_replaces_middle_keeps_head_tail(self):
        c, calls = self._compressor()
        msgs = make_history(20, filler_len=450)
        out = c.maybe_compress(msgs)
        assert out != msgs
        assert len(out) < len(msgs)
        # head system prompt preserved
        assert out[0]['role'] == 'system'
        assert out[0]['content'] == 'You are helpful.'
        # summary marker present
        joined = '\n'.join(m['content'] for m in out)
        assert 'earlier conversation summarized' in joined
        assert 'topic A and B' in joined
        # recent tail kept verbatim
        assert out[-1]['content'] == msgs[-1]['content']
        # summarizer got the transcript
        assert any('Question 0:' in p for p in calls)

    def test_compress_drops_estimated_size(self):
        c, _ = self._compressor()
        msgs = make_history(20, filler_len=450)
        out = c.compress(msgs)
        assert c.usage_pct(out) < c.usage_pct(msgs)

    def test_no_summarizer_returns_original(self):
        c = ContextCompressor(processor=None, max_tokens=4096, threshold_pct=1)
        msgs = make_history(10)
        assert c.maybe_compress(msgs) == msgs

    def test_empty_summary_skips(self):
        c = ContextCompressor(summarizer=lambda p: '', max_tokens=4096,
                              threshold_pct=1)
        msgs = make_history(10)
        assert c.maybe_compress(msgs) == msgs

    def test_custom_trigger_mark(self):
        c = ContextCompressor(summarizer=lambda p: 'S', max_tokens=4096,
                              threshold_pct=1, trigger_mark='[compressed]')
        msgs = make_history(10)
        out = c.maybe_compress(msgs)
        assert any(m['content'].startswith('[compressed]') for m in out)

    def test_custom_prompt_used(self):
        got = []

        def fake(prompt):
            got.append(prompt)
            return 'S'

        c = ContextCompressor(summarizer=fake, max_tokens=4096, threshold_pct=1,
                              summary_prompt='CUSTOM-PROMPT')
        c.maybe_compress(make_history(10))
        assert got[0].startswith('CUSTOM-PROMPT')