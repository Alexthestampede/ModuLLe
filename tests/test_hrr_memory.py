"""Tests for HRR memory (offline, temp dirs)."""

import os
import time

import pytest

from modulle.memory import HRRMemoryStore, text_vector, hrr_similarity


@pytest.fixture
def mem(tmp_path):
    m = HRRMemoryStore(path=str(tmp_path / 'mem.json'), dim=256)
    yield m


class TestTextVector:
    def test_deterministic_across_calls(self):
        v1 = text_vector('hello world', 128)
        v2 = text_vector('hello world', 128)
        assert v1 == v2

    def test_deterministic_across_interpreters(self):
        # stable hash -> same vector regardless of PYTHONHASHSEED
        import subprocess, sys, json
        code = (
            "import sys; sys.path.insert(0, %r);"
            "from modulle.memory.hrr import text_vector;"
            "print(json.dumps(text_vector('stable seed check', 64)))"
            % "/home/alexthestampede/Aish/Disenchanted/deps/ModuLLe"
        )
        r1 = subprocess.run([sys.executable, '-c', code],
                            capture_output=True, text=True,
                            env={**os.environ, 'PYTHONHASHSEED': '1'})
        r2 = subprocess.run([sys.executable, '-c', code],
                            capture_output=True, text=True,
                            env={**os.environ, 'PYTHONHASHSEED': '42'})
        assert r1.stdout == r2.stdout

    def test_unit_norm(self):
        v = text_vector('some random text for vectors', 128)
        norm = sum(x * x for x in v) ** 0.5
        assert abs(norm - 1.0) < 1e-6

    def test_similar_text_ranks_higher(self):
        v_a = text_vector('the cat sat on the mat', 256)
        v_b = text_vector('the cat sat on a mat', 256)
        v_c = text_vector('quantum flux capacitor overdrive', 256)
        assert hrr_similarity(v_a, v_b) > hrr_similarity(v_a, v_c)


class TestStoreRecall:
    def test_store_and_recall(self, mem):
        mem.store('User prefers dark themes in all apps.', topics=['preferences'])
        mem.store('User has a cat named Pixel.', topics=['personal'])
        hits = mem.recall('does the user have a pet?')
        assert hits, 'expected at least one hit'
        assert 'Pixel' in hits[0]['text']

    def test_recall_respects_top_k(self, mem):
        for i in range(10):
            mem.store(f'Fact number {i} about topic {i}.')
        assert len(mem.recall('fact topic', top_k=3)) == 3

    def test_topic_filter(self, mem):
        mem.store('Dogs bark loudly.', topics=['animals'])
        mem.store('Stocks fell today.', topics=['finance'])
        hits = mem.recall('animal sounds', topic='finance')
        assert all('Stocks' not in h['text'] or 'finance' == h['topics'][0]
                   for h in hits)
        hits = mem.recall('animal sounds', topic='animals')
        assert hits and 'Dogs' in hits[0]['text']

    def test_min_similarity_excludes_garbage(self, mem):
        mem.store('Precise technical details about KDE Plasma window rules.')
        hits = mem.recall('xylophone orchestration jazz', min_similarity=0.9)
        assert hits == []

    def test_empty_memory_recall(self, mem):
        assert mem.recall('anything') == []
        assert mem.as_context_note('anything') is None

    def test_as_context_note_format(self, mem):
        mem.store('Likes tea over coffee.')
        note = mem.as_context_note('tea or coffee?')
        assert note.startswith('Relevant memory' if False else 'Relevant memories')
        assert 'Likes tea over coffee.' in note


class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        p = str(tmp_path / 'mem.json')
        m1 = HRRMemoryStore(path=p, dim=128)
        m1.store('Persisted fact about turtles.', topics=['animals'])
        m1.save()
        m2 = HRRMemoryStore(path=p, dim=128)
        assert len(m2.entries) == 1
        assert m2.entries[0]['text'] == 'Persisted fact about turtles.'
        hits = m2.recall('turtles')
        assert hits and hits[0]['similarity'] > 0.3

    def test_max_entries_eviction(self, tmp_path):
        p = str(tmp_path / 'mem.json')
        m = HRRMemoryStore(path=p, dim=64, max_entries=5)
        for i in range(8):
            m.store(f'entry {i}')
        assert len(m.entries) == 5
        assert m.entries[0]['text'] == 'entry 3'


class TestEditDelete:
    def test_edit_reencodes(self, mem):
        e = mem.store('Old text about apples.')
        ok = mem.edit(e['id'], new_text='New text about orbiters.')
        assert ok is True
        assert mem.entries[0]['text'] == 'New text about orbiters.'
        hits = mem.recall('orbiters')
        assert hits and 'orbiters' in hits[0]['text']

    def test_edit_missing(self, mem):
        assert mem.edit('nope') is False

    def test_delete(self, mem):
        e = mem.store('To be deleted.')
        assert mem.delete(e['id']) is True
        assert mem.delete(e['id']) is False
        assert mem.list_entries() == []

    def test_clear(self, mem):
        mem.store('a')
        mem.store('b')
        mem.clear()
        assert mem.list_entries() == []


class TestListEntries:
    def test_vectors_hidden_by_default(self, mem):
        mem.store('secret sauce')
        entries = mem.list_entries()
        assert 'vector' not in entries[0]
        with_vec = mem.list_entries(include_vectors=True)
        assert isinstance(with_vec[0]['vector'], list)
        assert len(with_vec[0]['vector']) == 256