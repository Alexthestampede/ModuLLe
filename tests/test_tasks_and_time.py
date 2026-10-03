"""Tests for time injection + task routing (offline)."""

from datetime import datetime, timezone

import pytest

from modulle.context.time_inject import inject_time, format_now, DEFAULT_TIME_TEMPLATE
from modulle.tasks import TaskRouter, ModelRole, TASK_ROLES


class TestTimeInject:
    def test_empty_list_creates_system_message(self):
        out = inject_time([])
        assert len(out) == 1
        assert out[0]['role'] == 'system'
        assert out[0]['content'].startswith('Current date and time:')

    def test_appends_to_existing_system(self):
        msgs = [{'role': 'system', 'content': 'You are helpful.'},
                {'role': 'user', 'content': 'hi'}]
        out = inject_time(msgs)
        assert out[0]['content'].startswith('Current date and time:')
        assert 'You are helpful.' in out[0]['content']
        # original list untouched
        assert 'Current date' not in msgs[0]['content']

    def test_idempotent_replaces_old_line(self):
        msgs = [{'role': 'system', 'content': 'Base prompt.'},
                {'role': 'user', 'content': 'hi'}]
        t1 = inject_time(msgs, now=datetime(2026, 1, 1, 10, 0))
        t2 = inject_time(t1, now=datetime(2026, 1, 2, 11, 30))
        content = t2[0]['content']
        assert content.count('Current date and time:') == 1
        assert '2026-01-02 11:30' in content
        assert '2026-01-01' not in content
        assert 'Base prompt.' in content

    def test_no_system_message_inserts_one(self):
        msgs = [{'role': 'user', 'content': 'hi'}]
        out = inject_time(msgs)
        assert out[0]['role'] == 'system'

    def test_format_now_template(self):
        dt = datetime(2026, 10, 3, 14, 30, tzinfo=timezone.utc)
        line = format_now(now=dt)
        assert '2026-10-03 14:30' in line
        assert 'Saturday' in line
        custom = format_now(now=dt, template='Today: {now}')
        assert custom.startswith('Today: ')

    def test_custom_template(self):
        out = inject_time([], now=datetime(2026, 3, 1, 8, 0),
                          template='Now: {now}')
        assert out[0]['content'].startswith('Now: ')


class TestTaskRouter:
    def test_unknown_role_rejected(self):
        r = TaskRouter()
        with pytest.raises(ValueError):
            r.set('nonexistent', model='x')

    def test_effective_defaults_to_main(self):
        r = TaskRouter()
        eff = r.effective('title', main_provider='ollama', main_model='l3')
        assert (eff.provider, eff.model) == ('ollama', 'l3')

    def test_effective_overrides(self):
        r = TaskRouter()
        r.set('title', provider='openai', model='gpt-4o-mini',
              api_key='sk-x', base_url='http://x')
        eff = r.effective('title', main_provider='ollama', main_model='l3')
        assert eff.provider == 'openai' and eff.model == 'gpt-4o-mini'
        assert eff.api_key == 'sk-x'

    def test_partial_override_keeps_main_provider(self):
        r = TaskRouter()
        r.set('compress', model='small-model')
        eff = r.effective('compress', main_provider='ollama', main_model='l3',
                          main_api_key=None, main_base_url='http://localhost:11434')
        assert eff.provider == 'ollama'
        assert eff.model == 'small-model'
        assert eff.base_url == 'http://localhost:11434'

    def test_vision_defaults_main(self):
        r = TaskRouter()
        eff = r.effective('vision', main_provider='ollama', main_model='v-model')
        assert eff.model == 'v-model'
        r.set('vision', model='llava')
        assert r.effective('vision', 'ollama', 'v-model').model == 'llava'

    def test_clear_falls_back(self):
        r = TaskRouter()
        r.set('title', model='tiny')
        r.clear('title')
        eff = r.effective('title', 'ollama', 'main')
        assert eff.model == 'main'

    def test_roundtrip_serialization(self):
        r = TaskRouter()
        r.set('title', model='tiny', provider='ollama')
        r.set('compress', model='comp')
        d = r.to_dict()
        r2 = TaskRouter.from_dict(d)
        assert r2.resolve('title').model == 'tiny'
        assert r2.resolve('compress').model == 'comp'
        assert r2.resolve('vision').model is None

    def test_from_dict_ignores_unknown_roles(self):
        r = TaskRouter.from_dict({'bogus': {'model': 'x'}})
        assert r.roles == {}

    def test_from_dict_none(self):
        assert TaskRouter.from_dict(None).roles == {}

    def test_task_roles_constant(self):
        assert set(TASK_ROLES) == {'main', 'title', 'compress', 'vision'}