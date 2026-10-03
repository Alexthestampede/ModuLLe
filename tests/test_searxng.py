"""Unit tests for the SearXNG backend and tool wrapper (all offline, mocked)."""

from unittest.mock import MagicMock, patch

import pytest
from requests import RequestException

from modulle.web.searxng import SearxngSearcher, DEFAULT_SEARXNG_URL
from modulle.web.searxng_tools import SearchSearxngTool


def _mock_response(payload, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    resp.raise_for_status.side_effect = None
    return resp


class TestSearxngSearcherInit:
    def test_default_url_is_dummy_localhost(self):
        with patch.dict('os.environ', {}, clear=True):
            s = SearxngSearcher()
        assert s.base_url == DEFAULT_SEARXNG_URL
        assert s.base_url == 'http://localhost:8888'

    def test_env_var_fallback(self):
        with patch.dict('os.environ', {'SEARXNG_BASE_URL': 'http://env:9999/'}):
            s = SearxngSearcher()
        assert s.base_url == 'http://env:9999'

    def test_trailing_search_path_is_stripped(self):
        s = SearxngSearcher('http://x:8080/search')
        assert s.base_url == 'http://x:8080'

    def test_trailing_slash_stripped(self):
        s = SearxngSearcher('http://x:8080/')
        assert s.base_url == 'http://x:8080'


class TestSearxngSearch:
    def _searcher(self, **kw):
        return SearxngSearcher('http://searx:8888', **kw)

    @patch('modulle.web.searxng.requests.get')
    def test_basic_search_parses_results(self, mock_get):
        mock_get.return_value = _mock_response({
            'results': [
                {'title': 'A', 'url': 'http://a', 'content': 'aaa', 'engine': 'ddg', 'score': 5.0},
                {'title': 'B', 'url': 'http://b', 'content': 'bbb'},
            ]
        })
        results = self._searcher().search('q', max_results=5)
        assert len(results) == 2
        assert results[0] == {'title': 'A', 'url': 'http://a', 'snippet': 'aaa',
                              'engine': 'ddg', 'score': 5.0}
        assert results[1]['engine'] == ''
        # format=json must always be requested
        assert mock_get.call_args.kwargs['params']['format'] == 'json'

    @patch('modulle.web.searxng.requests.get')
    def test_max_results_cap(self, mock_get):
        payload = {'results': [{'title': str(i), 'url': '', 'content': ''} for i in range(20)]}
        mock_get.return_value = _mock_response(payload)
        results = self._searcher().search('q', max_results=5)
        assert len(results) == 5

    @patch('modulle.web.searxng.requests.get')
    def test_default_categories_language_safesearch_sent(self, mock_get):
        mock_get.return_value = _mock_response({'results': []})
        s = self._searcher(categories='it', language='it', safesearch=2)
        s.search('q')
        params = mock_get.call_args.kwargs['params']
        assert params['categories'] == 'it'
        assert params['language'] == 'it'
        assert params['safesearch'] == 2

    @patch('modulle.web.searxng.requests.get')
    def test_call_overrides(self, mock_get):
        mock_get.return_value = _mock_response({'results': []})
        s = self._searcher(categories='general', safesearch=1)
        s.search('q', categories='news', language='en', safesearch=0)
        params = mock_get.call_args.kwargs['params']
        assert params['categories'] == 'news'
        assert params['language'] == 'en'
        assert params['safesearch'] == 0

    @patch('modulle.web.searxng.requests.get')
    def test_non_json_response_raises_valueerror(self, mock_get):
        # requests raises JSONDecodeError (subclass of ValueError) for bad JSON
        import requests as _requests
        resp = MagicMock()
        resp.raise_for_status.side_effect = None
        try:
            import json as _json
            _json.loads('<html>')
            _decode_err = ValueError('not json')
        except _json.JSONDecodeError as e:
            _decode_err = e
        resp.json.side_effect = _decode_err
        mock_get.return_value = resp
        with pytest.raises(ValueError, match='did not return JSON'):
            self._searcher().search('q')

    @patch('modulle.web.searxng.requests.get')
    def test_connection_error_propagates(self, mock_get):
        mock_get.side_effect = RequestException('refused')
        with pytest.raises(RequestException):
            self._searcher().search('q')

    @patch('modulle.web.searxng.requests.get')
    def test_is_available_true(self, mock_get):
        mock_get.return_value = _mock_response({}, status_code=200)
        assert self._searcher().is_available() is True

    @patch('modulle.web.searxng.requests.get')
    def test_is_available_false_on_error(self, mock_get):
        mock_get.side_effect = RequestException('down')
        assert self._searcher().is_available() is False


class TestSearchSearxngTool:
    def test_tool_metadata(self):
        tool = SearchSearxngTool(SearxngSearcher('http://x'))
        assert tool.get_name() == 'search_web_searxng'
        schema = tool.get_parameters()
        assert schema['required'] == ['query']
        assert 'query' in schema['properties']
        # must fit the standard provider schemas
        assert 'search_web_searxng' in str(tool.to_ollama_schema())

    @patch('modulle.web.searxng.requests.get')
    def test_execute_formats_results(self, mock_get):
        mock_get.return_value = _mock_response({
            'results': [{'title': 'T', 'url': 'http://u', 'content': 'C'}]
        })
        tool = SearchSearxngTool(SearxngSearcher('http://x'))
        out = tool.execute('q', 3)
        assert 'T' in out and 'http://u' in out and 'C' in out

    @patch('modulle.web.searxng.requests.get')
    def test_execute_no_results(self, mock_get):
        mock_get.return_value = _mock_response({'results': []})
        tool = SearchSearxngTool(SearxngSearcher('http://x'))
        assert 'No results' in tool.execute('q')

    @patch('modulle.web.searxng.requests.get')
    def test_execute_error_returned_as_string(self, mock_get):
        mock_get.side_effect = RequestException('down')
        tool = SearchSearxngTool(SearxngSearcher('http://x'))
        out = tool.execute('q')
        assert out.startswith("Error searching for 'q'")