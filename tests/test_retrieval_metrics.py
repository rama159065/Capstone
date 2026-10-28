import math
import pytest
from fastapi.testclient import TestClient
from scripts.domain_common.retrieval_metrics import retrieval_metrics, response_metrics


def test_fractional_recall_and_linear_graded_ndcg():
    scores = retrieval_metrics(['b', 'x', 'a'], {'a': 3, 'b': 1, 'c': 1})
    assert scores['recall@1'] == pytest.approx(1 / 3)
    assert scores['recall@5'] == pytest.approx(2 / 3)
    assert scores['recall@10'] == pytest.approx(2 / 3)
    assert scores['ndcg@5'] == pytest.approx((1 + 3 / 2) / (3 + 1 / math.log2(3) + 1 / 2))


def test_duplicates_missing_and_empty_judgments():
    scores = retrieval_metrics(['a', 'a'], {'a': 1, 'b': 1})
    assert scores['recall@5'] == .5
    assert scores['ndcg@5'] < 1
    assert retrieval_metrics([], {})['ndcg@5'] == 0
    assert retrieval_metrics(['a'], None)['recall@1'] is None
    with pytest.raises(ValueError):
        retrieval_metrics([], {'a': float('nan')})
    assert response_metrics([{'chunk_id': 'a'}], {'a': 1}, top_k=1)['recall@10'] is None


def test_metrics_across_rag_routes(monkeypatch):
    from scripts import api_app
    monkeypatch.setattr(api_app, 'retrieve_evidence', lambda config: [{'chunk_id': 'a'}])
    monkeypatch.setattr(api_app, 'retrieve_hybrid', lambda *args, **kwargs: [{'chunk_id': 'a'}])
    from scripts.domain_common import multimodal_rag as mm
    monkeypatch.setattr(mm, 'retrieve_local', lambda *args, **kwargs: [{'chunk_id': 'a'}])
    class FakeRag:
        def retrieve_visual(self, *args, **kwargs):
            return [{'chunk_id': 'a'}]
        def answer(self, *args, **kwargs):
            return {'evidence': [{'chunk_id': 'a'}], 'answer': 'test'}
    monkeypatch.setattr(mm, 'get_rag', lambda: FakeRag())
    with TestClient(api_app.app) as client:
        for path in ['/retrieve/financial_report_qa', '/retrieve/hr_policy',
                     '/multimodal/hr_policy/retrieve', '/multimodal/hr_policy/visual/retrieve',
                     '/multimodal/hr_policy/chat']:
            result = client.post(path, json={'query': 'What is the leave policy?', 'top_k': 10,
                                            'qrels': {'a': 1, 'b': 1}})
            assert result.status_code == 200, result.text
            assert result.json()['retrieval_metrics']['recall@1'] == .5
            assert result.json()['retrieval_metrics']['ndcg@5'] > 0
        invalid = client.post('/retrieve/hr_policy', json={'query': 'leave policy', 'qrels': {'a': -1}})
        assert invalid.status_code == 422

def test_chat_metrics_use_financial_ids_and_domain_evidence(monkeypatch):
    from scripts import api_app
    monkeypatch.setattr(api_app, 'run_technique', lambda *a, **kw: {})
    monkeypatch.setattr(api_app, 'summarize_result', lambda result: {
        'final_status': 'insufficient_evidence', 'evidence_chunk_ids': ['a']})
    monkeypatch.setattr(api_app, 'run_domain_agent', lambda *a, **kw: {
        'domain': 'hr_policy', 'final_status': 'insufficient_evidence',
        'evidence': [{'chunk_id': 'a'}]})
    with TestClient(api_app.app) as client:
        for path in ['/chat/financial_report_qa', '/chat/hr_policy']:
            response = client.post(path, json={'query': 'leave policy', 'top_k': 10,
                                               'relevant_chunk_ids': ['a', 'b']})
            assert response.status_code == 200, response.text
            assert response.json()['retrieval_metrics']['recall@10'] == .5
