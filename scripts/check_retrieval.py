"""Exercise the notebook's TF-IDF and cache handling on a tiny synthetic corpus."""

from __future__ import annotations

import ast
import gc
import json
from pathlib import Path
import re
import string
import tempfile
import time
import unicodedata

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


def main():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / 'notebooks/claim_verification.ipynb').read_text(encoding='utf8'))
    definitions = {}
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            for node in ast.parse(''.join(cell['source'])).body:
                if isinstance(node, ast.FunctionDef):
                    definitions[node.name] = node
    selected = (
        'load_json', 'save_json', 'load_claims', 'load_evidence', 'clean_text',
        'normalize_for_stage1', 'file_sha256', 'stage1_meta', 'tfidf_cache_valid',
        'load_evidence_lists', 'build_or_load_tfidf', 'topk_from_sparse_scores',
        'tfidf_search_one', 'minmax_normalize', 'retrieve_stage1_candidates_for_claim',
        '_minmax', 'ce_evidence_text', 'run_retrieval', 'run_pipeline',
    )
    source = '\n\n'.join(ast.unparse(definitions[name]) for name in selected)
    with tempfile.TemporaryDirectory(prefix='claim-retrieval-check-') as directory:
        work = Path(directory)
        evidence = {
            'e0': 'Solar panels generate electricity from sunlight.',
            'e1': 'Wind turbines generate electricity from moving air.',
            'e2': 'A dog plays beside the wooden fence.',
            'e3': 'Ocean water contains dissolved salts.',
            'e4': 'Sunlight reaches solar panels during the day.',
        }
        claims = {'c0': {'claim_text': 'Solar panels generate electricity.', 'claim_label': 'SUPPORTS', 'evidences': ['e0']}}
        evidence_path = work / 'evidence.json'
        claims_path = work / 'claims.json'
        evidence_path.write_text(json.dumps(evidence), encoding='utf8')
        claims_path.write_text(json.dumps(claims), encoding='utf8')

        class TinyReranker:
            """Deterministic local scorer used only to test pipeline wiring."""
            batch_size = 4
            calls = 0
            def __init__(self, **kwargs):
                pass
            def score(self, pairs, **kwargs):
                type(self).calls += 1
                return np.array([float('sunlight' in passage) for _, passage in pairs])

        namespace = dict(globals(), _ujson=None, _WS_RE=re.compile(r'\s+'),
            _SUBSCRIPT_DIGITS=str.maketrans({chr(0x2080+i): str(i) for i in range(10)}),
            _DEGREE_SIGN=chr(176), EVIDENCE_FILE=evidence_path,
            STAGE1_METHOD='tfidf_hybrid_word_char', STAGE1_CANDIDATE_POOL_SIZE=3,
            WORD_NGRAM_RANGE=(1,2), WORD_MIN_DF=1, WORD_MAX_DF=1.0, WORD_MAX_FEATURES=200,
            CHAR_ANALYZER='char_wb', CHAR_NGRAM_RANGE=(3,5), CHAR_MIN_DF=1,
            CHAR_MAX_DF=1.0, CHAR_MAX_FEATURES=400, WORD_WEIGHT=1.0, CHAR_WEIGHT=0.7,
            CE_MODEL='synthetic-test-scorer', CE_RERANK_WEIGHT=1.0, CE_MAX_LEN=192,
            CE_EVIDENCE_CHAR_LIMIT=1200, K_RETRIEVE=2, CLASSIFIER_DIR=work/'classifier',
            CrossEncoderReranker=TinyReranker)
        for key, filename in {
            'WORD_VECTORIZER_FILE':'word.joblib', 'WORD_MATRIX_FILE':'word.npz',
            'CHAR_VECTORIZER_FILE':'char.joblib', 'CHAR_MATRIX_FILE':'char.npz',
            'STAGE1_EVIDENCE_IDS_FILE':'ids.json', 'STAGE1_META_FILE':'meta.json',
        }.items():
            namespace[key] = work / filename
        exec('from __future__ import annotations\n' + source, namespace)
        ids, word_vectorizer, word_matrix, char_vectorizer, char_matrix = namespace['build_or_load_tfidf']()
        assert namespace['tfidf_cache_valid']()
        candidates = namespace['retrieve_stage1_candidates_for_claim'](
            claims['c0']['claim_text'], ids, word_vectorizer, word_matrix,
            char_vectorizer, char_matrix, pool_size=3,
        )
        assert candidates[0]['evidence_id'] == 'e0'
        kwargs = dict(claims_path=claims_path, out_path=work/'retrieved.json', evidence=evidence,
            evidence_ids=ids, word_vectorizer=word_vectorizer, word_matrix=word_matrix,
            char_vectorizer=char_vectorizer, char_matrix=char_matrix, stage1_pool_size=3, k=2)
        first = namespace['run_retrieval'](**kwargs)
        assert len(first['c0']) == 2 and 'e0' in first['c0']
        assert TinyReranker.calls == 1
        assert namespace['run_retrieval'](**kwargs) == first
        assert TinyReranker.calls == 1, 'Verified cache should skip reranking.'
        kwargs['k'] = 1
        assert len(namespace['run_retrieval'](**kwargs)['c0']) == 1
        assert TinyReranker.calls == 2, 'Changing evidence depth must invalidate retrieval.'
        evidence['e2'] = 'Changed corpus content at the same input path.'
        evidence_path.write_text(json.dumps(evidence), encoding='utf8')
        assert not namespace['tfidf_cache_valid'](), 'Corpus changes must invalidate TF-IDF.'
        namespace['predict_labels'] = lambda *args, **kwargs: {}
        try:
            namespace['run_pipeline'](claims_path, work/'predictions.json', {'c0':['e0']})
        except ValueError as error:
            assert 'classifier prediction is missing' in str(error)
        else:
            raise AssertionError('Incomplete classification must not create a default label.')
    print('PASS: actual TF-IDF indexing/ranking, cache reuse/invalidation and prediction coverage')
    print('Cross-encoder and classifier weights/training were not loaded.')


if __name__ == '__main__':
    main()
