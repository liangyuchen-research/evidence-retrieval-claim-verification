# Two-Stage Evidence Retrieval and Transformer-Based Claim Verification

A retrieval and classification pipeline for checking claims against
**1,208,827 evidence passages**, developed for Natural Language Processing
at the University of Melbourne.

**Source code:** [Pipeline notebook](notebooks/claim_verification.ipynb) · [Offline validation](scripts/check_notebook.py) · [Dependencies](requirements.txt)

The implementation is provided as a Jupyter notebook, including candidate
retrieval, cross-encoder reranking, classifier training, evaluation, and prediction
export. Dataset files and pretrained model weights are separate dependencies.

The system first retrieves candidate evidence with word and character TF-IDF,
reranks the candidates with a MiniLM cross-encoder, and classifies each claim
with a fine-tuned DeBERTa-v3 model. Word and character features provide broad
candidate coverage, while the cross-encoder scores claim-passage relevance
before classification.

## Pipeline

```mermaid
flowchart LR
    A[Claim] --> B[Word and character TF-IDF]
    P[Evidence corpus] --> B
    B --> C[500 candidate passages]
    C --> D[MiniLM cross-encoder]
    D --> E[Top K evidence passages]
    E --> F[DeBERTa-v3 classifier]
    A --> F
    F --> G[Claim label and supporting evidence IDs]
```

| Component | Implementation |
| --- | --- |
| Candidate retrieval | Weighted word unigrams/bigrams and character 3-5-grams |
| Candidate pool | 500 passages per claim |
| Reranking | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| Evidence selection | K = 2-7, selected using development retrieval F1 |
| Classification | `microsoft/deberta-v3-base`, class-weighted cross-entropy |
| Labels | Supports, refutes, not enough information, disputed |
| Tools | PyTorch, Hugging Face Transformers, scikit-learn, SciPy |

## Recorded results

| Development metric | Recorded value |
| --- | ---: |
| Top-500 candidate hit rate | 88.96% (137 of 154 claims) |
| Mean gold-evidence recall in the candidate pool | 62.08% |
| End-to-end claim accuracy with retrieved evidence | 51.30% |
| End-to-end evidence F1 | 19.51% |

Hit rate measures whether at least one reference passage appears among the
500 candidates. It is distinct from claim accuracy. These results come from
the saved submission run. The development set was used to select evidence
depth and classifier checkpoints, and training has not been repeated for this
release. [Evaluation notes](docs/evaluation.md) include the baseline comparison
and the separate gold-evidence classification results.

## Run locally

Use a separate Python environment and install a PyTorch build appropriate for
your hardware. From the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/check_notebook.py
jupyter lab notebooks/claim_verification.ipynb
```

Place the dataset files described in [data/README.md](data/README.md), review
the configuration cell, and run the notebook cells in order. Package installation
inside the notebook is disabled by default.

The notebook resolves paths from the repository or its `notebooks/` directory.
`CLAIM_PROJECT_ROOT` can override that root. Large matrices, predictions and
checkpoints are written to the ignored `artifacts/` directory. Downloading data
and training the optional final train+development classifier require explicit
configuration switches. The default workflow still includes expensive index
construction and classifier training.

The full evidence index requires substantial host memory and disk space, and
classifier training benefits from a GPU. Model weights and the source corpus
are external dependencies. The requirements file gives dependency ranges rather
than an exact lock of the original environment.

## Repository contents

- `notebooks/claim_verification.ipynb`: retrieval, training and evaluation.
- `docs/evaluation.md`: metric definitions and limitations.
- `docs/provenance.md`: source and maintenance notes.
- `data/README.md`: input schema and source links.
- `scripts/check_notebook.py`: notebook validation and synthetic regression checks.

Run `python scripts/check_notebook.py` for local checks without downloading
models or running training. Notebook schema validation additionally uses
`nbformat` when installed.

## Attribution and reuse

Developed as **Team 32, Natural Language Processing, 2026** at the University
of Melbourne. The implementation is maintained here by Liang-Yu Chen.
See [NOTICE.md](NOTICE.md) for third-party model and data attribution.
The project has no separate open-source license.
