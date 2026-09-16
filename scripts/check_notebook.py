"""Check the curated notebook without downloading data or training models."""

from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks" / "claim_verification.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    sources = []
    definitions = {}
    code_cells = 0

    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell["source"])
        assert not re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", source)
        if cell["cell_type"] != "code":
            continue
        code_cells += 1
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        tree = ast.parse(source, filename=f"cell-{index}")
        compile(tree, f"cell-{index}", "exec")
        sources.append(source)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                definitions[node.name] = node

    all_code = "\n".join(sources)
    assert "RUN_TEST_SUBMISSION = False" in all_code
    assert "DOWNLOAD_DATA = False" in all_code
    assert "final_trainer.evaluate()" not in all_code
    assert "eval_dataset=final_dev_ds" not in all_code

    # Execute only the original metric function, with synthetic evidence IDs.
    # Model loading, package installation, downloads and training never execute.
    metric = ast.unparse(definitions["stage1_recall_metrics"])
    namespace = {
        "np": SimpleNamespace(mean=lambda values: sum(values) / len(values)),
        "CANDIDATE_POOL_SIZE_GRID": [1, 2],
    }
    exec("from __future__ import annotations\n" + metric, namespace)
    gold = {"c1": {"evidences": ["a", "b"]}, "c2": {"evidences": ["c"]}}
    candidates = {
        "c1": [{"evidence_id": "a"}, {"evidence_id": "x"}],
        "c2": [{"evidence_id": "z"}, {"evidence_id": "c"}],
    }
    result = namespace["stage1_recall_metrics"](gold, candidates)
    expected = {
        "@1": (0.25, 0.5, 0.0),
        "@2": (0.75, 1.0, 0.5),
    }
    fields = ("mean_gold_recall", "hit_rate_at_least_one_gold", "all_gold_recovered_rate")
    for cutoff, values in expected.items():
        for field, value in zip(fields, values):
            assert math.isclose(result[cutoff][field], value), (cutoff, field)

    # Confirm the compatibility helper preserves the explicit no-evaluation guard.
    import inspect

    class FakeTrainingArguments:
        def __init__(self, eval_strategy="no", load_best_model_at_end=False):
            self.eval_strategy = eval_strategy

    helper_namespace = {"inspect": inspect, "TrainingArguments": FakeTrainingArguments}
    helper = ast.unparse(definitions["make_training_args"])
    exec("from __future__ import annotations\n" + helper, helper_namespace)
    arguments = helper_namespace["make_training_args"]({"eval_strategy": "no"})
    assert arguments.eval_strategy == "no"

    try:
        import nbformat
    except ImportError:
        schema_status = "not run (nbformat is not installed)"
    else:
        nbformat.validate(nbformat.from_dict(notebook))
        schema_status = "passed"

    print(f"Notebook syntax: {code_cells} code cells passed")
    print("English source and cleared outputs: passed")
    print("Synthetic retrieval-metric semantics: passed")
    print("Final-fit evaluation guard: passed")
    print(f"Notebook schema: {schema_status}")
    print("Full indexing, model loading and training: not run")


if __name__ == "__main__":
    main()
