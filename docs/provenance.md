# Source and maintenance notes

## Source selection

The implementation comes from `COMP90042_team32_code.ipynb` in the final
`COMP90042_team32_resource.zip` submission. Its README describes the three-stage
pipeline: TF-IDF retrieval, cross-encoder reranking, and claim classification.

The submitted notebook and recorded outputs are retained in a separate archive.
The results in [evaluation.md](evaluation.md) refer to that recorded run.

## Maintenance changes

- Renamed the notebook and added a task-focused introduction.
- Removed execution outputs and execution counts.
- Kept English source comments and removed transient notebook metadata.
- Resolved repository-relative paths and moved generated files under `artifacts/`.
- Extracted declared dependencies into `requirements.txt`.
- Made downloading course data and the final train+development fit explicit opt-ins.
- Removed the unused download of the official evaluation script. The original
  internal evaluator remains present and is attributed in `NOTICE.md`.
- Preserved candidate retrieval, reranking and train-only classification logic.
- Disabled evaluation on overlapping development data during the optional final
  train+development fit and documented the historical metric limitation.
- Prevented silently replacing an existing submission archive.
- Made notebook package installation optional and respected `CLAIM_PROJECT_ROOT`
  during setup.
- Required retrieved evidence explicitly at inference time. Missing retrieval
  now raises an error instead of falling back to gold evidence or an arbitrary ID.
- Made the default evidence depth follow the selected K at call time.
- Required complete, valid predictions before evaluation instead of silently
  scoring only the available subset.

The 1.2-million-passage index and GPU training were not rerun. Lightweight
validation checks syntax, notebook structure, output removal, retrieval metrics,
evidence handling and complete-prediction evaluation using synthetic inputs.
