# Source provenance and cleanup

## Source selection

The implementation comes from `COMP90042_team32_code.ipynb` in the user-supplied
`COMP90042_team32_resource.zip`. The zip's README identifies it as the final
three-stage submission. A separately supplied notebook was archived as an
alternate experiment and is not duplicated in the public-facing repository.

The original notebook, zip, README, report-related materials and selected metric
artifacts were copied unchanged to a separate local preservation directory.
Original paths, file sizes and SHA-256 values are recorded in a local manifest.
That manifest and original execution outputs are not part of this repository.

## Changes in this copy

- Renamed the notebook and added a task-focused introduction.
- Removed execution outputs and execution counts from the curated copy.
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

The 1.2-million-passage index and GPU training were not rerun. Lightweight
validation checks syntax, notebook structure, output removal and retrieval-metric
semantics using synthetic inputs.
