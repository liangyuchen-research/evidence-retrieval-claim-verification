# Data setup

The source corpus, labels, prediction files and model artifacts are excluded
from this repository. Obtain course data through an authorized source and check
its redistribution terms independently.

The original notebook references the University of Melbourne course repository:
[COMP90042 2026](https://github.com/drcarenhan/COMP90042_2026).
It also references a course Google Drive download for `evidence.json`; the
existing downloader remains in the notebook behind `DOWNLOAD_DATA = False`.
Availability of those external sources has not been revalidated.

Place these files in this directory:

| File | Expected structure |
| --- | --- |
| `evidence.json` | Object mapping evidence IDs to passage strings |
| `train-claims.json` | Claim ID to text, label and gold evidence IDs |
| `dev-claims.json` | Same labelled schema, used for model selection and development evaluation |
| `test-claims-unlabelled.json` | Claim ID to text, without public target labels |
| `dev-claims-baseline.json` | Official baseline predictions for development comparison |

The recorded source run used 1,228 training claims, 154 development claims,
153 test claims and 1,208,827 evidence passages. Dataset counts alone do not
establish a dataset license or guarantee that an external copy is identical.

## Schema examples

The following strings are invented examples, not assignment answers.

```json
{
  "evidence-demo": "A hypothetical laboratory recorded a temperature of 20 C."
}
```

```json
{
  "claim-demo": {
    "claim_text": "The hypothetical measurement was 20 C.",
    "claim_label": "SUPPORTS",
    "evidences": ["evidence-demo"]
  }
}
```

The final classifier is trained on gold evidence, while end-to-end inference
uses retrieved evidence. That input difference is part of the original design
and contributes to the gap between the reported evaluation settings.
