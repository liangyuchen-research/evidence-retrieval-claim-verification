# Evaluation and limitations

## Candidate retrieval

The source notebook's development set contains 154 claims. At a cutoff of 500:

| Metric | Recorded value | Meaning |
| --- | --- | --- |
| At-least-one-gold hit rate | 0.8896103896 | Fraction of claims with at least one gold passage in the pool |
| Mean gold-evidence recall | 0.6207792208 | Mean fraction of a claim's gold passages recovered |
| All-gold recovery rate | 0.3506493506 | Fraction of claims whose complete gold evidence set is recovered |

The headline 89% is the rounded first value. It is not precision@500, complete
evidence recall, classifier accuracy, or a hidden test score.

## Recorded final-submission notebook run

The following results come from saved outputs in the supplied final notebook,
not from a fresh run of this curated copy:

| Evaluation setting | Recorded result |
| --- | --- |
| Train-only classifier, development gold evidence | Accuracy 0.8052, macro-F1 approximately 0.754 |
| End-to-end pipeline, development retrieved evidence | Claim accuracy 0.512987 |
| End-to-end pipeline, development retrieved evidence | Evidence F1 0.195068 |
| End-to-end pipeline, harmonic mean of accuracy and evidence F1 | 0.282654 |
| Official development baseline, harmonic mean | 0.344089 |

The reranker selects K = 3 using development evidence F1. Classifier checkpoint
selection also uses development accuracy. These development results are used
for tuning and should not be represented as an untouched test evaluation.
The recorded pipeline did not outperform the provided baseline on the combined
metric. Gold-evidence classifier scores are a different, easier evaluation
setting and must not be substituted for end-to-end results.

## Train+development overlap

The historical notebook fits a final classifier on train+development labels and
then evaluates it on that same development set. Its 100% development result is
an overlapping-training diagnostic and is not evidence of generalization.

The curated notebook disables final test-submission generation by default.
When enabled, the final fit uses the epoch count selected earlier, saves the
last model, and performs no overlapping development evaluation or checkpoint
selection. This evaluation hygiene change has not been validated by a fresh
full training run, and old output metrics do not describe the modified fit.

## Multiple local runs

A separate local artifact directory contains a train-only classifier accuracy
of 0.662338 and a selected epoch count of 11. The final-submission notebook
records a different classifier run and selected epoch count of 13. These outputs
are kept separate in the local provenance archive. Results from different runs
are not combined into one claimed experiment.

## Reproducibility limits

- The exact installed dependency versions were not captured in the original package.
- The original logs include model-loading warnings, so historical results require
  rerunning in a recorded environment before independent verification.
- TF-IDF caches check configuration and input path but do not fingerprint every
  corpus byte. Clear or choose a new artifact directory when the corpus changes.
- The cross-encoder is pretrained, not fine-tuned by this project.
- The corpus and pretrained model files are external dependencies.
- No hidden-test ground truth or verified hidden-test score was supplied.
