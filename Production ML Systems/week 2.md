## Adapting to change

These often change:

- An upstream model
- A data source maintained by another team
  - To fix, we should stop consuming data from a source that doesn't notify downstream consumers.
  - Maintain a local version of the upstream model and keeping it updated
- The relationship between features and labels
- The distributions of inputs


## Features
- Features should always be examined before adding into the model
- All features should be subjective to leave-one-out evaluations to examine their importance

## Protect for feature & label distribution change
- Monitor
- Check residuals - difference between labels and predictions
- Emphasize data recency - treating more recent observations as more important by writing a custom loss function
- Retraining the model on most recent data

## Ablation analysis
Assess the value of a feature between comparing a model trained with a particular feature vs the model trained without it.

### Legacy features and bundled features
Legacy features: old features that were added because the were valuable at the time. Since then better features have been added which made them redundant without our knowledge.
Bundled features: features added as part of a bundle, which collectively are valuable but individually may not be.
Both features represent additional unnecessary data dependencies.
