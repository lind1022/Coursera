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

### Code smell
Introducing code that we're unable to inspect or easily modify into testing and production frameworks.


## Concept drift:
ML algorithm assumptions:
1. Instances are generated at random according to some probability distribution D
2. Instances are independent and identically distributed
3. D is stationary with fixed distributions

**Drift is the change in an entity with respect to a baseline**
Production can diverge or drift from the baseline data over time due to changes in the real world.

### Types of drift in ML models:
- Data drift: a change in P(X) is a shift in the model's input data distribution. E.g. incomes of all applicants increased by 5% but the economic fundamentals are the same.
- Concept drift: a change in P(Y/X) is a shift in the actual relationship between the model inputs and the output. E.g. macro economic factors make the lending riskier, and there is a higher standard to be eligible for a loan. A income was previously considered creditworthy is no longer creditworthy.
2 types of concept drift:
1. Stationary supervised learning: model trained only on historical data
2. Learning under concept drift: a new secondary data source is injected to provide both historical and new data to make prediction, the new data can be in batch or real time. Statistical properties of the target variable may change over time.

**Concept drift** occurs when the distribution of our observations shifts over time, or that the joint probability distribution changes.

4 types of concept drift:
- Sudden drift; a new concept occurs within a short time.
- Gradual drift: a new concept gradually replaces an old one over a period of time.
- Incremental drift; an old concept incrementally changes to a new concept over a period of time.
- Recurring concepts: an old concept may reoccur after some time.

- Prediction drift: a change in Y(y_hat|X) is a shift in the model's predictions. E.g. a large number of creditworthy applications when the production is launched in a more affluent area. Model still holds, but business maybe unprepared.
- Label drift: a change in P(Y Ground Truth) is a shift in the model's output or label distribution.















end of script
