## New data validation
- Data Scientists build models based on initial set of data
- New data is continuously acquired for automated model retraining
- **New data needs to be monitored for changes that may impact modelling**

### Automated feature validation
- Basic feature validation
  1. Missing data and erroneous data
  2. Data formats (string, date, etc.)
- Data distribution validation
  1. Mean, SD, quartiles for continuous data
  2. Class distribution for categorical data
- Out-of-distribution validation
  1. Value beyond quartiles
  2. New class values
- Correlation validation
  1. Feature vs. Target
  2. Between features


## Managed feature stores
- A centralised store of features
- Pre-processed and ready for ML
- Shared across multiple teams and projects
- Regularly updated with new training data and features
- Registry for features - for users to understand the data that's available in the store

### Feature store: Best Practices
- Shared ownership with defined responsibilities
- Flexible schema for regular additions
- Loosely coupled datasets, yet linkable: may have multiple related datasets, recommended to keep them separate and not force a hard merge and denormalised data, as long as linking data are available.
- Updated registry for available data
- Flexibility for last mile post-processing: should keep a common format as much as possible, let the teams customise when they query for training. Multiple technologies as needed, low cost as possible.


## Data versioning - key aspect of training lineage in data
Managing training data with a data versioning system provides the ability to change data continuously while ensuring to change data continuously while ensuring consistent training results and collaboration.

- Creates an immutable baseline for datasets (raw, intermediate, feature)
- Version changes when contents change
- Version at a feature, record, or dataset level
- Reference specific version for model training and testing
- Different teams can consume different versions of the same dataset

### Benefits of Data Versioning
- Traceability - dataset to model or experiment version mapping
- Reproducibility - recreate the same ML results again
- Rollback to last known good state
- Multiple users can reference different versions
- Change log capture alternative
- Multiple users can reference different versions
