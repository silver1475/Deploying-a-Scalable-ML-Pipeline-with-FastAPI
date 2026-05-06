# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a Random Forest classifier trained to predict whether an individual's income is greater than $50K per year based on census data. The model was trained as part of the Deploying a Scalable ML Pipeline with FastAPI project.

## Intended Use
The intended use of this model is educational. It is designed to demonstrate how to train, evaluate, and deploy a machine learning model using FastAPI. The model predicts income class as either `>50K` or `<=50K`.

## Training Data
The training data comes from the Census Income dataset. It includes demographic and employment-related features such as age, workclass, education, marital status, occupation, race, sex, hours per week, and native country.

## Evaluation Data
The evaluation data is a held-out test split from the same Census Income dataset used for training. This test set was used to measure model performance after training.

## Metrics
The model was evaluated using precision, recall, and F1 score.

- Precision: 0.7419
- Recall: 0.6384
- F1 Score: 0.6863

## Ethical Considerations
This model may reflect biases present in the Census Income dataset. Because the dataset includes sensitive demographic attributes, predictions may differ across groups in ways that are unfair or harmful. This model should not be used for real-world decision-making in areas such as hiring, lending, insurance, or law enforcement.

## Caveats and Recommendations
This model is limited by the quality and age of the dataset, and by any bias present in the original data. Its predictions should be interpreted cautiously. Before any real-world use, the model should undergo fairness assessment, bias testing across demographic slices, and more extensive validation.
