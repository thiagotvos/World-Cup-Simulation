# Evaluation Plan

## Model Evaluation

The classifier is evaluated on the test split using:

- Goal MAE
- Goal RMSE
- Match-result accuracy derived from sampled scorelines
- Confusion matrix on win/draw/loss outcomes
- Log loss on derived win/draw/loss probabilities

Goal errors measure how close the predicted scorelines are to the observed scores. Result-level metrics measure how well the model supports tournament simulation.

## Baseline Evaluation

The MLP is compared against a sportsbook-odds baseline.

Sportsbook odds are converted into implied win/draw/loss probabilities and evaluated with the same result-level metrics:

- Accuracy
- Log loss
- Macro F1-score

This comparison shows whether the learned model performs better than a market-derived benchmark.

## Validation Strategy

The training process uses a validation split from the training data.

Validation loss is monitored during training to support early stopping and reduce overfitting.

## Simulation Evaluation

The simulation layer is evaluated by checking whether it produces realistic tournament behavior:

- Strong teams advance more frequently
- Draws occur with non-zero probability
- Goal difference affects group ranking
- Third-place qualification is resolved correctly
- Repeated runs produce stable aggregate probabilities

## Interpretation Outputs

The final report will include:

- the learned feature behavior
- the final training curves
- the confusion matrix
- predicted and sampled scorelines
- simulated advancement probabilities
- the comparison between the MLP and sportsbook-odds baseline

## Success Criteria

The project is successful if all of the following are true:

- The MLP trains on the prepared dataset
- The model produces plausible score predictions
- The tournament simulation runs end to end
- The frontend can display the simulation step by step
- The final documentation matches the implemented pipeline
