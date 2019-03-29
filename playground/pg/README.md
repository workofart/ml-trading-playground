Policy Gradient (PG) Approach to the trading problem

# Run
`python -m playground.pg.pg`

# Logs
Logs are stored in `playground/logs/pg`

# Trained Models
The checkpoints for trained models are stored in `playground/saved_networks/pg`


# Challenges
Not sure how to tweak the Gamma hyperparameter.

Use cross-entropy for classification instead of MSE because it takes into account how far off your predictions probabilities are from the actual predictions not just the number of correct/incorrect predictions.