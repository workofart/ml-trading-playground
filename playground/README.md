# Reinforcement Learning for trading

So far I've experimented with the following approaches:
- Policy Gradient (PG)
- Deep Q-Learning/Network (DQN)
- Actor Critic (AC)
- Advantage Asynchornous Actor Critic (A3C)

## To run
```
# For Policy Gradient
python -m playground.pg.pg

# For Deep Q-Learning/Network (DQN)
python -m playground.dqn.dqn

# For Actor Critic (AC)
python -m playground.ac.ac

# For Advantage Asynchornous Actor Critic (A3C)
python -m playground.a3c.a3c

# For Advantage Asynchornous Actor Critic (A3C) - New Implementation
python -m playground.a3c_new.a3c_new
```

## To view training logs
`tensorboard --logdir=playground/logs`
