
## Training Time
100%|█████████████| 8000/8000 [8:58:31<00:00,  4.21s/it]

## Hyperparameters

Main Parameters
```python
EPISODE = 8000 # Episode limitation
TRAIN_EVERY_STEPS = 16
TEST_EVERY_EP = 50
DATA_LENGTH = 3000 # How many times steps to use in the data
```

Neural Network Parameters
```python
NN1_NEURONS = 32
NN2_NEURONS = 16
beta = 0.01 # regularization
dropout = 0.03 # dropout
UPDATE_TARGET_NETWORK = 10
BATCH_SIZE = 128 # size of minibatch
```

Training Parameters
```python
LEARNING_RATE = 1e-4
INITIAL_EPSILON = 1 # starting value of epsilon
FINAL_EPSILON = 0.05 # ending value of epislon
GAMMA = 0.95 # discount factor for q value
```

Trading Parameters
```python
HOLD_PENALTY = 0
TXN_COST = 0.002
```

## Reward Function

```python
def process_action(self, action, state):
    cur_price = state[0][2]
    error = False
    # Buy
    if action == 0:
        if cur_price > self.cash:
            error = True
        else:
            self.cash -= cur_price * (1+TXN_COST)# buy with current price of current state
            # self.portfolio += 1
            self.portfolio.append(cur_price)
    # Sell
    elif action == 1:
        if self._get_holding_stock_count() == 0:
            error = True
        else:
            self.cash += cur_price * (1-TXN_COST)
            self.portfolio.append(-cur_price)
    # Hold
    elif action == 2:
        self.cash = self.cash - HOLD_PENALTY
        self.portfolio = self.portfolio

    book_val = self.cash + self._avg_holding_price() * self._get_holding_stock_count()
    market_val = self.cash + cur_price * self._get_holding_stock_count()
    
    reward = book_val + (market_val - self.previous_portfolio)

    if error:
        reward -= 0.03

    # Previous reward should be the previous portfolio market value, not the difference
    self.previous_portfolio = market_val
    return reward
```