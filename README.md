# Work-trader
There are 2 parts to this project. The first part is for me to learn deep learning and to implement neural networks from scratch. The second part is to learn reinforcement learning and apply and evaluate various techniques to teach an agent to trade.


## Creating Neural Networks with Python/Keras/Tensorflow to Predict the Future
I wanted to first try to predict prices given the _current, high, low prices and volume_.

To aid my learning process, I created 3 versions with the same purpose:
1. Primitive numpy package and out-of-the-box python code to implement a neural network
2. High-level frameworks (Keras)
3. Tensorflow

All 3 versions follow this architecture:
4 Layers
- **Relu Activation Function** for first (n-1) layers
- last layer being **linear output**.


[Version 1 - Raw Python](https://github.com/workofart/work-trader/tree/master/v1)

[Version 2 - Keras](https://github.com/workofart/work-trader/tree/master/v2)

[Version 3 - TensorFlow](https://github.com/workofart/work-trader/tree/master/v3)

Check out the [accompanying tutorial](http://www.henrypan.com/blog/machine-learning/2019/03/20/ml-tut-price-prediction.html)


## Using Reinforcement Learning to Trade

The common assumptions are:
- The agent is allowed to enter both long and short positions regardless of current cash balance

Approaches:
- [Policy Gradient (PG)](https://github.com/workofart/ml-trading-playground/tree/master/playground/pg)
- [Deep Q-Network/Learning (DQN)](https://github.com/workofart/ml-trading-playground/tree/master/playground/dqn)
- [Actor-Critic (AC)](https://github.com/workofart/ml-trading-playground/tree/master/playground/ac)
- [Asynchronous Advantage Actor-Critic (A3C)](https://github.com/workofart/ml-trading-playground/tree/master/playground/a3c)
