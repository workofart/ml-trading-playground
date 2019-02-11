# Work-Trader

Simple trading bot that utilizes deep learning and reinforcement learning

---
## Simple Experiment
I wanted to first try to predict prices given the current, high, low prices and volume. To aid my learning process, I created two versions with the same purpose, one using primitive numpy package and out-of-the-box python code to implement a neural network, and the second version using third-party frameworks/libraries.

> Both neural network follow this architecture:
4 Layers, Relu Activation Function for first (n-1) layers, with last layer being linear output.

### Version 1
Hand-coded Neural Network (without using any 3rd party framework)


Sample Accuracy for predicting the price at the next timestamp (10-sec timestamps)

![Img](https://raw.githubusercontent.com/workofart/work-trader/master/v1/trainingset.png)

### Version 2
Keras-based Neural Network

![Img](https://raw.githubusercontent.com/workofart/work-trader/master/v2/trainingset.png)

### Version 3
Custom Neural Network with Reinforcement Learning (Deep Q Learning) using gym