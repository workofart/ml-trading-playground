Q: Check the convergence of the actor's trades. If no action is taken, try removing the transaction cost and retry.

A: even after removing the transaction cost, the agent still converges to no action

Q: Try to reduce the number of neurons per layer of NN, because the agent is performing all buys

A: reducing the neurons from 1024 to 256, 512 to 128 allowed the agent to start with buy and sell actions instead of all buys. Actually, reducing the learning rate also allowed the agent to become more dispersed with its actions

Q: Try to record rewards for each worker independently


Q: What does allow_soft_placement and log_device_placement mean?

A:
```
  // Whether soft placement is allowed. If allow_soft_placement is true,
  // an op will be placed on CPU if
  //   1. there's no GPU implementation for the OP
  // or
  //   2. no GPU devices are known or registered
  // or
  //   3. need to co-locate with reftype input(s) which are from CPU.
```

Q: How does adding entropy to the loss encourage exploration?
A: Intuitively, it adds more cost to actions that too quickly dominate, and the higher cost favors more exploration (on top of the random e-greediness).

You have overlooked the minus sign before the whole equation I think:
> [self.losses = - (tf.log(self.picked_action_probs) * self.targets + 0.01 * self.entropy)](https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/a3c/estimators.py#L86)

As you know it is equivalent to:
> self.losses = -tf.log(self.picked_action_probs) * self.targets - 0.01 * self.entropy

In the original paper Williams et. al. where **maximising** expected reward. In DeepMind implementation Mnih et. al. have decided to use **minimisation** algorithm, so they added minus before the whole equation.