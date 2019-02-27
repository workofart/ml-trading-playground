Q: Check the convergence of the actor's trades. If no action is taken, try removing the transaction cost and retry.

A: even after removing the transaction cost, the agent still converges to no action

Q: Try to reduce the number of neurons per layer of NN, because the agent is performing all buys

A: reducing the neurons from 1024 to 256, 512 to 128 allowed the agent to start with buy and sell actions instead of all buys. Actually, reducing the learning rate also allowed the agent to become more dispersed with its actions