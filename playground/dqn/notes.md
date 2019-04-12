## Tips

- Make sure the next_state is properly set to prevent the Q-values from being all the same.

- need to normalize the input data, because volume and price are not on the same scale, which makes training hard to converge to global min

- never leave the epsilon the same across episodes, as it will stay at the min value and cause the agent to fully trust the actions output by the q-network

- no need to necessarily train at every time step, maybe once every N timesteps

- the last layer of the neural network should be of "action_dim" not "1"

- I've been seeing, after 3000 episodes of training, the reward function following a sine-way like function and I suspect it has something to do with epsilon. I think it makes sense to update epsilon after every episode, as opposed to after every timestep, because timestep 0 and timestep n doesn't really mean anything for exploration and exploitation, but episode 0 and episode 1000 means a lot, the latter will more likely have a better trained model and thus require less for exploration and more for exploitation.

- Don't nest copying the target network into the training code block because if the training is not done after every timestep, but the copying of the target network is based on timesteps, so there is a possibility that the copying of the network will not get triggered frequently as defined. 

- Using the "tanh" activation function eliminated the use of q-value loss clipping, the q-values became non-linear and not always in a fixed trend (up or down)

## FAQ

#### Difference between `tf.Variable()` and `tf.get_variable()`
I can find two main differences between one and the other:
First is that tf.Variable will always create a new variable, whether `tf.get_variable` gets from the graph an existing variable with those parameters, and if it does not exists, it creates a new one.
`tf.Variable` requires that an initial value be specified.
It is important to clarify that the function tf.get_variable prefixes the name with the current variable scope to perform reuse checks.

Reference: https://stackoverflow.com/questions/37098546/difference-between-variable-and-get-variable-in-tensorflow


#### Tensorboard Summaries raising placeholder type mismatch error
The merged summary merged = tf.merge_all_summaries() is taking into account previous summaries (coming from I don't know where), which depend on placeholders not initialized.

#### Pyplot drawing graphs without blocking the main thread

```
plt.axis([-50,50,0,10000])
plt.ion()
plt.show()

x = np.arange(-50, 51)
for pow in range(1,5):   # plot x^1, x^2, ..., x^4
    y = [Xi**pow for Xi in x]
    plt.plot(x, y)
    plt.draw()
    plt.pause(0.001)
    input("Press [enter] to continue.")
```


#### Problem with Q values of multiple actions being close together
[Reference](https://ai.stackexchange.com/questions/8605/ensure-convergence-of-ddqn-if-true-q-values-are-very-close)

> Let Q∗(s,a) denote the "true" Q-value for a state-action pair (s,a), i.e. the values that we're hoping to learn to approximate using a neural network that outputs Q(s,a) values. The problem you describe is basically that you have situations where Q∗(s,a1)=Q∗(s,a2)+ϵ for some very small value ϵ, where a1≠a2.

> An important thing to note is that the "true" Q∗ values are directly determined by the reward function, which essentially encodes your objective. If your reward function is constructed in such a way that it results in the situation described above, that means the reward function is essentially saying: "it doesn't really matter much whether we pick action a1 or a2, there is some difference in returns but it's a tiny difference, we hardly care." **The objective implied by the reward function in some sense contradicts what you describe in the question that you want**. You say that you care (quite a lot) about a tiny difference in Q∗-values, but normally the "extent to which we care" is proportional to the difference in Q∗-values. So, if there is only a tiny difference in Q∗-values, we should really only care a tiny bit.

> Now, with tabular RL algorithms (like standard Q-learning), we do expect to be able to eventually converge to the truly correct solutions, we expect to eventually be capable of correctly ranking different actions even when the differences in Q∗-values are small. When you enter function approximation (especially neural networks), this story changes. It's pretty much in the name already; function **approximation**, we cannot guarantee being able to do better than just approximating. So, 100% **ensuring** that you'll be able to learn the correct rankings is **not** going to be feasible. There may be some things that can help though:

> 1. If you are "in control" of your environment and its reward function (meaning, you implemented it yourself and/or can modify it if you like), modifying the reward function would probably be the single most effective solution to your problem. As I described above, a reward function that results in tiny differences in Q∗-values essentially encodes that you only care about those difference to a tiny (negligible) extent. If that's not the case, if you actually do care about those tiny differences... well, you can try changing the reward function to accentuate those tiny differences. ~~For example, you could try simply multiplying all rewards by 100, then all differences in returns are also multipled by 100. Note that this does mean you get larger values everywhere (larger losses, larger gradients, etc.), so it would require changing hyperparameters (like learning rate) accordingly, and could destabilize things. But at least it will emphasize those small differences in Q∗-values.~~ (probably won't work, because you also scale the variance of the reward by the same factor squared, meaning that there is no improvement in the learning agent's ability to discern between the two similar features based on the samples it has seen)
> 2. Increasing batch size. You already mentioned this yourself, but I'll also explain why it can help. A major issue in your problem is that you described the environment to be highly stochastic. This means you get a large amount of **variance** in your observed returns, and therefore also in your updates. If the true differences in Q∗-values are very small, these differences will get dominated and become completely invisible due to high variance in observations. You can smooth out this variance by increasing the batch size.
> 3. Decreasing learning rate. You also mentioned this in the question, and that it didn't really help. I suppose a high learning rate early on can help to learn more quickly, but decreasing it later on can help to take more "careful", small update steps that don't take "jumps" of a magnitude greater than the true difference in Q∗-values.
> 4. Distributional Reinforcement Learning: algorithms like Q-learning, DQN, Double DQN, etc., they all learn what we can call values, or scalars, or point estimates. Given a state-action pair (s,a), such a Q-value represents the expected returns obtained by executing a in s and continuing with a greedy policy afterwards. In stochastic environments, it is possible that the **true expected value Q∗(s,a) never coincides with any conrete observation at the end of an actual trajectory**. For example, suppose that in some state s, an action a has a probability of 0.5 of giving a reward of 1 and instantly terminating afterwards, but also has a probability of 0.5 of giving a reward of 0 and instantly terminating. Then, we have Q∗(s,a)=0.5, but we **never** actually observe a return of 0.5; we always observe returns of either 0 or 1. When comparing the output of the learned Q-function (which should hopefully be around 0.5 in such a situation) to any observed trajectories, we'll almost always have a non-zero error and take update steps that keep bouncing around the optimal solution, rather than converging to the optimal solution. Since you mentioned that you see Q-values fluctuating around, you may have observed this. **Excessive fluctuations around Q-value estimates will likely make it difficult to get a consistent, stable ranking when the optimal Q-values that you are fluctuating around are themselves very close to each other**. There is a class of algorithms (extensions of DQN mostly) referred to as **distributional RL**, which aim to address this issue by learning a full **distribution** of returns we expect to be capable of observing, rather than the single-point estimates that Q-values essentially are. I believe the first major paper on this topic was the following: https://arxiv.org/abs/1707.06887. There have been multiple follow-up papers too, the most recent of which I believe (but may be wrong here) to be this one: https://arxiv.org/abs/1806.06923. In general, I do know Marc Bellemare and Rémi Munos at least have done quite some work on that, so you could search for those names too.

#### Difference between Target Network approach and Double DQN approach
[Reference](https://datascience.stackexchange.com/questions/32246/q-learning-target-network-vs-double-dqn)

Target Network Approach
> 1. Select an item from Memory Bank
> 2. Using Target Network, from St+1 determine the index of the best action At+1 and its Q-value
> 3. Do corrections as usual

Double DQN Approach
> 1. Select an item from Memory Bank
> 2. Using Online Network, from St+1 determine the index of the best action At+1.
> 3. Using Target Network, from St+1 get the Q-value of that action.
> 4. Do corrections as usual using that Q-value

Were we to let Target Network select a best action, it could have very well selected some other index.


#### Difference between Q-Learning and Policy Gradient
The policy implied by Q-Learning is deterministic. This means that Q-Learning can’t learn stochastic policies, which can be useful in some environments. __It also means that we need to create our own exploration strategy since following the policy will not perform any exploration.__ We usually do this with ϵϵ-greedy exploration, which can be quite inefficient.

There is no straightforward way to handle continuous actions in Q-Learning. In policy gradient, handling continous actions is relatively easy.

As its name implies, in policy gradient we are following gradients with respect to the policy itself, which means we are constantly improving the policy. By contrast, in Q-Learning we are improving our estimates of the values of different actions, which only implicitely improves the policy. You would think that improving the policy directly would be more efficient, and indeed it very often is.

Reference: https://towardsdatascience.com/an-intuitive-explanation-of-policy-gradient-part-1-reinforce-aa4392cbfd3c

#### Gradient Clipping
```
optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE)
grads = optimizer.compute_gradients(self.loss)
capped_grads = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads]
self.train_opt = optimizer.apply_gradients(capped_grads)
```

#### Clipping Q-Values
The problem with the current approach is that the Q-values are initially sparsed out across the 3 actions, and during training, the q-values all increase which results in nothing learned. This is because the best action will always be the action with the highest initialized q-value. After clipping q-values between -1 and 1, the q-values tend to converge around 0 and 1

#### Dying ReLU Neuron
Reference: https://datascience.stackexchange.com/questions/5706/what-is-the-dying-relu-problem-in-neural-networks

A "dead" ReLU always outputs the same value (zero as it happens, but that is not important) for any input. Probably this is arrived at by learning a large negative bias term for its weights.

In turn, that means that it takes no role in discriminating between inputs. For classification, you could visualise this as a decision plane outside of all possible input data.

Once a ReLU ends up in this state, it is unlikely to recover, because the function gradient at 0 is also 0, so gradient descent learning will not alter the weights. "Leaky" ReLUs with a small positive gradient for negative inputs (y=0.01x when x < 0 say) are one attempt to address this issue and give a chance to recover.

The sigmoid and tanh neurons can suffer from similar problems as their values saturate, but there is always at least a small gradient allowing them to recover in the long term.

## TODO

Find a way to build in the portfolio and cash amount into the states. The tricky part is that for double DQN, you need the next state for the target network to work properly.