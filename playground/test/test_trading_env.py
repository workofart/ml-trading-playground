import numpy as np
from index import TradingEnv

EPOCHS = 10

env = TradingEnv()

for ep in range(EPOCHS):
    done = False
    state = env.reset()
    while done is False:
        action = np.random.randint(0, 3) # Random sampling
        state, reward, done, _ = env.step(action)
    
    assert(env.time_step == len(env.observation_space)) # check max time_step at end of epoch
    assert(state.shape[1] == env.observation_space.shape[1] + 2) # state + cash + portfolio = 6 dims
    assert(state.shape[0] == 1) # one observation per state/step
    
    avg_reward = env.episode_total_reward / env.time_step
    print('EP: %d | Average reward: %.2f' %(ep, avg_reward))

