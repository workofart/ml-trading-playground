import tensorflow as tf
import numpy as np
import scipy.signal

from playground.a3c_new.ac_network import AC_Network
from playground.utilities.utils import update_target_graph, plot_trades
from playground.dqn.experience_buffer import Experience_Buffer

DATA_LENGTH = 300

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def test(env, sess, actor, ep = 0, name=''):
    state = env.reset() # To start the process

    prices = []
    actions = []
    for i in range(DATA_LENGTH):
        prices.append(state[0][2])
        #Take an action using probabilities from policy network output.
        a_dist,v = sess.run([actor.policy,actor.value], 
            feed_dict={actor.state:state})
        a = np.random.choice(a_dist[0],p=a_dist[0])
        action = np.argmax(a_dist == a)
        actions.append(action)
        state, reward, done, _ = env.step(action)
    plot_trades(ep, prices, actions, env.permitted_trades, name)

EXPERIENCE_BUFFER_SIZE = 100

class Worker():
    def __init__(self,env,name,s_size,a_size,trainer,model_path,global_episodes):
        self.name = "worker_" + str(name)
        self.number = name        
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_"+str(self.number))

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(s_size,a_size,self.name,trainer)
        self.update_local_ops = update_target_graph('global',self.name)        
        
        # TODO: setup the env


        self.actions = np.identity(a_size,dtype=bool).tolist()
        
        self.env = env
        
    # Assumes the rollout follows this format
    # [state, actions, rewards, next_state, values]
    def train(self,rollout,sess,gamma,bootstrap_value):
        rollout = np.array(rollout.buffer)
        states = [i.astype(np.float32) for i in rollout[:,0]]
        actions = rollout[:,1].astype(np.int32)
        rewards = rollout[:,2].astype(np.float32)
        next_states = [i.astype(np.float32) for i in rollout[:,3]]
        values = rollout[:,5].astype(np.float32)
        
        # Here we take the rewards and values from the rollout, and use them to 
        # generate the advantage and discounted returns. 
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1].astype(np.float32)
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages,gamma).astype(np.float32)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_AC.target_v:discounted_rewards,
            self.local_AC.state:np.vstack(states),
            self.local_AC.actions:actions,
            self.local_AC.advantages:advantages}
        v_l,p_l,e_l,g_n,v_n,_, gradients= sess.run([self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.entropy,
            self.local_AC.grad_norms,
            self.local_AC.var_norms,
            # self.local_AC.state_out,
            self.local_AC.apply_grads,
            self.local_AC.gradients],
            feed_dict=feed_dict)
        print(min([i.min() for i in gradients]))
        print(max([i.max() for i in gradients]))
        return v_l / len(rollout),p_l / len(rollout),e_l / len(rollout), g_n,v_n
        
    def work(self,max_episode_length,gamma,sess,coord,saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print ("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():                 
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                experience_buffer = Experience_Buffer(EXPERIENCE_BUFFER_SIZE)
                episode_values = []
                episode_reward = 0
                episode_step_count = 0
                d = False
                
                s = self.env.reset()
                while d == False:
                    #Take an action using probabilities from policy network output.
                    a_dist,v = sess.run([self.local_AC.policy,self.local_AC.value], 
                        feed_dict={self.local_AC.state:s})
                    a = np.random.choice(a_dist[0],p=a_dist[0])
                    a = np.argmax(a_dist == a)
                    
                    s, r, d, _ = self.env.step(a)
                    if d == False:
                        s1 = self.env._get_obs()
                    else:
                        s1 = s
                        
                    experience_buffer.add([s[0],a,r,s1[0],d,v[0,0]])
                    episode_values.append(v[0,0])

                    episode_reward += r
                    s = s1                    
                    total_steps += 1
                    episode_step_count += 1
                    
                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if experience_buffer.size() == EXPERIENCE_BUFFER_SIZE and d != True and episode_step_count != max_episode_length - 1:
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        v1 = sess.run(self.local_AC.value, 
                            feed_dict={self.local_AC.state:s})[0,0]
                        v_l,p_l,e_l,g_n,v_n = self.train(experience_buffer,sess,gamma,v1)
                        experience_buffer.clear()
                        sess.run(self.update_local_ops)
                    if d == True:
                        break
                                            
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))
                
                # Update the network using the episode buffer at the end of the episode.
                if experience_buffer.size() != 0:
                    v_l,p_l,e_l,g_n,v_n = self.train(experience_buffer,sess,gamma,0.0)
                                
                    
                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    if episode_count % 250 == 0 and self.name == 'worker_0':
                        saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
                        print ("Saved Model")

                    # test(self.env, sess, self.local_AC, episode_count, self.name)
                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1