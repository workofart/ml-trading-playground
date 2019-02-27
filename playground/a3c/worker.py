import numpy as np
from playground.env.trading_env import TradingEnv
from playground.a3c.actorcritic import ActorCriticNet
from playground.utilities.utils import plot_trades, log_scalars, get_latest_run_count, log_histogram

DATA_LENGTH = 600
INIT_CASH = 100
MAX_GLOBAL_EP = 3000
UPDATE_GLOBAL_ITER = 10
TEST_EVERY_EP = 100
SAVE_NETWORK = 100
GLOBAL_EP = 0
GAMMA = 0.9


def test(env, actor, ep = 0):
    state = env.reset() # To start the process

    prices = []
    actions = []
    for i in range(DATA_LENGTH):
        prices.append(state[0][2])
        action = actor.choose_action(state)
        actions.append(action)
        state, reward, done, _ = env.step(action)
    plot_trades(ep, prices, actions, env.permitted_trades)


class Worker(object):
    def __init__(self, name, globalActorCritc, SESS, summary_writer):
        self.env = TradingEnv(DATA_LENGTH, INIT_CASH)
        self.name = name
        self.AC = ActorCriticNet(name, SESS, globalActorCritc)
        self.session = SESS
        self.summary_writer = summary_writer

    def work(self, COORD, saver):
        global GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            print(self.name,"Ep:", GLOBAL_EP)
            state = self.env.reset()
            reward_list = []
            done = False
            while done is False:
                action = self.AC.choose_action(state)
                next_state, reward, done, info = self.env.step(action)
                reward_list.append(reward)
                buffer_s.append(state)
                buffer_a.append(action)
                buffer_r.append(reward)

                if total_step % UPDATE_GLOBAL_ITER == 0:   # update global and assign to local net
                    v_s_ = self.session.run(self.AC.v, {self.AC.state: next_state})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.state: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)

                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                state = next_state
                total_step += 1
                ### One episode End
            
            GLOBAL_EP += 1
            v_s_ = 0   # terminal
            log_scalars(self.summary_writer, 'profit', info['marketValue'] - INIT_CASH, GLOBAL_EP)
            log_scalars(self.summary_writer, 'avg_reward', np.mean(reward_list), GLOBAL_EP)
            log_histogram(self.summary_writer, 'reward_dist', reward_list, GLOBAL_EP)
            log_scalars(self.summary_writer, 'drawdown', np.mean(np.sum(np.array(reward_list) < INIT_CASH, axis=0)), GLOBAL_EP)

            # TODO: Might not be able to naively create test trades, because
            # the workers are working asynchronously, so the trades
            # might not make sense
            if GLOBAL_EP % TEST_EVERY_EP == 0 and GLOBAL_EP > 0:
                test(self.env, self.AC, GLOBAL_EP)
            
            # save network frequently
            if GLOBAL_EP % SAVE_NETWORK == 0 and GLOBAL_EP > 0:
                saver.save(self.session, 'logs/' + str(get_latest_run_count()-1) + '/saved_networks/' + 'network' + '-a3c', global_step = GLOBAL_EP)
