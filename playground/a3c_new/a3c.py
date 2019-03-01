import os, multiprocessing, threading
from time import sleep
import tensorflow as tf

from playground.env.trading_env import TradingEnv
from playground.a3c_new.ac_network import AC_Network
from playground.a3c_new.worker import Worker

max_episode_length = 300
gamma = .99 # discount rate for advantage estimation and reward discounting
load_model = False
model_path = './model'

DATA_LENGTH = 300
INIT_CASH = 100
dummy_env = TradingEnv(DATA_LENGTH, INIT_CASH)
s_size = dummy_env.observation_space.shape[1]
a_size = len(dummy_env.action_space)

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)
    
# #Create a directory to save episode playback gifs to
# if not os.path.exists('./frames'):
#     os.makedirs('./frames')

with tf.device("/cpu:0"): 
    global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-6)
    master_network = AC_Network(s_size,a_size,'global',None) # Generate global network
    num_workers = multiprocessing.cpu_count() # Set workers to number of available CPU threads
    workers = []
    # Create worker classes
    for i in range(num_workers):
        workers.append(Worker(TradingEnv(DATA_LENGTH, INIT_CASH),i,s_size,a_size,trainer,model_path,global_episodes))
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
        
    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_length,gamma,sess,coord,saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)