import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import gym
import os
import shutil
import matplotlib.pyplot as plt

from playground.utilities.utils import get_latest_run_count
from playground.a3c.worker import Worker
from playground.a3c.actorcritic import ActorCriticNet

N_WORKERS = multiprocessing.cpu_count()

GLOBAL_EP = 0
DATA_LENGTH = 600
INIT_CASH = 100

GLOBAL_NET_SCOPE = 'Global_Net'

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--device', required=True, help='either "CPU" or "GPU" for training')
    args = vars(ap.parse_args())
    SESS = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

    with tf.device("/{}:0".format(args['device'])):
        
        GLOBAL_ACTORCRITIC = ActorCriticNet(GLOBAL_NET_SCOPE, SESS)  # we only need its params

        COORD = tf.train.Coordinator()
        
        # Setup Tensorboard
        summary_writer = tf.summary.FileWriter('logs/' + str(get_latest_run_count()))
        summary_writer.add_graph(SESS.graph)

        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(i_name, GLOBAL_ACTORCRITIC, SESS, summary_writer))

        SESS.run(tf.global_variables_initializer())
        # loading networks
        saver = tf.train.Saver()

        worker_threads = []
        for worker in workers:
            # checkpoint = tf.train.get_checkpoint_state('logs/' + str(get_latest_run_count()-2) + '/saved_networks_' + worker.name)
            # if checkpoint and checkpoint.model_checkpoint_path:
            #     saver.restore(SESS, checkpoint.model_checkpoint_path)
            #     print("Successfully loaded:", checkpoint.model_checkpoint_path)
            # else:
            #     print("Could not find old network weights")

            job = lambda: worker.work(COORD, saver)
            t = threading.Thread(target=job)
            t.start()
            worker_threads.append(t)
        COORD.join(worker_threads)