# shota ishikawa

import chainer
from chainer import cuda
import chainer.functions as F
import copy
from logging import getLogger
from chainerrl.misc.copy_param import synchronize_parameters
import numpy as np
import os

def compute_value_loss(self, y, t):

    y = F.reshape(y, (-1, 1))
    t = F.reshape(t, (-1, 1))

    loss_sum = F.sum(F.huber_loss(y, t, delta=1.0))
    loss = loss_sum / y.shape[0]

    return loss

class DQL():
    def __init__(self, q_function,optimizer,gamma,gpu=None,target_update_interval=10**4,
                 logger=getLogger(__name__), savingDir=__name__):
        self.model = q_function
        self.q_function = q_function  # For backward compatibility

        if gpu is not None and gpu >= 0:
            cuda.get_device(gpu).use()
            self.model.to_gpu(device=gpu)

        self.optimizer = optimizer
        self.gamma = gamma
        self.gpu = gpu
        self.target_update_interval = target_update_interval
        self.target_model = None
        self.target_update_method = 'hard'
        self.logger = logger
        self.t = 0
        self.saving_dir = savingDir
        self.target_q_function = self.target_model
        self.average_loss = 0
        self.average_loss_decay = 0.999
        self.average_q = 0
        self.average_q_decay = 0.999

    def makeDataForTraining(self, data):

        outputData = {}

        outputData['current_state'] = 0
        outputData['next_state'] = 0
        outputData['reward'] = 0
        outputData['is_state_terminal'] = 0

        return outputData

    def _compute_target_values(self, data, gamma):
        batch_next_state = data['next_state']

        with chainer.using_config('train', False):
            next_q_values = self.q_function(batch_next_state)

        target_next_q_values = self.target_q_function(batch_next_state)

        actions = chainer.Variable(next_q_values.data.argmax(axis=1).astype(np.int32))
        target_next_q_max = F.select_item(target_next_q_values, actions)

        batch_rewards = data['reward']
        batch_terminal = data['is_state_terminal']

        return batch_rewards + gamma * (1.0 - batch_terminal) * target_next_q_max

    def _compute_y_and_t(self, data):

        batch_current_state = data['current_state']

        q_batch = self.model(data)

        with chainer.no_backprop_mode():
            batch_q_target = self._compute_target_values(data, self.gamma)

        return q_batch, batch_q_target

    def _compute_loss(self, data):
        y, t = self._compute_y_and_t(data)
        return compute_value_loss(y, t)


    def update(self, data):
        loss = self._compute_loss(data)

        self.average_loss *= self.average_loss_decay
        self.average_loss += (1 - self.average_loss_decay) * float(loss.data)

        self.model.cleargrads()
        loss.backward()
        self.optimizer.update()

    def sync_target_network(self):
        """Synchronize target network with current network."""
        if self.target_model is None:
            self.target_model = copy.deepcopy(self.model)
            call_orig = self.target_model.__call__

            def call_test(self_, x):
                with chainer.using_config('train', False):
                    return call_orig(self_, x)

            self.target_model.__call__ = call_test
        else:
            synchronize_parameters(
                src=self.model,
                dst=self.target_model,
                method=self.target_update_method,
                tau=self.soft_update_tau)

    def train_using_data(self, data):
        if self.t % self.target_update_interval==0:
            self.sync_target_network()
            if os.path.isdir(self.saving_dir)==False:
                os.mkdir(self.saving_dir)

        trainData = self.makeDataForTraining(data)

        self.update(trainData)

        self.t += 1