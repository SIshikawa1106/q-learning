# shota ishikawa

import chainer
import chainer.functions as F
import copy
import numpy as np
import os

from chainer import cuda
from logging import getLogger
from chainerrl.misc.copy_param import synchronize_parameters
from chainerrl.misc.makedirs import makedirs
from chainer import serializers

DataLabels = ['current_state',
                 'next_state',
                 'action',
                 'reward',
                 'is_state_terminal']

InputSubLabels = ['s',
                  'r',
                  'last',
                  'a'
                  ]

def compute_value_loss(y, t):
    #print("compute value loss")
    y = F.reshape(y, (-1, 1))
    t = F.reshape(t, (-1, 1))

    loss_sum = F.sum(F.huber_loss(y, t, delta=1.0))
    loss = loss_sum / y.shape[0]
    return loss

class DQL():
    """
    example
    labelTable = {'next_state':('xs1', 's'),
                  'current_state':('s0', 's'),
                  'action':('x', 'a'),
                  'reward':('reward', 'r'),
                  'is_state_terminal':('isLast', 'last')}
    """
    def __init__(self, q_function,optimizer,gamma, labelTable,gpu=None,target_update_interval=10**4,
                 logger=getLogger(__name__), savingDir=__name__):
        self.model = q_function
        self.q_function = q_function  # For backward compatibility

        if gpu is not None and gpu >= 0:
            cuda.get_device(gpu).use()
            self.model.to_gpu(device=gpu)

        self.labelTable = labelTable
        self.optimizer = optimizer
        self.gamma = gamma
        self.gpu = gpu
        self.target_update_interval = target_update_interval
        self.target_model = None
        self.target_update_method = 'hard'
        self.logger = logger
        self.t = 0
        self.saving_dir = savingDir
        self.sync_target_network()
        self.target_q_function = self.target_model
        self.average_loss = 0
        self.average_loss_decay = 0.999
        self.average_q = 0
        self.average_q_decay = 0.999
        self.soft_update_tau = 1e-2

        self.saved_attributes = ('model', 'target_model', 'optimizer')

    def makeDataForTraining(self, data):

        outputData = {}

        global DataLabels
        labels = DataLabels

        #print("makeData")
        #print(self.labelTable)
        #print(data)
        for label in labels:
            #print('label={}'.format(label))
            outputData[label] = {}
            assert label in self.labelTable
            for tmp in self.labelTable[label]:
                assert len(data) > tmp[0]
                #print(tmp)
                outputData[label][tmp[1]] = data[tmp[0]]

        return outputData

    def _compute_target_values(self, data, gamma):
        batch_next_state = data['next_state']

        assert batch_next_state is not None
        assert self.target_model is not None
        assert self.target_q_function is not None

        with chainer.using_config('train', False):
            next_q_values = self.q_function(batch_next_state)

        target_next_q_values = self.target_q_function(batch_next_state)

        actions = chainer.Variable(next_q_values.data.argmax(axis=1).astype(np.int32))
        target_next_q_max = F.select_item(target_next_q_values, actions)

        #print("target_q_max={}".format(target_next_q_max))

        batch_rewards = data['reward']['r'].reshape(target_next_q_max.shape[0])
        batch_terminal = data['is_state_terminal']['last'].reshape(target_next_q_max.shape[0])

        return batch_rewards + gamma * (1.0 - batch_terminal) * target_next_q_max

    def _compute_y_and_t(self, data):
        batch_size = data['reward']['r'].shape[0]
        batch_current_state = data['current_state']
        batch_actions = data['action']['a'].astype(np.int32)

        qout = self.model(batch_current_state)

        if qout.shape[1]==1:
            q_batch = F.reshape(qout, (batch_size, 1))
        else:
            #print('qout={},\naction={}'.format(qout, batch_actions))
            q_batch = F.reshape(F.select_item(qout,batch_actions.reshape(batch_size)), (batch_size,1))
            #print(q_batch)

        self.average_q *= self.average_q_decay
        self.average_q += (1.0 - self.average_q_decay)*(F.mean(q_batch)).data

        with chainer.no_backprop_mode():
                batch_q_target = F.reshape(self._compute_target_values(data, self.gamma), (batch_size, 1))

        #print("q_batch={},batch_q_target={}".format(q_batch,batch_q_target))
        return q_batch, batch_q_target

    def _compute_loss(self, data):
        y, t = self._compute_y_and_t(data)
        #print("y={},\nt={}".format(y,t))
        return compute_value_loss(y, t)

    def _update(self, data):
        loss = self._compute_loss(data)

        #print("loss={}".format(loss))
        self.average_loss *= self.average_loss_decay
        self.average_loss += (1 - self.average_loss_decay) * float(loss.data)

        #print("average_loss={}".format(self.average_loss))
        self.model.cleargrads()
        loss.backward()
        self.optimizer.update()

    def sync_target_network(self):
        """Synchronize target network with current network."""
        if self.target_model is None:
            print("target model generation")
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

        self._update(trainData)

        self.t += 1

    def save(self, dirname):
        """Save internal states."""
        if os.path.isdir(dirname)==False:
            os.mkdir(dirname)

        serializers.save_npz('{}.npz'.format(self.saved_attributes[0]), self.model)
        serializers.save_npz('{}.npz'.format(self.saved_attributes[1]), self.target_model)
        serializers.save_npz('{}.npz'.format(self.saved_attributes[2]), self.optimizer)

    def act(self, state):

        qout = self.model(state, test=True)

        return qout.data.argmax(axis=1).astype(np.int32)


    def load(self, dirname):
        """Load internal states."""
        assert os.path.isdir(dirname)==True
        serializers.load_npz('{}.npz'.format(self.saved_attributes[0]), self.model)
        serializers.load_npz('{}.npz'.format(self.saved_attributes[1]), self.target_model)
        serializers.load_npz('{}.npz'.format(self.saved_attributes[2]), self.optimizer)

        if self.gpu is not None and self.gpu >= 0:
            cuda.get_device(self.gpu).use()
            self.model.to_gpu(device=self.gpu)