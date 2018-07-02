import os
import chainer
from nnabla.utils.data_iterator import data_iterator_csv_dataset, data_iterator_cache
import nnabla

from Model import PieceModel
from agent import DQL

def loadData(batch_size):

    cache_dir = "./cache"
    if os.path.isdir(cache_dir)==False:
        os.mkdir(cache_dir)
        dataset = data_iterator_csv_dataset("../AIStudy.BoardGame/experience.csv", batch_size, shuffle=True, normalize=True, cache_dir=cache_dir)
    else:
        dataset = data_iterator_cache(cache_dir, batch_size, shuffle=True, normalize=True)

    variables = dataset.variables
    print(variables)
    """

    s0Index = variables.index('s0')
    print(("index(s0)={}").format(s0Index))
    for n in range(1000):
        data = dataset.next()
        if n==0:
            print(("shape(s0)={}").format(data[s0Index].shape))
        print(("epoch={},position={},size={}, data_size={}").format(dataset.epoch, dataset.position,dataset.size, len(data[0])))
    """

    return dataset


if __name__ == "__main__":
    print("DATA LOAD")
    dataset = loadData(32)
    labelTable = {'next_state':[(5, 's')],
                  'current_state':[(3, 's')],
                  'action':[(4, 'a')],
                  'reward':[(2, 'r')],
                  'is_state_terminal':[(1, 'last')]}

    print("model generation")
    model = PieceModel((dataset.batch_size, 1, 8, 5), 5)
    print("optimaizer setup")
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    print("agent generation")
    agent = DQL(model, optimizer, 0.9, labelTable=labelTable)

    print("training start")
    for i in range(10000):
        data = list(dataset.next())

        #print(data)
        data[4] = data[4] - 1

        agent.train_using_data(data)

        print("i={:04d},loss={},q={}".format(i, agent.average_loss, agent.average_q))


