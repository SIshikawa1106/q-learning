import chainer
import chainer.links as L
import chainer.functions as F

class PieceModel(chainer.Chain):
    def __init__(self, shape, n_actions, n_hidden_channels=64):
        super(PieceModel, self).__init__(
            conv1=L.Convolution2D(shape[1], n_hidden_channels, ksize=3),
            conv2=L.Convolution2D(n_hidden_channels, n_hidden_channels, ksize=3),
            l1=L.Linear((shape[2]-4) * (shape[3]-4) * n_hidden_channels, n_hidden_channels),
            l2=L.Linear(n_hidden_channels, n_actions)
        )

    def __call__(self, x, test=False):

        img = x['s']
        print("image shape={}".format(img.shape))
        print("type={}".format(type(img)))

        s = chainer.Variable(img)
        h = F.leaky_relu(self.conv1(s))
        h = F.leaky_relu(self.conv2(h))
        h = F.leaky_relu(self.l1(h))
        #h = F.sigmoid(self.l2(h))
        h = self.l2(h)
        return h