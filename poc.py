
from util import *
import numpy as np
from opt import *
from model_sc_snap import *


np.random.seed(10)

# Setup
batch_size = 1
input_size = 15
layer_size = 30
output_size = 1
num_iters = 100


act = lambda x: np.maximum(x, 0.0)


x = np.zeros((num_iters, batch_size, input_size))
for ni in xrange(0, num_iters, 5):
    x[ni, :, (ni/7) % input_size] = 1.0

y = np.zeros((num_iters, batch_size, output_size))
y[(25, 50, 75), :, 0] = 1.0
y = smooth_batch_matrix(y)


# w0 = 0.1*np.random.randn(input_size, layer_size)
# w1 = 0.1*np.random.randn(layer_size, output_size)
#
# x = np.random.randn(batch_size, input_size)
# y = act(np.dot(act(np.dot(x, w0)), w1))

dt = 0.1

net_params = dict(
    gSom=0.8,
    Ee=4.66,
    Ei=-1.0/3.0,
)

layer_params = dict(
    dt=dt,
    gL=0.1,
    gB=1.0,
    gA=0.8,
    weight_factor=0.1,
    act=act,
)

out_params = dict(
    dt=dt,
    gL=0.1,
    gB=1.0,
    weight_factor=0.1,
    act=act,
)

net = Net(
    layers=[
        Layer(num_iters, batch_size, input_size, layer_size, output_size, **layer_params),
    ],
    output_layer=OutputLayer(num_iters, batch_size, layer_size, output_size, **out_params),
    **net_params
)

# net.symmetric_feedback()

opt = SGDOpt((0.01, 0.01, 0.01))
# opt = AdamOpt((0.0001, 0.0001, 0.0, 0.0001, 0.0001), 0.99)

L0 = net.layers[0]
L1 = net.output_layer

opt.init(L0.Wb, L0.Wa, L1.Wb)


epochs = 1
for epoch in xrange(epochs):
    for t in xrange(num_iters): net.run(t, x[t], y[t], test=False)

    opt.update(-L0.dWb, -L0.dWa, -L1.dWb)
    # net.symmetric_feedback()

    net.reset_state()

    if epoch % 10 == 0:
        train_stat = (
            np.mean(L0.Eh[:, 0]),
            np.mean(L0.Eh[:, 1]),
            np.mean(L1.Eh[:, 0]),
        )

        for t in xrange(num_iters): net.run(t, x[t], y[t], test=True)
        global_loss = np.linalg.norm(L1.Ah - y)

        # noinspection PyStringFormat
        print "Epoch {}, b {:.4f} a {:.4f} b {:.4f}, Loss {:.4f}".format(
            epoch, *(train_stat + (global_loss,))
        )

        net.reset_state()


for t in xrange(num_iters): net.run(t, x[t], y[t], test=True)
