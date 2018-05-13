
from util import *
import numpy as np
from opt import *
from model import *

np.random.seed(10)

# Setup
batch_size = 1
input_size = 15
layer_size = 30
output_size = 1
num_iters = 100


x = np.zeros((num_iters, batch_size, input_size))
for ni in xrange(0, num_iters, 5):
    x[ni, :, (ni/7) % input_size] = 1.0

y = np.zeros((num_iters, batch_size, output_size))
y[(25, 50, 75), :, 0] = 1.0
y = smooth_batch_matrix(y)

gD = 1.0
act = lambda x: np.maximum(x, 0.0)
dt = 0.1
gSom = 0.8
Ee = 4.66
Ei = -1.0 / 3.0

exc_params = dict(
    dt = dt,
    gL = 0.1,
    gB = 1.0,
    gA = 0.8,
    weight_factor=0.1,
    act = act,
)
inh_params = dict(
    dt = dt,
    gL = 0.1,
    gD = 1.0,
    weight_factor=0.1,
    act = act,
)
out_params = dict(
    dt=dt,
    gL=0.1,
    gB=1.0,
    weight_factor=0.1,
    act=act,
)

net = [
    [
        ExcLayer(num_iters, batch_size, input_size, layer_size, output_size, **exc_params),
        InhLayer(num_iters, batch_size, layer_size, output_size, **inh_params)
    ],
]

out = OutputLayer(num_iters, batch_size, layer_size, output_size, **out_params)
net[0][0].Wa = out.Wb.T.copy()

def net_run(t, x, y, test=False):
    for li, (l_exc, l_inh) in enumerate(net):
        l_exc_next = net[li + 1][0] if li < len(net) - 2 else out

        Aff = net[li - 1][0].A if li > 0 else x
        Afb = l_exc_next.A
        Ainh = l_inh.A

        gExc = gSom * (l_exc_next.U - Ei)/(Ee - Ei)
        gInh = - gSom * (l_exc_next.U - Ee)/(Ee - Ei)
        Ik = gExc * (Ee - l_inh.U) + gInh * (Ei - l_inh.U)

        l_exc.run(t, Aff, Afb, Ainh)
        l_inh.run(t, l_exc.A, Ik)

    if test:
        Iteach = 0.0
    else:
        Iteach = y * (Ee - out.U) + 2.0 * (Ei - out.U)
    out.run(t, net[-1][0].A, Iteach)


def net_reset(net, out):
    for l_exc, l_inh in net:
        l_exc.reset_state()
        l_inh.reset_state()
    out.reset_state()


opt = SGDOpt((0.001, 0.001, 0.0, 0.01, 0.01))
# opt = AdamOpt((0.001, 0.0001, 0.01, 0.0), 0.99)
opt.init(net[0][0].Wb, net[0][0].Winh, net[0][0].Wa, net[0][1].W, out.Wb)


epochs = 10000
for epoch in xrange(epochs):
    for t in xrange(num_iters): net_run(t, x[t], y[t])

    opt.update(-net[0][0].dWb, -net[0][0].dWinh, -net[0][0].dWa, -net[0][1].dW, -out.dWb)
    net[0][0].Wa = out.Wb.T.copy()

    net_reset(net, out)

    if epoch % 100 == 0:
        train_stat = (
            np.mean(net[0][0].Eh[:, 0]),
            np.mean(net[0][0].Eh[:, 1]),
            np.mean(net[0][0].Eh[:, 2]),
            np.mean(net[0][1].Eh[:, 0]),
            np.mean(out.Eh[:, 0]),
        )
        for t in xrange(num_iters): net_run(t, x[t], y[t], test=True)
        global_loss = np.linalg.norm(out.Ah - y)

        # noinspection PyStringFormat
        print "Epoch {}, {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}, Loss {:.4f}".format(
            epoch, *(train_stat + (global_loss,))
        )

        net_reset(net, out)


for t in xrange(num_iters): net_run(t, x[t], y[t], test=True)
