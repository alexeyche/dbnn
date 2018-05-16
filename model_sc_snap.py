import numpy as np


class Layer(object):
    def __init__(
        s,
        num_iters,
        batch_size,
        input_size,
        layer_size,
        output_size,
        act,
        gL,
        gB,
        gA,
        dt,
        weight_factor,
        Vrest = 0.0
    ):
        s.batch_size, s.input_size, s.layer_size = batch_size, input_size, layer_size
        s.num_iters = num_iters
        s.gL, s.gB, s.gA = gL, gB, gA

        s.act = act
        s.dt = dt
        s.Vrest = Vrest

        s.Wb = weight_factor*np.random.random((input_size, layer_size)) - weight_factor/2.0
        s.Wa = weight_factor*np.random.random((output_size, layer_size)) - weight_factor/2.0

        s.reset_state()

        s.Uh = np.zeros((num_iters, batch_size, layer_size))
        s.Ah = np.zeros((num_iters, batch_size, layer_size))

        s.Vah = np.zeros((num_iters, batch_size, layer_size))
        s.Vbh = np.zeros((num_iters, batch_size, layer_size))

        s.Eh = np.zeros((num_iters, 2))

    def reset_state(s):
        s.U = np.zeros((s.batch_size, s.layer_size))
        s.A = np.zeros((s.batch_size, s.layer_size))
        s.Va = np.zeros((s.batch_size, s.layer_size))
        s.Vb = np.zeros((s.batch_size, s.layer_size))

        s.dWb = np.zeros(s.Wb.shape)
        s.dWa = np.zeros(s.Wa.shape)

    def run(s, t, Aff, Afb):
        s.Vb[:] = np.dot(Aff, s.Wb)
        s.Va[:] = np.dot(Afb, s.Wa)

        dU = - s.gL * s.U + s.gB * (s.Vb - s.U) + s.gA * (s.Va - s.U)

        s.U += s.dt * dU
        s.A[:] = s.act(s.U)

        eb = s.A - s.act(s.Vb * s.gB / (s.gL + s.gB + s.gA))
        ea_fb = s.A - s.act(s.Va * s.gA / (s.gL + s.gB + s.gA))

        s.dWb += np.dot(Aff.T, eb)
        s.dWa += np.dot(Afb.T, ea_fb)

        s.Vbh[t] = s.Vb.copy()
        s.Vah[t] = s.Va.copy()
        s.Uh[t] = s.U.copy()
        s.Ah[t] = s.A.copy()
        s.Eh[t, :] = (np.linalg.norm(eb), np.linalg.norm(ea_fb))



class OutputLayer(object):
    def __init__(
        s,
        num_iters,
        batch_size,
        input_size,
        layer_size,
        act,
        gL,
        gB,
        dt,
        weight_factor
    ):
        s.batch_size, s.input_size, s.layer_size = batch_size, input_size, layer_size
        s.num_iters = num_iters
        s.gL, s.gB = gL, gB

        s.act = act
        s.dt = dt

        s.Wb = weight_factor*np.random.random((input_size, layer_size)) - weight_factor/2.0

        s.reset_state()

        s.Uh = np.zeros((num_iters, batch_size, layer_size))
        s.Ah = np.zeros((num_iters, batch_size, layer_size))

        s.Vbh = np.zeros((num_iters, batch_size, layer_size))

        s.Eh = np.zeros((num_iters, 1))

    def reset_state(s):
        s.U = np.zeros((s.batch_size, s.layer_size))
        s.A = np.zeros((s.batch_size, s.layer_size))
        s.Vb = np.zeros((s.batch_size, s.layer_size))

        s.dWb = np.zeros(s.Wb.shape)

    def run(s, t, Aff, Iteach):
        s.Vb[:] = np.dot(Aff, s.Wb)

        dU = - s.gL * s.U + s.gB * (s.Vb - s.U) + Iteach

        s.U += s.dt * dU
        s.A[:] = s.act(s.U)

        eb = s.A - s.act(s.Vb * s.gB / (s.gL + s.gB))
        # if t == s.num_iters-1:
        s.dWb += np.dot(Aff.T, eb)

        s.Vbh[t] = s.Vb.copy()
        s.Uh[t] = s.U.copy()
        s.Ah[t] = s.A.copy()
        s.Eh[t] = np.linalg.norm(eb)



class Net(object):
    def __init__(self, layers, output_layer, gSom, Ee, Ei):
        self.layers = layers
        self.output_layer = output_layer
        self.gSom = gSom
        self.Ei = Ei
        self.Ee = Ee

    def reset_state(self):
        for l in self.layers:
            l.reset_state()
        self.output_layer.reset_state()

    def run(s, t, x, y, test=False):
        for li, l in enumerate(s.layers):
            l_next = (
                s.layers[li + 1]
                if li < len(s.layers) - 2 else s.output_layer
            )

            Aff = s.layers[li - 1].A if li > 0 else x
            Afb = l_next.A

            l.run(t, Aff, Afb)

        if test:
            Iteach = 0.0
        else:
            Iteach = y * (s.Ee - s.output_layer.U) + 2.0 * (s.Ei - s.output_layer.U)

        s.output_layer.run(t, s.layers[-1].A, Iteach)

    def symmetric_feedback(self):
        for l, Wb in zip(
            self.layers,
            [l.Wb for l in self.layers[1:]] + [self.output_layer.Wb]
        ):
            l.Wa = Wb.T.copy()
