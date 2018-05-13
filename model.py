import numpy as np


class ExcLayer(object):
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
        s.gL, s.gB, s.gA = gL, gB, gA

        s.act = act
        s.dt = dt
        s.Vrest = Vrest

        s.Wb = weight_factor*np.random.random((input_size, layer_size)) - weight_factor/2.0
        s.Wa = weight_factor*np.random.random((output_size, layer_size)) - weight_factor/2.0
        s.Winh = weight_factor * np.random.random((output_size, layer_size)) - weight_factor / 2.0

        s.reset_state()

        s.Uh = np.zeros((num_iters, batch_size, layer_size))
        s.Ah = np.zeros((num_iters, batch_size, layer_size))

        s.Vah = np.zeros((num_iters, batch_size, layer_size))
        s.Vbh = np.zeros((num_iters, batch_size, layer_size))

        s.Eh = np.zeros((num_iters, 3))

    def reset_state(s):
        s.U = np.zeros((s.batch_size, s.layer_size))
        s.A = np.zeros((s.batch_size, s.layer_size))
        s.Va = np.zeros((s.batch_size, s.layer_size))
        s.Vb = np.zeros((s.batch_size, s.layer_size))

        s.dWb = np.zeros(s.Wb.shape)
        s.dWa = np.zeros(s.Wa.shape)
        s.dWinh = np.zeros(s.Winh.shape)

    def run(s, t, Aff, Afb, Ainh):
        s.Vb[:] = np.dot(Aff, s.Wb)

        Vfb = np.dot(Afb, s.Wa)
        s.Va[:] = Vfb + np.dot(Ainh, s.Winh)

        dU = - s.gL * s.U + s.gB * (s.Vb - s.U) + s.gA * (s.Va - s.U)

        s.U += s.dt * dU
        s.A[:] = s.act(s.U)

        eb = s.A - s.act(s.Vb * s.gB / (s.gL + s.gB + s.gA))
        ea_inh = s.Vrest - s.Va
        ea_fb = s.A - s.act(Vfb) # * s.gA / (s.gL + s.gB + s.gA))

        s.dWb += np.dot(Aff.T, eb)
        s.dWinh += np.dot(Ainh.T, ea_inh)
        s.dWa += np.dot(Afb.T, ea_fb)

        s.Vbh[t] = s.Vb.copy()
        s.Vah[t] = s.Va.copy()
        s.Uh[t] = s.U.copy()
        s.Ah[t] = s.A.copy()
        s.Eh[t, :] = (np.linalg.norm(eb), np.linalg.norm(ea_inh), np.linalg.norm(ea_fb))

class InhLayer(object):
    def __init__(
        s,
        num_iters,
        batch_size,
        input_size,
        layer_size,
        act,
        gL,
        gD,
        dt,
        weight_factor
    ):
        s.batch_size, s.input_size, s.layer_size = batch_size, input_size, layer_size
        s.gL, s.gD = gL, gD

        s.act = act
        s.dt = dt

        s.W = weight_factor*np.random.random((input_size, layer_size)) - weight_factor/2.0

        s.reset_state()

        s.Uh = np.zeros((num_iters, batch_size, layer_size))
        s.Ah = np.zeros((num_iters, batch_size, layer_size))

        s.Vh = np.zeros((num_iters, batch_size, layer_size))

        s.Eh = np.zeros((num_iters, 1))

    def reset_state(s):
        s.U = np.zeros((s.batch_size, s.layer_size))
        s.A = np.zeros((s.batch_size, s.layer_size))
        s.V = np.zeros((s.batch_size, s.layer_size))

        s.dW = np.zeros(s.W.shape)

    def run(s, t, Aexc, Ik):
        s.V[:] = np.dot(Aexc, s.W)

        dU = - s.gL * s.U + s.gD * (s.V - s.U) + Ik

        s.U += s.dt * dU

        s.A[:] = s.act(s.U)

        e = s.A - s.act(s.V * s.gD / (s.gL + s.gD))

        s.dW += np.dot(Aexc.T, e)

        s.Vh[t] = s.V.copy()
        s.Uh[t] = s.U.copy()
        s.Ah[t] = s.A.copy()
        s.Eh[t] = np.linalg.norm(e)


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
        s.dWb += np.dot(Aff.T, eb)

        s.Vbh[t] = s.Vb.copy()
        s.Uh[t] = s.U.copy()
        s.Ah[t] = s.A.copy()
        s.Eh[t] = np.linalg.norm(eb)