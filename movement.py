import numpy as np
import sdeint

# a = np.array([[0, -1, 0], [3, 1, 3], [0, 0, 7]])
a = np.array([[0, -1, 0], [1, 1, 1], [0, 0, 1]])
b = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]])


def f(x, t, a=a):
    return -a.dot(x)


def G(x, t, b=b):
    return b


def step_sde(Yn, tn, m=3, f=f, G=G, dt=0.05):
    """
    Parameters
    ----------
    Yn : array (d,)
        Current value

    tn : float
        Current time

    m : int
        Dimension of ...
    """
    d = len(Yn)
    ##tn = tspan[n]
    ##tn1 = tspan[n+1]
    tn = dt * tn
    tn1 = tn + dt
    ##h = tn1 - tn
    h = dt
    sqrth = np.sqrt(h)
    ##Yn = y[n] # shape (d,)
    dW = sdeint.deltaW(1, m, h)
    _, I = sdeint.Ikpw(dW, h)
    Ik = dW[0, :]  # shape (m,)
    Iij = I[0, :, :]  # shape (m, m)
    fnh = f(Yn, tn) * h  # shape (d,)
    Gn = G(Yn, tn)
    sum1 = np.dot(Gn, Iij) / sqrth  # shape (d, m)
    H20 = Yn + fnh  # shape (d,)
    H20b = np.reshape(H20, (d, 1))
    H2 = H20b + sum1  # shape (d, m)
    H30 = Yn
    H3 = H20b - sum1
    fn1h = f(H20, tn1) * h
    Yn1 = Yn + 0.5 * (fnh + fn1h) + np.dot(Gn, Ik)
    for k in range(0, m):
        Yn1 += 0.5 * sqrth * (G(H2[:, k], tn1)[:, k] - G(H3[:, k], tn1)[:, k])
    # y[n+1] = Yn1
    return Yn1
