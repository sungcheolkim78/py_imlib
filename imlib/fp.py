import numpy as np
# from scipy.integrate import trapz
import numba
from tqdm import trange

# ==========================================


@numba.jit(nopython=True)
def phi(x, pe):
    '''
    Effective potential energy

    Input:
    x = numpy array of positions, [-0.5, 0.5]
    pe = scalar, peclet number
    Return:
    numpy array of the potential evaluated for each x
    '''
    #if pe > 0: return pe*(x + 0.5)
    return pe*(0.5 - x)


@numba.jit(nopython=True)
def psi_s(x, pe):
    '''
    Stationary probability density

    Input:
    x = numpy array of positions, [-0.5, 0.5]
    pe = scalar, peclet number
    Return:
    numpy array of the probability density evaluated at each x
    '''
    p = np.exp(-phi(x, pe))
    Z = pe / (1 - np.exp(-pe))
    return p * Z

# ==========================================


@numba.jit
def psi_odd(x, xp, t, tp, pe, n, tolerance=1e-6):
    '''
    Compute the eigenfunctions \psi(x)\psi(x') for even integers

    Input:
    x : numpy array of x positions, [-0.5, 0.5]
    xp : scalar [-0.5, 0.5] indicates the initial condition
    t : scalar > tp, time point
    tp : scalar, initial time point
    pe : scalar, peclet number
    n : scalar, number of eigenfunctions to compute, default 100

    Returns:
    numpy array evaluating the eigenfunction
    '''
    psi = 0
    w = 1
    N = 1
    while N <= n:
        kn = N * np.pi
        ln = kn**2 / pe + pe/4.
        norm = 2. / (pe**2 + 4*kn**2)
        tmp = pe * np.cos(kn*x) + 2 * kn * np.sin(kn * x)
        tmp = tmp * (pe * np.cos(kn * xp) + 2 * kn * np.sin(kn * xp))

        psi_add = norm * tmp * np.exp(-ln*(t-tp))
        if abs(psi_add.max()) < tolerance: break

        psi += psi_add
        w += 1
        N = 2 * w - 1
    return psi

# ==========================================


@numba.jit
def psi_even(x, xp, t, tp, pe, n, tolerance=1e-6):
    '''
    Compute the eigenfunctions \psi(x)\psi(x') for even integers

    Input:
    x : numpy array of x positions, [-0.5, 0.5]
    xp : scalar [-0.5, 0.5] indicates the initial condition
    t : scalar > tp, time point
    tp : scalar, initial time point
    pe : scalar, peclet number
    n : scalar, number of eigenfunctions to compute, default 100

    Returns:
    numpy array evaluating the eigenfunction
    '''
    psi = 0
    w = 1
    N = 2
    while N <= n:
        kn = N * np.pi
        ln = kn**2 / pe + pe/4.
        norm = 2. / (pe**2 + 4.*kn**2)
        tmp = 2*kn*np.cos(kn*x) - pe*np.sin(kn*x)
        tmp = tmp * (2*kn*np.cos(kn*xp) - pe*np.sin(kn*xp))

        psi_add = norm * tmp * np.exp(-ln*(t-tp))
        if abs(psi_add.max()) < tolerance: break

        psi += psi_add
        w += 1
        N = 2*w
    return psi


# ==========================================


@numba.jit
def transition_prob(x, xp, t, tp, pe, n=100):
    '''
    Compute the transition probability of a particle located at xp at t=tp which is subject to a constant force and diffusion until time t.

    Input:
    x : numpy array of x positions, [-0.5, 0.5]
    xp : scalar [-0.5, 0.5] indicates the initial condition
    t : scalar > tp, time point
    tp : scalar, initial time point
    pe : scalar, peclet number
    n : scalar, number of eigenfunctions to compute, default 100
    Returns:
    numpy array of length x.size, transtiion probabilities for each x
    '''
    # eigenfunction equations
    const = np.exp(-0.5*phi(x, pe) + 0.5*phi(xp, pe))
    U = const*psi_odd(x, xp, t, tp, pe, n)
    U += const*psi_even(x, xp, t, tp, pe, n)
    # add stationary solution
    U += psi_s(x, pe)
    return U


# ==========================================


@numba.jit
def W(x, wo, t, tp=0, pe=1.0, n=100):
    '''
    Compute the probability density of a particle's location from its time zero distribution.

    Inputs:
    x : numpy array of x positions, [-0.5, 0.5]
    wo : numpy array of length x.size, probability of the partile's location at each x for t = 0
    t : scalar > 0, time point of interest
    tp : scalar, initial time point
    pe : scalar, peclet number
    n : scalar, number of eigenfunctions to compute, default 100
    Returns:
    numpy array of length x.size, probabilities for each x
    '''
    W = np.zeros(x.size)
    U = np.zeros(shape=(x.size, x.size))
    # get the propgator for each initial position
    #for wx in tqdm.tqdm(range(x.size)):
    for wx in range(x.size):
        U[wx, :] = transition_prob(x, x[wx], t, tp, pe, n=n)
    # integrate over each initial position
    for wx in range(W.size):
        W[wx] = np.trapz(U[:, wx] * wo, x)
    return W

# ==========================================
# tools


@numba.jit(nopython=True)
def get_timescale(pe):
    return 4*pe / (4*np.pi**2 + pe**2)

# ==========================================
# initial distributions to try


def gauss(x, mu, sig):
    '''
    Gaussian subject to boundaries of the problem, -0.5<= x <= 0.5.
    '''
    G = np.exp(-0.5 * (x - mu)**2 / sig)
    return G / np.trapz(G, x)


def uniform(x):
    '''
    Uniform distribution for x \in [-0.5, 0.5]
    '''
    return np.ones(x.size) / (x.max() - x.min())
