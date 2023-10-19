import numpy as np 
from time import perf_counter
from matplotlib import pyplot as plt 


def power_iteration(A,u,nit):
    norms = np.zeros((nit,))
    times = np.zeros((nit,))
    tic = perf_counter()

    A = A.T@A

    for i in range(nit):
        u = A@u
        norm = np.linalg.norm(u)
        u = u/norm
        norms[i] = norm**0.5
        times[i] = perf_counter()-tic

    return norms, times


def accelerated_power_iteration(A,u,nit, beta=0.1):
    norms = np.zeros((nit,))
    times = np.zeros((nit,))
    tic = perf_counter()
    A = A.T@A
    old_u = u
    u = A@u 
    old_u = old_u/ np.linalg.norm(u)
    u = u / np.linalg.norm(u)
    for i in range(nit):
        Au = A@u
        u, old_u = Au - beta*old_u, u
        norm = np.linalg.norm(u)
        u = u/norm
        old_u /= norm
        norms[i] = np.linalg.norm(Au)**0.5
        times[i] = perf_counter()-tic

    return np.array([norms, times])


def gramm_iteration(A,nit):
    norms = np.zeros((nit,))
    times = np.zeros((nit,))
    tic = perf_counter()
    r = 0
    fro2 = (A*A).sum()

    for i in range(1,1+nit):
        r = 2*r + np.log(fro2)
        A = A.T@A/fro2
        fro2 = (A*A).sum()
        norms[i-1] = fro2**(0.5*2**-i) * np.exp((2**-i) * r)
        times[i-1] = perf_counter()-tic
    return np.array([norms, times])

def svd(A):
    tic = perf_counter()
    ret = np.array([np.linalg.norm(A, ord=2)])
    return np.array([ret, np.array([perf_counter() - tic])])


def make_graph(A, u, nit_gi=20, nit_pi=1000, reps=10):
    sigma, tmax = np.mean([svd(A) for _ in range(reps)], axis=0); plt.scatter(tmax, 0, label='SVD')
    norms, time = np.mean([accelerated_power_iteration(A,u,nit_pi, 0.24) for _ in range(reps)], axis=0); plt.plot(time, norms/sigma-1, label='accelerated PI')
    norms, time = np.mean([power_iteration(A,u,nit_pi) for _ in range(reps)], axis=0); plt.plot(time, norms/sigma-1, label='PI')
    norms, time = np.mean([gramm_iteration(A,nit_gi) for _ in range(reps)], axis=0); plt.plot(time, norms/sigma-1, label='GI')
    plt.legend()
    plt.yscale('symlog',linthresh=10**-14)
    plt.ylim(-1,1)
    plt.xlim(0, tmax*1.05)
    plt.show()


if(__name__ == '__main__'):
    n=3000
    nit_pi = 1000
    nit_gi = 20 
    

    A = np.random.randn(n,n)
    u = np.random.randn(A.shape[1])
    u = u/np.linalg.norm(u)
    make_graph(A, u, nit_gi=nit_gi, nit_pi=nit_pi, reps=2)
