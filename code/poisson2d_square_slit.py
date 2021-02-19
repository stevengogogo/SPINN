import numpy as np

from common import tensor
from pde2d_base import RegularPDE
from spinn2d import Plotter2D, App2D, SPINN2D


class SquareSlit(RegularPDE):
    def __init__(self, n_nodes, ns, nb=None, nbs=None, sample_frac=1.0):
        self.sample_frac = sample_frac

        # Interior nodes
        n = round(np.sqrt(n_nodes) + 0.49)
        self.n = n
        dxb2 = 1.0/(n + 1)
        xl, xr = dxb2 - 1.0, 1.0 - dxb2
        sl = slice(xl, xr, n*1j)
        x, y = np.mgrid[sl, sl]
        cond = ~((x >= 0) & (np.abs(y) < dxb2))
        self.i_nodes = (x[cond].ravel(), y[cond].ravel())

        # Fixed nodes
        nb = n if nb is None else nb
        self.nb = nb
        dxb2 = 1.0/(nb)
        _x = np.linspace(dxb2 - 1.0, 1.0 - dxb2, nb)
        _o = np.ones_like(_x)
        nslit = int(nb//2 + 1)
        x = np.hstack((_x, _o, _x, -1*_o, np.linspace(0, 1, nslit)))
        y = np.hstack((_o*-1, _x, _o, _x, np.zeros(nslit)))
        self.f_nodes = (x, y)

        # Interior samples
        self.ns = ns = round(np.sqrt(ns) + 0.49)
        dxb2 = 1.0/(ns)
        xl, xr = dxb2 - 1.0, 1.0 - dxb2
        sl = slice(xl, xr, ns*1j)
        x, y = np.mgrid[sl, sl]
        cond = ~((x >= 0) & (np.abs(y) < dxb2))
        xs, ys = (tensor(t.ravel(), requires_grad=True)
                  for t in (x[cond], y[cond]))
        self.p_samples = (xs, ys)

        self.n_interior = len(self.p_samples[0])
        self.rng_interior = np.arange(self.n_interior)
        self.sample_size = int(self.sample_frac*self.n_interior)

        # Boundary samples
        nbs = ns if nbs is None else nbs
        self.nbs = nbs
        sl = slice(-1.0, 1.0, nbs*1j)
        x, y = np.mgrid[sl, sl]
        cond = ((x < xl) | (x > xr)) | ((y < xl) | (y > xr))
        x, y = x[cond].ravel(), y[cond].ravel()
        nslit = int(nbs//2 + 1)
        xb = np.hstack((x, np.linspace(0, 1.0, nslit)))
        yb = np.hstack((y, np.zeros(nslit)))
        xb, yb = (tensor(t, requires_grad=True) for t in (xb, yb))
        self.b_samples = (xb, yb)

    def plot_points(self):
        n = self.ns*2
        x, y = np.mgrid[-1:1:n*1j, -1:1:n*1j]
        return x, y

    def pde(self, x, y, u, ux, uy, uxx, uyy):
        return uxx + uyy + 1.0

    def has_exact(self):
        return False

    def boundary_loss(self, nn):
        xb, yb = self.boundary()
        u = nn(xb, yb)
        ub = 0.0
        bc = u - ub
        return (bc**2).sum()


if __name__ == '__main__':
    app = App2D(
        pde_cls=SquareSlit, nn_cls=SPINN2D,
        plotter_cls=Plotter2D
    )
    app.run(nodes=50, samples=200, lr=1e-2)