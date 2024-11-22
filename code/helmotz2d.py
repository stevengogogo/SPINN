# Solve Poisson equation in 2D using SPINN
# \nabla^2 u(x, y) = 20pi^2 sin(2pi x) sin(4pi y) on [0,1]x[0,1]
# Zero Dirichlet boundary condition on boundary
#%%
import numpy as np
import torch

from common import tensor
from spinn2d import Plotter2D, SPINN2D, App2D
from pde2d_base import RegularPDE

PI = np.pi
a1 = 1.
a2 = 1.
k=1.
xspan = yspan = [-3,3]
def q(x, y):

    E1 =  - (a1 * torch.pi)**2 * torch.sin(a1 * torch.pi * x) * torch.sin(a2 * torch.pi * y)
    E2 =  - (a2 * torch.pi)**2 * torch.sin(a1 * torch.pi * x) * torch.sin(a2 * torch.pi * y)
    E3 = k**2 * torch.sin(a1 * torch.pi * x) * torch.sin(a2 * torch.pi * y)
    
    return E1 + E2 + E3 

class Helmotz2D(RegularPDE):
    def __init__(self, n_nodes, ns, nb=None, nbs=None, sample_frac=1.0):
        self.sample_frac = sample_frac

        # Interior nodes
        n = round(np.sqrt(n_nodes) + 0.49)
        self.n = n
        dxb2 = 0.5/(n + 1)
        xl, xr = dxb2, 1.0 - dxb2
        sl = slice(xl, xr, n*1j)
        x, y = np.mgrid[sl, sl]
        self.i_nodes = (x.ravel(), y.ravel())

        # Fixed nodes
        nb = n if nb is None else nb
        self.nb = nb
        dxb2 = 0.5/(nb)
        _x = np.linspace(dxb2, 1.0 - dxb2, nb)
        _o = np.ones_like(_x)
        x = np.hstack((_x, _o, _x, 0*_o))
        y = np.hstack((_o*0, _x, _o, _x))
        self.f_nodes = (x, y)

        # Interior samples
        self.ns = ns = round(np.sqrt(ns) + 0.49)
        dxb2 = 0.5/(ns)
        xl, xr = dxb2, 1.0 - dxb2
        sl = slice(xl, xr, ns*1j)
        xs, ys = (tensor(t.ravel(), requires_grad=True)
                  for t in np.mgrid[sl, sl])

        xs = xs * (xspan[1] - xspan[0]) + xspan[0]
        ys = ys * (yspan[1] - yspan[0]) + yspan[0]

        self.p_samples = (xs, ys)

        self.n_interior = len(self.p_samples[0])
        self.rng_interior = np.arange(self.n_interior)
        self.sample_size = int(self.sample_frac*self.n_interior)

        # Boundary samples
        nbs = ns if nbs is None else nbs
        self.nbs = nbs
        sl = slice(0.0, 1.0, nbs*1j)
        x, y = np.mgrid[sl, sl]
        cond = ((x < xl) | (x > xr)) | ((y < xl) | (y > xr))
        xb, yb = (tensor(t.ravel(), requires_grad=True)
                  for t in (x[cond], y[cond]))
        xb = xb * (xspan[1] - xspan[0]) + xspan[0]
        yb = yb * (yspan[1] - yspan[0]) + yspan[0]
        self.b_samples = (xb, yb)

    def pde(self, x, y, u, ux, uy, uxx, uyy):
        return uxx + uyy + k**2 * u - q(x,y)

    def has_exact(self):
        return True

    def exact(self, x, y):
        return np.sin(a1 * np.pi * x) * np.sin(a2 * np.pi * y)

    def boundary_loss(self, nn):
        xb, yb = self.boundary()
        xbn, ybn = (t.detach().cpu().numpy() for t in (xb, yb))

        u = nn(xb, yb)
        ub = tensor(self.exact(xbn, ybn))
        bc = u - ub
        return (bc**2).sum()


#%%
if __name__ == '__main__':
    app = App2D(
        pde_cls=Helmotz2D, nn_cls=SPINN2D,
        plotter_cls=Plotter2D
    )
    app.run(nodes=500, samples=1000, n_train=25000, lr=1e-3, tol=1e-3)

# %%
