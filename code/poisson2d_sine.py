# Solve Poisson equation in 2D using SPINN
# \nabla^2 u(x, y) = 20pi^2 sin(2pi x) sin(4pi y) on [0,1]x[0,1]
# Zero Dirichlet boundary condition on boundary

import numpy as np
import torch

from common import tensor
from spinn2d import Plotter2D, SPINN2D, App2D
from pde2d_base import RegularPDE

PI = np.pi

class Poisson2D(RegularPDE):
    @classmethod
    def from_args(cls, args):
        return cls(args.nodes, args.samples,
                   args.b_nodes, args.b_samples, args.de)

    @classmethod
    def setup_argparse(cls, parser, **kw):
        super().setup_argparse(parser, **kw)
        p = parser
        p.add_argument(
            '--de', dest='de', default=kw.get('de', 'sin2'),
            choices=['sin2', 'bump'],
            help='Differential equation to solve.'
        )

    def __init__(self, n_nodes, ns, nb=None, nbs=None, de='bump'):
        super().__init__(n_nodes, ns, nb=nb, nbs=nbs)
        self.deq = de

    def pde(self, x, y, u, ux, uy, uxx, uyy):
        f = 20*PI**2*torch.sin(2*PI*x)*torch.sin(4*PI*y)
        return 0.1*(uxx + uyy + f)

    def has_exact(self):
        return True

    def exact(self, x, y):
        return np.sin(2*PI*x)*np.sin(4*PI*y)

    def boundary_loss(self, nn):
        xb, yb = self.boundary()
        xbn, ybn = (t.detach().cpu().numpy() for t in (xb, yb))

        u = nn(xb, yb)
        ub = tensor(self.exact(xbn, ybn))
        bc = u - ub
        return (bc**2).sum()

if __name__ == '__main__':
    app = App2D(
        pde_cls=Poisson2D, nn_cls=SPINN2D,
        plotter_cls=Plotter2D
    )
    app.run(nodes=40, samples=120, lr=1e-2)