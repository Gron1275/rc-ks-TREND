import numba as nb
from pde import PDEBase, ScalarField, UnitGrid, CartesianGrid, MemoryStorage,plot_kymograph, FileStorage


class KuramotoSivashinskyPDE(PDEBase):
    """Implementation of the normalized Kuramoto–Sivashinsky equation"""

    def __init__(self, bc="periodic"):
        super().__init__()
        self.bc = bc
    def evolution_rate(self, state, t=0):
        """implement the python version of the evolution equation"""
        state_lap = state.laplace(bc=self.bc)
        state_lap2 = state_lap.laplace(bc=self.bc)
        state_grad_sq = state.gradient_squared(bc=self.bc)
        return -state_grad_sq / 2 - state_lap - state_lap2
    def _make_pde_rhs_numba(self, state):
        """nunmba-compiled implementation of the PDE"""
        deriv = state.grid.make_operator("d_dx", bc=self.bc)
        laplace = state.grid.make_operator("laplace", bc=self.bc)
        gradient_squared = state.grid.make_operator("gradient_squared", bc=self.bc)

        @nb.njit
        def pde_rhs(data, t):
            return - data * deriv(data) - laplace(data + laplace(data))

        return pde_rhs


grid = CartesianGrid([[0,22]],[64], periodic=True)
state = ScalarField.random_uniform(grid,vmin=-0.6,vmax=0.6)
storage = MemoryStorage()
output = FileStorage("jun11numba.hdf")
eq = KuramotoSivashinskyPDE()
eq.solve(state, t_range=8000,dt=0.25, solver="scipy", tracker=[storage.tracker(0.25),output.tracker(0.25)])
plot_kymograph(storage,transpose=True)
# looks different from the oneD file. maybe it is the difference in appearance of the KS eq?
