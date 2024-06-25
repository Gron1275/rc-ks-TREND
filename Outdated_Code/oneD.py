from pde import PDE, CartesianGrid, MemoryStorage, ScalarField, plot_kymograph, FileStorage

eq = PDE({"v":"-laplace(v) - laplace(laplace(v)) - v * d_dx(v)"})
grid = CartesianGrid([[0,22]],[64], periodic=True)
state = ScalarField.random_uniform(grid,vmin=-3,vmax=3)
storage = MemoryStorage()
output = FileStorage("jun11.hdf")

eq.solve(state, t_range=4000,dt=0.25, solver="scipy", tracker=[storage.tracker(0.25),output.tracker(0.25)])


print(storage.extract_time_range(10.0))
output.end_writing()
print(output.data)
plot_kymograph(storage,transpose=True)

