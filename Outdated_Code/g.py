from pde import PDE, ScalarField, UnitGrid

grid = UnitGrid([32,32])
state = ScalarField.random_uniform(grid)
eq = PDE({"u":"-gradient_squared(u)/2 - laplace(u + laplace(u))"})
result = eq.solve(state, t_range=10, dt=0.01)
result.plot()
