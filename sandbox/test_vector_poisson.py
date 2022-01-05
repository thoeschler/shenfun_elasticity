from shenfun import FunctionSpace, Array, TrialFunction, TestFunction, \
    TensorProductSpace, comm, inner, grad, VectorSpace, BlockMatrix
import sympy as sp
import numpy as np

x, y = sp.symbols("x, y")

# manufactured solution with:
# x - component:
# ux'(x = - 1) = -4 (y - 1) (y + 1), ux(x = 1) = 0, ux(y = - 1) = 0, ux(y = 1) = 0
# y - component:
# uy(x = - 1) = 0, uy(x = 1) = 0, ux(y = - 1) = 0, ux'(y = 1) = 2 (x -1)**2
ua = ((x - 1) ** 2 * (y - 1) * (y + 1),
      5 * (y + 1) ** 2 * (x - 1) * (x + 1))
f = (- ua[0].diff(x, 2) - ua[0].diff(y, 2),
     - ua[1].diff(x, 2) - ua[1].diff(y, 2))

neumann_condition_x = ua[0].diff(x).evalf(subs={x: - 1})
neumann_condition_y = ua[1].diff(y).evalf(subs={y: 1})

FXX = FunctionSpace(20, family='legendre', bc=(None, 0))
FXY = FunctionSpace(20, family='legendre', bc=(0, 0))
FYX = FunctionSpace(20, family='legendre', bc=(0, 0))
FYY = FunctionSpace(20, family='legendre', bc=(0, None))

TX = TensorProductSpace(comm, (FXX, FXY))
TY = TensorProductSpace(comm, (FYX, FYY))

V = VectorSpace([TX, TY])

u = TrialFunction(V)
v = TestFunction(V)
mat = inner(grad(u), grad(v))
fj = Array(V, buffer=f)
rhs = inner(v, fj)

# boundary integrals
# x - component
v_bndry_x = TestFunction(FXY)
gn_x = Array(FXY, buffer=neumann_condition_x)
evaluate_bndry_x = FXX.evaluate_basis_all(-1)
project_gn_x = inner(gn_x, v_bndry_x)
bndry_integral_x = - np.outer(evaluate_bndry_x, project_gn_x)
# y - component
v_bndry_y = TestFunction(FYX)
gn_y = Array(FYX, buffer=neumann_condition_y)
evaluate_bndry_y = FYY.evaluate_basis_all(1)
project_gn_y = inner(gn_y, v_bndry_y)
bndry_integral_y = np.outer(project_gn_y, evaluate_bndry_y)

# add boundary integrals to rhs
rhs[0] += bndry_integral_x
rhs[1] += bndry_integral_y

M = BlockMatrix(mat)

u_hat = M.solve(rhs)

u_ana = Array(V, buffer=ua)
l2_error = np.sum(
    [np.linalg.norm(num.backward() - ana) for num, ana in zip(u_hat, u_ana)]
     )
print(l2_error)
