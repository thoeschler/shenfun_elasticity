from shenfun import FunctionSpace, Array, project, TrialFunction, TestFunction, \
    TensorProductSpace, comm, inner, grad, Function, div
import sympy as sp
import shenfun as sf
import matplotlib.pyplot as plt
import numpy as np

x, y = sp.symbols("x, y")

ua = (x - 1) * (y - 1) * (y + 1)
f = - ua.diff(x, 2) - ua.diff(y, 2)

neumann_condition = ua.diff(x).evalf(subs={x: - 1})
FX = FunctionSpace(20, family='legendre',
                   bc=(None, 0))
FY = FunctionSpace(20, family='legendre', bc=(0, 0))
T = TensorProductSpace(comm, (FX, FY))

# project_f = inner(Array(FX, buffer=f), TestFunction(FX))
# print(f, project_f)

u = TrialFunction(T)
v = TestFunction(T)
mat = inner(grad(u), grad(v))
print(mat)
fj = Array(T, buffer=f)
rhs = inner(v, fj)

# boundary integral for x = -1
v_bndry = TestFunction(FY)

gn = Array(FY, buffer=neumann_condition)
evaluate_x_bndry = Array(FX, buffer=FX.evaluate_basis_all(-1))
print(evaluate_x_bndry.shape)
project_g = inner(gn, v_bndry)
# print(project_g)

bndry_integral = np.outer(evaluate_x_bndry, project_g)
# print(bndry_integral)

# print(rhs)
rhs += bndry_integral
# print(rhs)

Sol = sf.la.SolverGeneric2ND(mat)

u_hat = Function(T)
u_hat = Sol(rhs, u_hat)

u_ana = Array(T, buffer=ua)
l2_error = np.linalg.norm(u_hat.backward() - u_ana)
print(l2_error)
xx, yy = T.mesh()

X, Y = np.meshgrid(xx.squeeze(), yy.squeeze())
fig = plt.figure()

ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(X, Y, np.transpose(u_ana))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_surface(X, Y, np.transpose(u_hat.backward()))
ax.set_xlabel('x')
ax.set_ylabel('y')
