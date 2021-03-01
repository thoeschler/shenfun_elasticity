from shenfun import Function, dx, FunctionSpace, TensorProductSpace, VectorSpace, Array, div, project, grad
from sympy import cos, symbols, pi, sin
from mpi4py import MPI
from mpi4py_fft.pencil import Subcomm

subcomms = Subcomm(MPI.COMM_WORLD, (0, 1))

x, y = symbols('x, y')
u = cos(x)*sin(y)

F1 = FunctionSpace(N=11, family='legendre', domain=(-3.141592654/2, 3.141592654/2), bc=(0, 0))
F2 = FunctionSpace(N=11, family='legendre', domain=(-3.141592654/2, 3.141592654/2), bc=(-cos(x), cos(x)))

T = TensorProductSpace(subcomms, [F1, F2])
V = VectorSpace([T, T])

F0 = FunctionSpace(N=11, family='legendre', domain=(-3.141592654/2, 3.141592654/2), bc=None)

T_none = TensorProductSpace(subcomms, [F0, F0])
V_none = VectorSpace([T_none, T_none])

u_quad = Array(T, buffer=u)

u_hat = u_quad.forward()

d = grad(u_hat)

d_quad = project(d[0], T).backward()

d_quad_real = project(d[0], T_none).backward()

print(u_quad)

val = dx(u_quad)

print(val)