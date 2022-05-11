import shenfun as sf
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

N = 10
L = sf.FunctionSpace(N, family='legendre', bc=(0, 0))
print(L.__class__)

# plot quadrature points
quad_points = L.mpmath_points_and_weights()[0]
plt.scatter(quad_points, np.zeros_like(quad_points))

# a projection
x = sp.symbols('x', real=True)
f = 1 - 1 / 2 * (3 * x ** 2 - 1)  # the first Shen-Dirichlet basis function
P = sf.Function(L)
func = sf.project(f, L)
func_quad = sf.Array(L, buffer=f)

plt.scatter(quad_points, func_quad)

# should be orthogonal to all others except 0th and 2nd
scalprod = L.scalar_product(func_quad)
assert np.allclose(scalprod[np.r_[1, 3:N - 1]], 0)
assert np.logical_not(np.allclose(scalprod[np.r_[0, 2]], 0))
