import shenfun as sf
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

B = sf.FunctionSpace(20, family='legendre')

r, phi = sp.symbols('x,y', real=True, positive=True)
psi = (r, phi)
# position vector
rv = (r * sp.cos(phi), r * sp.sin(phi))

# some parameters
r = 2

N, M = 30, 30

R = sf.FunctionSpace(N, family='legendre', domain=(0, r), dtype='D')
PHI = sf.FunctionSpace(M, family='fourier', domain=(0,  2.0 * np.pi))

T = sf.TensorProductSpace(sf.comm, (R, PHI), coordinates=(psi, rv))

u = sf.TrialFunction(T)
v = sf.TestFunction(T)

X, Y = T.cartesian_mesh()

plt.scatter(X, Y)

coors = T.coors

# test coordinates
assert coors.is_cartesian is False
assert coors.is_orthogonal is True

print('position vector\n', coors.rv)
print('covariante basis\n', coors.b)
print('normed covariant basis\n', coors.e)
print('contravariant basis\n', coors.get_contravariant_basis())
print('covariante metric tensor\n', coors.get_metric_tensor(kind='covariant'))
print('determinant of covariant metric tensor\n', coors.get_sqrt_det_g(covariant=True))
print('determinant of contravariant metric tensor\n', coors.get_sqrt_det_g(covariant=False))
print('christoffel symbols\n', coors.get_christoffel_second())
