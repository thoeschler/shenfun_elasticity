import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols
from mpi4py import MPI
from shenfun import inner, Dx, TestFunction, extract_bc_matrices, \
    TrialFunction, Function, Array, FunctionSpace, ShenfunFile
from mpi4py_fft import generate_xdmf

comm = MPI.COMM_WORLD

x = symbols("x")
family = 'legendre'

# Elastische Konstanten
F = 100 # Kraft
E = 210000 # in MPa
nu = 0.3
lambd = E*nu/((1+nu)*(1-2*nu))
mu = 79300
I = 1000
l = 1
fe = (0, 0)

# computational domain
domain = (0, l)

ue = 8*F/(6*E*I) * (x/l)**2 * (3 - x/l)

# size of discretization
N = 6

# Function Space 
B = FunctionSpace(N, family=family, domain=domain, bc=['beamfixedfree', (0, 0, 0, -F/(E*I))])
X = B.mesh()
u = TrialFunction(B)
v = TestFunction(B)

fj = Array(B)

# system matrix
matrix = inner(Dx(v, 0, 2), Dx(u, 0, 2))
f_hat = inner(v, fj)

# Function to hold the solution
u_hat = Function(B).set_boundary_dofs()

# Some work required for inhomogeneous boundary conditions only
if B.has_nonhomogeneous_bcs:
    bc_mats = extract_bc_matrices([matrix])

    w0 = np.zeros_like(u_hat)
    for m in bc_mats:
        f_hat -= m.matvec(u_hat, w0)

# solver
uh_hat = matrix[0].solve(f_hat)
uh_hat += u_hat

uj = Array(B)

# numerical solution
u_sol = u_hat.backward()

# exact solution
u_sol_exact = Array(B, buffer=ue)

plt.plot(X, u_sol)
plt.show()

plt.plot(X, u_sol_exact)
plt.show()


# print error
print(comm.reduce(np.linalg.norm((u_sol_exact - u_sol))))

# output
output_file_displacement = ShenfunFile('elastic_balken_fest_frei_displacement', B, backend='hdf5', mode='w', uniform=True)
displacement_output = uh_hat.backward(uniform=True)
output_file_displacement.write(0, {'displacement': [displacement_output]}, as_scalar=True)

generate_xdmf('elastic_balken_fest_frei_displacement.h5')