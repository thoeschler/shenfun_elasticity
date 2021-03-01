import matplotlib.pyplot as plt
from sympy import symbols
from mpi4py import MPI
import os
os.chdir("/home/student01/tubCloud/Gradientenelastizität/Codes/")
from stresses import cauchy_stresses
from solve_elastic_problem import solve_cauchy_elasticity
from plot_template import save_disp_figure, save_cauchy_stress
os.chdir("/home/student01/tubCloud/Gradientenelastizität/CodesCauchy/")

comm = MPI.COMM_WORLD
family = 'legendre'
x, y = symbols("x,y")
# computational domain
l = 100.
h = l/2
domain_x = (0, l)
domain_y = (0, h)
domain = (domain_x, domain_y)
# Verschiebungswert
u0 = 1.0
# elastic constants
E = 400. # Young's modulus
nu = 0.4 # Poisson's ratio
lambd = E*nu/((1+nu)*(1-2*nu))
mu = E/(2*(1+nu))
# body forces
b = (0., 0.)
# boundary conditions
bc = (((0, 1), None), ((0, 0), None))
# size of discretization
for z in range(30, 32, 2):
    plt.close('all')
    N = z
    # calculate solution
    u_hat = solve_cauchy_elasticity(N=N, dom=domain, boundary_conditions=bc, body_forces=b, material_parameters=(lambd, mu),\
                                    nondim_disp=u0, nondim_length=l, nondim_mat_param=lambd, plot_disp=False, \
                                    measure_time=False, compute_error=True, u_ana=None)
    # calculate stresses
    T = cauchy_stresses(material_parameters=(lambd, mu), u_hat=u_hat, plot=False)

    # save displacement as png
    save_disp_figure(u_hat, multiplier=5.0)
    
    # save cauchy_stress as png
    save_cauchy_stress(T)