import matplotlib.pyplot as plt
from sympy import symbols, cos, sin, pi
import os
os.chdir("/home/student01/tubCloud/Gradientenelastizität/Codes/")
from stresses import cauchy_stresses
from solve_elastic_problem import solve_cauchy_elasticity
from body_forces import body_forces_cauchy
from plot_template import save_disp_figure, save_cauchy_stress
os.chdir("/home/student01/tubCloud/Gradientenelastizität/CodesCauchy/")

family = 'legendre'
x, y = symbols("x,y")
# computational domain
l = 100.
h = l/2
domain_x = (0, l)
domain_y = (0, h)
domain = (domain_x, domain_y)
# displacement value
u0 = 1.
# manufactured solution
ua = (u0*((1+x/l)*(y/h)**2*(1-y/h)**2*sin(2*pi*x/l)*cos(3*pi*y/h) + x/l*4*y/h*(1-y/h)), u0*x/l*(1-x/l)*sin(2*pi*y/h))
# elastic constants
E = 400. # Young's modulus
nu = 0.4 # Poisson's ratio
lambd = E*nu/((1+nu)*(1-2*nu))
mu = E/(2*(1+nu))
# body forces
b = body_forces_cauchy(u=ua, material_parameters=(lambd, mu))
# boundary conditions
bc = (((0., u0*4*y/h*(1-y/h)), (0., 0.)), ((0., 0.), (0., 0.)))
# size of discretization
for z in range(30, 32, 2):
    plt.close('all')
    # size of discretization
    N = z
    # calculate solution
    u_hat = solve_cauchy_elasticity(N=N, dom=domain, boundary_conditions=bc, body_forces=b, material_parameters=(lambd, mu),\
                                    nondim_disp=u0, nondim_length=l, nondim_mat_param=lambd, plot_disp=False, \
                                    measure_time=False, compute_error=True, u_ana=ua)
    # calculate stresses
    T = cauchy_stresses(material_parameters=(lambd, mu), u_hat=u_hat, plot=False)
    
    # save displacement as png
    save_disp_figure(u_hat, multiplier=5.0)
    
    # save stresses as png
    save_cauchy_stress(T)