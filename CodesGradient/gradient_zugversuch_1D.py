import matplotlib.pyplot as plt
from sympy import symbols, cos, sin, pi
import os
os.chdir("/home/student01/tubCloud/Gradientenelastizität/Codes/")
from stresses import hyper_stresses, cauchy_stresses, traction_vector_gradient
from solve_elastic_problem import solve_gradient_elasticity
from body_forces import body_forces_gradient
from plot_template import save_disp_figure, save_cauchy_stress, save_hyper_stress, save_traction_vector_gradient
os.chdir("/home/student01/tubCloud/Gradientenelastizität/CodesGradient/")

family = 'legendre'
x, y = symbols("x,y")
# computational domain
l = 100.
h = l/2
domain_x = (0, l)
domain_y = (0, h)
domain = (domain_x, domain_y)
# displacement value
u0 = -1.
# elastic constants
E = 400. # Young's modulus
nu = 0.4 # Poisson's ratio
lambd = E*nu/((1+nu)*(1-2*nu))
mu = E/(2*(1+nu))
c1 = 0.01
c2 = 0.01
c3 = 0.01
c4 = 0.01
c5 = 0.01
# analytical solution
ua = (x/l*u0, nu/(1-nu)*u0/l*(h-y))
# body forces
b = (0., 0.)
# boundary conditions
bc = (((0., u0), None), (None, 'upperdirichlet'))
# size of discretization
for z in range(30, 32, 2):
    plt.close('all')
    # size of discretization
    N = z
    # calculate solution
    u_hat = solve_gradient_elasticity(N=N, dom=domain, boundary_conditions=bc, body_forces=b, \
                                      material_parameters=(lambd, mu, c1, c2, c3, c4, c5), nondim_disp=u0, \
                                      nondim_length=l, nondim_mat_param=lambd, plot_disp=False, measure_time=False, \
                                      compute_error=True, u_ana=ua)
    # calculate stresses
    T = cauchy_stresses(material_parameters=(lambd, mu), u_hat=u_hat, plot=False)
    T3 = hyper_stresses(material_parameters=(c1, c2, c3, c4, c5), u_hat=u_hat, plot=True)
    t_upper_lower = traction_vector_gradient(T, T3, normal_vector=(0., 1.))
    t_left_right = traction_vector_gradient(T, T3, normal_vector=(1., 0.))
    
    # save displacement as png
    save_disp_figure(u_hat, multiplier=5.0)

    # save stresses as png
    save_cauchy_stress(T)
    save_hyper_stress(T3)
    save_traction_vector_gradient(t_upper_lower, (0., 1.))
    save_traction_vector_gradient(t_left_right, (1., 0.))