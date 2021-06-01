from shenfun_elasticity.Solver.stresses import cauchy_stresses, hyper_stresses, traction_vector_gradient
from shenfun_elasticity.Solver.solve_elastic_problem import solve_gradient_elasticity
from shenfun_elasticity.Solver.plot_template import save_disp_figure, save_cauchy_stress, \
    save_hyper_stress, save_traction_vector_gradient
from shenfun_elasticity.Solver.change_dimensions import get_dimensionless_values, get_dimensionful_values
from shenfun_elasticity.Solver.body_forces import body_forces_gradient
from sympy import symbols, sin, cos, pi

# some parameters
x, y = symbols("x,y")
# computational domain
l = 100.
h = l/2
domain_x = (0, l)
domain_y = (0, h)
domain = (domain_x, domain_y)
# displacement value
u0 = 1
# manufactured solution
ua = (u0*(50*(1-x/l)**2*(x/l)**2*(y/h)**2*(1-y/h)**2*sin(2*pi*x/l)*cos(3*pi*y/h) + 16*(1-y/h)**2*(y/h)**2*(3*(x/l)**2 - 2*(x/l)**3)), \
      u0*50*(1-x/l)**2*(x/l)**2*(y/h)**2*(1-y/h)**2*sin(2*pi*y/h)*cos(3*pi*x/l))
# + 16*(1-y/h)**2*(y/h)**2*(3*(x/l)**2 - 2*(x/l)**3)
# elastic constants
E = 400. # Young's modulus
nu = 0.4 # Poisson's ratio
lambd = E*nu/((1+nu)*(1-2*nu))
mu = E/(2*(1+nu))
c1 = 0.1
c2 = 0.1
c3 = 0.1
c4 = 0.1
c5 = 0.1
# body forces
body_forces = body_forces_gradient(u=ua, material_parameters=(lambd, mu, c1, c2, c3, c4, c5))
# boundary conditions
bc = (((0, u0*16*(1-y/h)**2*(y/h)**2, 0, 0), (0, 0, 0, 0)), ((0, 0, 0, 0), (0, 0, 0, 0)))
#bc = (((0, 0, 0, 0), (0, 0, 0, 0)), ((0, 0, 0, 0), (0, 0, 0, 0)))
# size of discretization
for z in range(10, 41, 2):
    # size of discretization
    N = z

    # get dimensionless values
    dom_dimless, bc_dimless, body_forces_dimless, material_parameters_dimless, u_ana_dimless = get_dimensionless_values(
            dom=domain, boundary_conditions=bc, body_forces=body_forces, \
            material_parameters=(lambd, mu, c1, c2, c3, c4, c5), nondim_disp=u0, nondim_length=l, \
            nondim_mat_param=lambd, u_ana=ua
            )

    # calculate solution
    u_hat_dimless = solve_gradient_elasticity(
        N=N, dom=dom_dimless, boundary_conditions=bc_dimless, body_forces=body_forces_dimless, \
        material_parameters=material_parameters_dimless, measure_time=False, compute_error=True, \
        u_ana=u_ana_dimless
        )

    # get dimensionfull values
    u_hat = get_dimensionful_values(
        u_hat_dimless=u_hat_dimless, boundary_conditions=bc, 
        nondim_disp=u0, nondim_length=l
        )

    # calculate stresses
    T = cauchy_stresses(material_parameters=(lambd, mu), u_hat=u_hat)
    T3 = hyper_stresses(material_parameters=(c1, c2, c3, c4, c5), u_hat=u_hat)
#    t_upper_lower = traction_vector_gradient(cauchy_stresses=T, hyper_stresses=T3, normal_vector=(0., 1.))
#    t_left_right = traction_vector_gradient(cauchy_stresses=T, hyper_stresses=T3, normal_vector=(1., 0.))

    # save displacement as png
    save_disp_figure(u_hat, multiplier=5.0)

    # save stresses as png
    save_cauchy_stress(T)
    save_hyper_stress(T3)
#    save_traction_vector_gradient(traction_vector=t_upper_lower, normal_vector=(0., 1.))
#    save_traction_vector_gradient(traction_vector=t_left_right, normal_vector=(1., 0.))