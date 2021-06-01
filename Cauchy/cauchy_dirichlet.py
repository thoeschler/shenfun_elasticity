from shenfun_elasticity.Solver.stresses import cauchy_stresses
from shenfun_elasticity.Solver.solve_elastic_problem import solve_cauchy_elasticity
from shenfun_elasticity.Solver.plot_template import save_disp_figure, save_cauchy_stress
from shenfun_elasticity.Solver.change_dimensions import get_dimensionless_values, get_dimensionful_values
from shenfun_elasticity.Solver.body_forces import body_forces_cauchy
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
u0 = 1.
# manufactured solution
ua = (u0*((1+x/l)*(y/h)**2*(1-y/h)**2*sin(2*pi*x/l)*cos(3*pi*y/h) + x/l*4*y/h*(1-y/h)), u0*x/l*(1-x/l)*sin(2*pi*y/h))
# elastic constants
E = 400. # Young's modulus
nu = 0.4 # Poisson's ratio
lambd = E*nu/((1+nu)*(1-2*nu))
mu = E/(2*(1+nu))
# body forces
body_forces = body_forces_cauchy(u=ua, material_parameters=(lambd, mu))
# boundary conditions
bc = (((0., u0*4*y/h*(1-y/h)), (0., 0.)), ((0., 0.), (0., 0.)))
# size of discretization
for z in range(10, 41, 2):
    # size of discretization
    N = z
    
    # get dimensionless values
    dom_dimless, bc_dimless, body_forces_dimless, material_parameters_dimless, u_ana_dimless = get_dimensionless_values(
            dom=domain, boundary_conditions=bc, body_forces=body_forces, material_parameters=(lambd, mu), 
            nondim_disp=u0, nondim_length=l, nondim_mat_param=lambd, u_ana=ua
            )
    
    # calculate solution
    u_hat_dimless = solve_cauchy_elasticity(
        N=N, dom=dom_dimless, boundary_conditions=bc_dimless, body_forces=body_forces_dimless, \
        material_parameters=material_parameters_dimless, measure_time=False, compute_error=True, u_ana=u_ana_dimless)
    
    # get dimensionfull values
    u_hat = get_dimensionful_values(
        u_hat_dimless=u_hat_dimless, boundary_conditions=bc, 
        nondim_disp=u0, nondim_length=l
        )
    
    # calculate stresses
    T = cauchy_stresses(material_parameters=(lambd, mu), u_hat=u_hat)
    
    # save displacement as png
    save_disp_figure(u_hat, multiplier=5.0)
    
    # save stresses as png
    save_cauchy_stress(T)