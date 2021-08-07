from shenfun_elasticity.Solver.stresses import cauchy_stresses, hyper_stresses, traction_vector_gradient
from shenfun_elasticity.Solver.solve_elastic_problem import solve_gradient_elasticity
from shenfun_elasticity.Solver.plot_template import save_disp_figure, save_cauchy_stress, \
    save_hyper_stress, save_traction_vector_gradient
from shenfun_elasticity.Solver.change_dimensions import get_dimensionless_values, get_dimensionful_values
from math import floor

# computational domain
l = 0.1
h = l/2
domain_x = (0, l)
domain_y = (0, h)
domain = (domain_x, domain_y)
# displacement value
u0 = l/10
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
# body forces
body_forces = (0., 0.)
# boundary conditions
bc = ((None, (0, u0)),(None, (0, 0)))
# size of discretization
for z in (10.,):
    c1 = z
    c2 = z
    c3 = z
    c4 = z
    c5 = z
    # size of discretization
    N = 50
    
    # get dimensionless values
    dom_dimless, bc_dimless, body_forces_dimless, material_parameters_dimless = get_dimensionless_values(
            dom=domain, boundary_conditions=bc, body_forces=body_forces, \
            material_parameters=(lambd, mu, c1, c2, c3, c4, c5), nondim_disp=u0, nondim_length=l, \
            nondim_mat_param=lambd
            )
        
    # calculate solution
    u_hat_dimless = solve_gradient_elasticity(
        N=N, dom=dom_dimless, boundary_conditions=bc_dimless, body_forces=body_forces_dimless, \
        material_parameters=material_parameters_dimless, measure_time=False, compute_error=True
        )
        
    # get dimensionfull values
    u_hat = get_dimensionful_values(
        u_hat_dimless=u_hat_dimless, boundary_conditions=bc, 
        nondim_disp=u0, nondim_length=l
        )
        
    # calculate stresses
    T = cauchy_stresses(material_parameters=(lambd, mu), u_hat=u_hat)
    T3 = hyper_stresses(material_parameters=(c1, c2, c3, c4, c5), u_hat=u_hat)
    
    # save displacement as png
    save_disp_figure(u_hat, multiplier=1.0)
    t_upper_lower = traction_vector_gradient(T, T3, normal_vector=(0., 1.))
    t_left_right = traction_vector_gradient(T, T3, normal_vector=(1., 0.))
    
#    # save stresses as png
    save_cauchy_stress(T)
    save_hyper_stress(T3)
#    save_traction_vector_gradient(t_upper_lower, (0., 1.))
#    save_traction_vector_gradient(t_left_right, (1., 0.))


    # postprocessing
    x_, y_ = u_hat.function_space().spaces[0].local_mesh()
    x_dimless, y_dimless = u_hat_dimless.function_space().spaces[0].local_mesh()
    u_phy = u_hat.backward()
    u_phy_dimless = u_hat_dimless.backward()
    ux_left = u_phy[0][0]
    ux_left_dimless = u_phy_dimless[0][0]
    ux_right = u_phy[0][-1]
    ux_right_dimless = u_phy_dimless[0][-1]
    ux_center = u_phy[0][floor(N / 2)]
    ux_center_dimless = u_phy_dimless[0][floor(N / 2)]
    
    with open('ux_center_l_' + str(l).replace('.', 'p') +  '_h_' + str(h).replace('.', 'p') + '_ci_'
        + str(c1).replace('.', 'p') + '_num.dat', 'w') as f:
        for yy, ux in zip(y_[0], ux_center):
            f.write(str(yy) + ' ' + str(ux) + '\n')