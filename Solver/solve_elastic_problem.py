from shenfun import VectorSpace, TensorProductSpace, Array, TrialFunction, TestFunction, inner, grad, div, Dx, \
    extract_bc_matrices, BlockMatrix, Function, FunctionSpace, project, comm
from shenfun.legendre.bases import ShenDirichlet, ShenBiharmonic
from check_solution import check_solution_cauchy, check_solution_gradient
import numpy as np
import sympy
import matplotlib.pyplot as plt
from matplotlib import cm
from plot_template import plot_grid
import time
import math


def solve_cauchy_elasticity(N, dom, boundary_conditions, body_forces, material_parameters, \
                            nondim_disp=1.0, nondim_length=1.0, nondim_mat_param=1.0, plot_disp=False, \
                            measure_time=False, compute_error=False, u_ana=None):

    # assert input
    assert isinstance(N, int)
    assert isinstance(dom, tuple)
    assert isinstance(boundary_conditions, tuple)
    assert isinstance(body_forces, tuple)
    assert isinstance(nondim_disp, float)
    assert isinstance(nondim_length, float)
    assert isinstance(nondim_mat_param, float)
    assert isinstance(material_parameters, tuple)
    assert len(material_parameters) == 2
    if compute_error:
        assert u_ana is None or isinstance(u_ana, tuple)
    
    # start time measurement if desired
    if measure_time:
        time_start=time.time()
    
    # some parameters
    dim = len(dom)
    
    # dimensionless domain    
    dom_dimless = tuple([tuple([dom[i][j]/nondim_length for j in range(dim)]) for i in  range(dim)])
    
    # dimensionless material_parameters
    lambd = material_parameters[0]/nondim_mat_param
    mu = material_parameters[1]/nondim_mat_param
    
    # dimensionless boundary conditions
    bcs_dimless = []
    for i in range(dim): # nb of components
        bcs_for_one_component = []
        for j in range(dim): # nb of bcs for each component
            if isinstance(boundary_conditions[i][j], tuple): # dirichlet-bcs are given as a tuple
                temp_list = []
                for k in range(len(boundary_conditions[i][j])):
                    tmp = boundary_conditions[i][j][k]
                    if isinstance(tmp, sympy.Expr): # coordinate transformation
                        for coord in tmp.free_symbols:
                            tmp = tmp.replace(coord, coord*nondim_length)
                    temp_list.append(tmp / nondim_disp)
                bcs_for_one_component.append(tuple(temp_list))
            elif isinstance(boundary_conditions[i][j], list):
                assert isinstance(boundary_conditions[i][j][0], str)
                assert isinstance(boundary_conditions[i][j][1], tuple)
                temp_list = []
                for k in range(len(boundary_conditions[i][j][1])):
                    tmp = boundary_conditions[i][j][1][k]
                    if isinstance(tmp, sympy.Expr): # coordinate transformation
                        for coord in tmp.free_symbols:
                            tmp = tmp.replace(coord, coord*nondim_length)
                    temp_list.append(tmp / nondim_disp)
                bcs_for_one_component.append([boundary_conditions[i][j][0], tuple(temp_list)]) # change tuple, leave str as it is
            else: # e.g. 'upperdirichlet', 'lowerdirichlet', None
                bcs_for_one_component.append(boundary_conditions[i][j])
        bcs_dimless.append(bcs_for_one_component)
    bcs_dimless = tuple(tuple( bc for bc in comp ) for comp in bcs_dimless) # transform lists to tuples    
    
    # dimensionless body forces
    b = list(body_forces)
    for i in range(dim):
        if isinstance(b[i], sympy.Expr):
            for coord in b[i].free_symbols:
                b[i] = b[i].replace(coord, coord*nondim_length)
        b[i] *= nondim_length**2 /nondim_disp/nondim_mat_param
    body_forces_dimless = tuple(b)
    
    # create VectorSpace for displacement
    # check if nonhomogeneous boundary conditions are applied
    # check if only dirichlet-boundary conditions are applied
    vec_space = []
    only_dirichlet_bcs = True
    nonhomogeneous_bcs = False
    
    for i in range(dim): # nb of displacement components
        tens_space = []
        for j in range(dim): # nb of FunctionSpaces for each component
            basis = FunctionSpace(N, family='legendre', bc=bcs_dimless[i][j], domain=dom_dimless[j])
            tens_space.append(basis)
            if basis.has_nonhomogeneous_bcs:
                nonhomogeneous_bcs = True
            if not isinstance(basis, ShenDirichlet):
                only_dirichlet_bcs = False
        vec_space.append(TensorProductSpace(comm, tuple(tens_space)))
    V = VectorSpace(vec_space)
    
    # body_forces on quadrature points
    tmp = []
    for i in range(dim):
        tmp.append(FunctionSpace(N, domain=dom_dimless[i], family='legendre', bc=None))
    T = TensorProductSpace(comm, tuple(tmp))
    V_none = VectorSpace([T, T])
    body_forces_quad = Array(V_none, buffer=body_forces_dimless)
    
    # test and trial functions
    u = TrialFunction(V)
    v = TestFunction(V)
    
    # matrices
    matrices = []
    
    A = inner(mu*grad(u), grad(v))
    
    if only_dirichlet_bcs:
        B = inner((lambd + mu)*div(u), div(v))
        matrices = A + B
    else:
        B = []
        for i in range(dim):
            for j in range(dim):
                    temp = inner(mu*Dx(u[i], j), Dx(v[j], i))
                    if isinstance(temp, list):
                        B += temp
                    else:
                        B += [temp]
        C = inner(lambd*div(u), div(v))
        matrices = A + B + C
    
    # right hand side of the weak formulation
    b = inner(v, body_forces_quad)
    
    # solution
    u_hat = Function(V)
    if nonhomogeneous_bcs:
        # get boundary matrices
        bc_mats = extract_bc_matrices([matrices])
        # BlockMatrix for homogeneous part
        M = BlockMatrix(matrices)
        # BlockMatrix for inhomogeneous part
        BM = BlockMatrix(bc_mats)
        
        # inhomogeneous part of solution
        uh_hat = Function(V).set_boundary_dofs()        
        
        # additional part to be passed to the right hand side
        b_add = Function(V)
        b_add = BM.matvec(-uh_hat, b_add) # negative because added to right hand side
        
        # homogeneous part of solution
        u_hat = M.solve(b + b_add)
        
        # solution
        u_hat += uh_hat
    else:
        # BlockMatrix
        M = BlockMatrix(matrices)
        
        # solution
        u_hat = M.solve(b)
    
    
    # actual solution
    # vector space for solution
    vec_space = []
    for i in range(dim): # nb of displacement components
        tens_space = []
        for j in range(dim): # nb of FunctionSpaces for each component
            basis = FunctionSpace(N, domain=dom[j], family='legendre', bc=boundary_conditions[i][j])
            tens_space.append(basis)
        vec_space.append(TensorProductSpace(comm, tuple(tens_space)))
    V_dim = VectorSpace(vec_space)
    
    # actual solution (same coefficients, different vector space)
    u = Function(V_dim)
    for i in range(dim):
        u[i] = u_hat[i] # u has the same expansions coefficients as u_hat
    
    u *= nondim_disp    
    if measure_time:
        time_stop = time.time()
        with open('N_time.dat', 'a') as file:
            file.write(str(N) + ' ' + str(time_stop - time_start) + '\n')
        
    # compute error using analytical solution if desired
    if compute_error:
        error = check_solution_cauchy(u_hat=u_hat, material_parameters=(lambd, mu), body_forces=body_forces_dimless)
        with open('N_errorLameNavier.dat', 'a') as file:
                file.write(str(N) + ' ' + str(error) + '\n')
        if u_ana is not None:
            # transform analytical solution to dimensionless domain
            analytical_solution = list(u_ana)
            for i in range(dim):
                if isinstance(analytical_solution[i], sympy.Expr):
                    for coord in analytical_solution[i].free_symbols:
                        analytical_solution[i] = analytical_solution[i].replace(coord, coord*nondim_length)
                analytical_solution[i] /= nondim_disp
            
            # error
            error_array = Array(V, buffer=tuple(analytical_solution))
            for i in range(dim):
                error_array[i] -= project(u_hat[i], V.spaces[i]).backward()
    
            for i in range(dim):
                for j in range(len(error_array[i])):
                    for k in range(len(error_array[i][j])):
                        error_array[i][j][k] = error_array[i][j][k]**2
            error = inner((1, 1), error_array)
            
            with open('N_error_u_ana.dat', 'a') as file:
                file.write(str(N) + ' ' + str(math.sqrt(error)) + '\n')
                
    # plot displacement components if desired
    if plot_disp:
        u_sol = u.backward()
        x_, y_ = V.spaces[0].local_mesh()
        X, Y = np.meshgrid(x_, y_, indexing='ij')
        # for k in range(dim):
        #     fig = plt.figure()
        #     title = 'u' + str(k + 1)
        #     ax = fig.gca(projection='3d')
        #     ax.plot_surface(X, Y, u_sol[k], cmap=cm.coolwarm)
        #     ax.set_xlabel('x')
        #     ax.set_ylabel('y')
        #     ax.set_title(title)
        #     plt.show()
        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
        ax.set_xlim(0, dom[0][1])
        ax.set_xlabel('$x$ in $\mathrm{mm}$', fontsize=12)
        ax.set_ylabel('$y$ in $\mathrm{mm}$', fontsize=12)
        plot_grid(X, Y, ax=ax,  color="lightgrey")
        plot_grid(X + u_sol[0], Y + u_sol[1], ax=ax, color="C0")
        
        plt.show()
    return u

def solve_gradient_elasticity(N, dom, boundary_conditions, body_forces, material_parameters, \
                            nondim_disp=1.0, nondim_length=1.0, nondim_mat_param=1.0, plot_disp=False, \
                            measure_time=False, compute_error=False, u_ana=None):
    
    # assert input
    assert isinstance(N, int)
    assert isinstance(dom, tuple)
    assert isinstance(boundary_conditions, tuple)
    assert isinstance(body_forces, tuple)
    assert isinstance(nondim_disp, float)
    assert isinstance(nondim_length, float)
    assert isinstance(nondim_mat_param, float)
    assert isinstance(material_parameters, tuple)
    assert len(material_parameters) == 7
    if compute_error:
        assert u_ana is None or isinstance(u_ana, tuple)
    
    # start time measurement if desired
    if measure_time:
        time_start=time.time()
        
    # some parameters
    dim = len(dom)
    
    # dimensionless domain
    dom_dimless = []
    for i in range(dim):
        tmp = []
        for j in range(len(dom[i])):
            tmp.append(dom[i][j]/nondim_length)
        dom_dimless.append(tuple(tmp))
    dom_dimless = tuple(dom_dimless)
    
    
    # dimensionless material_parameters
    lambd = material_parameters[0]/nondim_mat_param
    mu = material_parameters[1]/nondim_mat_param
    c1 = material_parameters[2]/nondim_mat_param/nondim_length**2
    c2 = material_parameters[3]/nondim_mat_param/nondim_length**2
    c3 = material_parameters[4]/nondim_mat_param/nondim_length**2
    c4 = material_parameters[5]/nondim_mat_param/nondim_length**2
    c5 = material_parameters[6]/nondim_mat_param/nondim_length**2

    # dimensionless boundary conditions
    bcs_dimless = []
    for i in range(dim): # nb of components
        bcs_for_one_component = []
        for j in range(dim): # nb of bcs for each component
            if isinstance(boundary_conditions[i][j], tuple): # dirichlet-bcs are given as a tuple
                temp_list = []
                for k in range(len(boundary_conditions[i][j])):
                    tmp = boundary_conditions[i][j][k]
                    if isinstance(tmp, sympy.Expr): # coordinate transformation
                        for coord in tmp.free_symbols:
                            tmp = tmp.replace(coord, coord*nondim_length)
                    temp_list.append(tmp / nondim_disp)
                bcs_for_one_component.append(tuple(temp_list))
            elif isinstance(boundary_conditions[i][j], list):
                assert isinstance(boundary_conditions[i][j][0], str)
                assert isinstance(boundary_conditions[i][j][1], tuple)
                temp_list = []
                for k in range(len(boundary_conditions[i][j][1])):
                    tmp = boundary_conditions[i][j][1][k]
                    if isinstance(tmp, sympy.Expr): # coordinate transformation
                        for coord in tmp.free_symbols:
                            tmp = tmp.replace(coord, coord*nondim_length)
                    temp_list.append(tmp / nondim_disp)
                bcs_for_one_component.append([boundary_conditions[i][j][0], tuple(temp_list)]) # change tuple, leave str as it is
            else: # e.g. 'upperdirichlet', 'lowerdirichlet', None
                bcs_for_one_component.append(boundary_conditions[i][j])
        bcs_dimless.append(bcs_for_one_component)
    bcs_dimless = tuple(tuple( bc for bc in comp ) for comp in bcs_dimless) # transform lists to tuples
        
    # dimensionless body forces
    b = list(body_forces)
    for i in range(dim):
        if isinstance(b[i], sympy.Expr):
            for coord in b[i].free_symbols:
                b[i] = b[i].replace(coord, coord*nondim_length)
        b[i] *= nondim_length**2 /nondim_disp/nondim_mat_param
    body_forces_dimless = tuple(b)
    
# create VectorSpace for displacement
    # check if nonhomogeneous boundary conditions are applied
    # check if only dirichlet-boundary conditions are applied
    vec_space = []
    only_dirichlet_bcs = True
    nonhomogeneous_bcs = False
    
    for i in range(dim): # nb of displacement components
        tens_space = []
        for j in range(dim): # nb of FunctionSpaces for each component
            basis = FunctionSpace(N, family='legendre', bc=bcs_dimless[i][j], domain=dom_dimless[j])
            tens_space.append(basis)
            if basis.has_nonhomogeneous_bcs:
                nonhomogeneous_bcs = True
            if not isinstance(basis, ShenBiharmonic):
                only_dirichlet_bcs = False
        vec_space.append(TensorProductSpace(comm, tuple(tens_space)))
    V = VectorSpace(vec_space)
    
    
    # body_forces on quadrature points
    tmp = []
    for i in range(dim):
        tmp.append(FunctionSpace(N, domain=dom_dimless[i], family='legendre', bc=None))
    T = TensorProductSpace(comm, tuple(tmp))
    V_none = VectorSpace([T, T])
    body_forces_quad = Array(V_none, buffer=body_forces_dimless)
    
    # test and trial functions
    u = TrialFunction(V)
    v = TestFunction(V)
    
    # matrices
    matrices = []
    
    A = []
    if c1 != 0.0:
        A = inner(c1*div(grad(u)), div(grad(v)))
        # for i in range(dim):
        #     for j in range(dim):
        #         for k in range(dim):
        #             temp = inner(c1*Dx(Dx(u[i], j), k), Dx(Dx(v[i], j), k)) # works somehow (dirichlet-bcs)
        #             if isinstance(temp, list):
        #                 A += temp
        #             else:
        #                 A += [temp]
    if c2 != 0.0:
        B = inner(c2*div(grad(u)), grad(div(v)))
    else:
        B = []
    if c3 != 0.0:
        C = inner(c3*grad(div(u)), grad(div(v)))
    else:
        C = []
    D = []
    if c4 != 0.0:
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    temp = inner(c4*Dx(Dx(u[i], j), k), Dx(Dx(v[i], j), k))
                    if isinstance(temp, list):
                        D += temp
                    else:
                        D += [temp]
                        
    E = []
    if c5 != 0.0:
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    temp = inner(c5*Dx(Dx(u[j], i), k), Dx(Dx(v[i], j), k))
                    if isinstance(temp, list):
                        E += temp
                    else:
                        E += [temp]
    F = inner(mu*grad(u), grad(v))

    if only_dirichlet_bcs:
        G = inner((lambd + mu)*div(u), div(v))
        matrices = A + B + C + D + E + F + G
    else:
        G = []
        for i in range(dim):
            for j in range(dim):
                    temp = inner(mu*Dx(u[i], j), Dx(v[j], i))
                    if isinstance(temp, list):
                        G += temp
                    else:
                        G += [temp]
        H = inner(lambd*div(u), div(v))
        matrices = A + B + C + D + E + F + G + H
    
    # right hand side of the weak formulation
    b = inner(v, body_forces_quad)
    
    # solution
    u_hat = Function(V)
    if nonhomogeneous_bcs:
        # get boundary matrices
        bc_mats = extract_bc_matrices([matrices])
        # BlockMatrix for homogeneous part
        M = BlockMatrix(matrices)
        # BlockMatrix for inhomogeneous part
        BM = BlockMatrix(bc_mats)
        
        # inhomogeneous part of solution
        uh_hat = Function(V).set_boundary_dofs()        
        
        # additional part to be passed to the right hand side
        b_add = Function(V)
        b_add = BM.matvec(-uh_hat, b_add) # negative because added to right hand side
        
        # homogeneous part of solution
        u_hat = M.solve(b + b_add)
        
        # solution
        u_hat += uh_hat
    else:
        # BlockMatrix
        M = BlockMatrix(matrices)
        
        # solution
        u_hat = M.solve(b)
    
    # actual solution
    # vector space for solution
    vec_space = []
    for i in range(dim): # nb of displacement components
        tens_space = []
        for j in range(dim): # nb of FunctionSpaces for each component
            basis = FunctionSpace(N, domain=dom[j], family='legendre', bc=boundary_conditions[i][j])
            tens_space.append(basis)
        vec_space.append(TensorProductSpace(comm, tuple(tens_space)))
    V_dim = VectorSpace(vec_space)
    
    # actual solution (same coefficients, different vector space)
    u = Function(V_dim)
    for i in range(dim):
        u[i] = u_hat[i] # u has the same expansions coefficients as u_hat
    
    u *= nondim_disp    
    if measure_time:
        time_stop = time.time()
        with open('N_time.dat', 'a') as file:
            file.write(str(N) + ' ' + str(time_stop - time_start) + '\n')
    
     # compute error using analytical solution if desired
    if compute_error:
        error = check_solution_gradient(u_hat=u_hat, material_parameters=(lambd, mu, c1, c2, c3, c4, c5), body_forces=body_forces_dimless)
        with open('N_errorBalanceLinMom.dat', 'a') as file:
                file.write(str(N) + ' ' + str(error) + '\n')
        if u_ana is not None:
            # transform analytical solution to dimensionless domain
            analytical_solution = list(u_ana)
            for i in range(dim):
                if isinstance(analytical_solution[i], sympy.Expr):
                    for coord in analytical_solution[i].free_symbols:
                        analytical_solution[i] = analytical_solution[i].replace(coord, coord*nondim_length)
                analytical_solution[i] /= nondim_disp
            
            # error
            error_array = Array(V, buffer=tuple(analytical_solution))
            for i in range(dim):
                error_array[i] -= project(u_hat[i], V.spaces[i]).backward()
    
            for i in range(dim):
                for j in range(len(error_array[i])):
                    for k in range(len(error_array[i][j])):
                        error_array[i][j][k] = error_array[i][j][k]**2
            error = inner((1, 1), error_array)
            
            with open('N_error_u_ana.dat', 'a') as file:
                file.write(str(N) + ' ' + str(math.sqrt(error)) + '\n')
    
    # plot displacement components if desired
    if plot_disp:
        u_sol = u.backward()
        x_, y_ = V.spaces[0].local_mesh()
        X, Y = np.meshgrid(x_, y_, indexing='ij')
        # for k in range(dim):
        #     fig = plt.figure()
        #     title = 'u' + str(k + 1)
        #     ax = fig.gca(projection='3d')
        #     ax.plot_surface(X, Y, u_sol[k], cmap=cm.coolwarm)
        #     ax.set_xlabel('x')
        #     ax.set_ylabel('y')
        #     ax.set_title(title)
        #     plt.show()
        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
        ax.set_xlim(0, dom[0][1])
        ax.set_xlabel('$x$ in $\mathrm{mm}$', fontsize=12)
        ax.set_ylabel('$y$ in $\mathrm{mm}$', fontsize=12)
        plot_grid(X, Y, ax=ax,  color="lightgrey")
        plot_grid(X + u_sol[0], Y + u_sol[1], ax=ax, color="C0")
        
        plt.show()
    return u
