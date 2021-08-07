from shenfun import VectorSpace, TensorProductSpace, Array, TrialFunction, \
    TestFunction, inner, grad, div, Dx, extract_bc_matrices, BlockMatrix, \
    Function, FunctionSpace, project, comm
from shenfun.legendre.bases import ShenDirichlet, ShenBiharmonic
from .check_solution import check_solution_cauchy, check_solution_gradient
from math import sqrt


def solve_cauchy_elasticity(N, dom, boundary_conditions, body_forces,
                            material_parameters, compute_error=False,
                            u_ana=None):
    '''
    Solve problems in linear Cauchy elasticity using shenfun.

    Parameters
    ----------
    N : int
        Size of discretization.
    dom : tuple
        Reference domain.
    boundary_conditions : tuple
        Boundary conditions of boundary value problem.
    body_forces : tuple
        Volumetric forces as sympy expression.
    material_parameters : tuple
        Lamé-parameters: (lambd, mu).
    measure_time : bool, optional
        If True, computation time is being measured. The default is False.
    compute_error : bool, optional
        If True, two different errors are being computed. The default is False.
    u_ana : tuple, optional
        Analytical solution as sympy Expr. Needs to be specified only if
        an analytical solution is known. The default is None.

    Returns
    -------
    u_hat : shenfun Function
        Solution of boundary value problem in spectral space.

    '''
    # assert input
    assert isinstance(N, (tuple, list))
    assert isinstance(dom, tuple)
    assert isinstance(boundary_conditions, tuple)
    assert isinstance(body_forces, tuple)
    assert isinstance(material_parameters, tuple)
    assert len(material_parameters) == 2
    if compute_error:
        assert u_ana is None or isinstance(u_ana, tuple)

    # some parameters
    dim = len(dom)

    #  material_parameters
    lambd = material_parameters[0]
    mu = material_parameters[1]

    # create VectorSpace for displacement
    # check if nonhomogeneous boundary conditions are being applied
    # check if only dirichlet-boundary conditions are being applied
    vec_space = []
    only_dirichlet_bcs = True
    nonhomogeneous_bcs = False

    for i in range(dim):  # nb of displacement components
        tens_space = []
        for j in range(dim):  # nb of FunctionSpaces for each component
            basis = FunctionSpace(N[j], family='legendre',
                                  bc=boundary_conditions[i][j], domain=dom[j])
            tens_space.append(basis)
            if basis.has_nonhomogeneous_bcs:
                nonhomogeneous_bcs = True
            if not isinstance(basis, ShenDirichlet):
                only_dirichlet_bcs = False
        vec_space.append(TensorProductSpace(comm, tuple(tens_space)))
    V = VectorSpace(vec_space)

    # body_forces on quadrature points
    V_none = V.get_orthogonal()
    body_forces_quad = Array(V_none, buffer=body_forces)

    # test and trial functions
    u = TrialFunction(V)
    v = TestFunction(V)

    # matrices
    A = inner(mu*grad(u), grad(v))

    if only_dirichlet_bcs:
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
        # negative because added to right hand side
        b_add = BM.matvec(-uh_hat, b_add)

        # homogeneous part of solution
        u_hat = M.solve(b + b_add)

        # solution
        u_hat += uh_hat
    else:
        # BlockMatrix
        M = BlockMatrix(matrices)

        # solution
        u_hat = M.solve(b)

    # compute error using analytical solution if desired
    if compute_error:
        error = check_solution_cauchy(u_hat=u_hat,
                                      material_parameters=(lambd, mu),
                                      body_forces=body_forces)
        with open('N_errorLameNavier.dat', 'a') as file:
            file.write(str(N) + ' ' + str(error) + '\n')

        if u_ana is not None:
            # evaluate u_ana at quadrature points
            error_array = Array(V, buffer=u_ana)
            # subtract numerical solution
            error_array -= project(u_hat, V).backward()
            # compute integral error
            error = sqrt(inner((1, 1), error_array**2))

            with open('N_error_u_ana.dat', 'a') as file:
                file.write(str(N) + ' ' + str(error) + '\n')

    return u_hat


def solve_gradient_elasticity(N, dom, boundary_conditions, body_forces,
                              material_parameters, compute_error=False,
                              u_ana=None):
    '''
    Solve problems in linear second order gradient elasticity using shenfun.

    Parameters
    ----------
    N : int
        Size of discretization.
    dom : tuple
        Reference domain.
    boundary_conditions : tuple
        Boundary conditions of boundary value problem.
    body_forces : tuple
        Volumetric forces as sympy expression.
    material_parameters : tuple
        Lamé-parameters: (lambd, mu, c1, c2, c3, c4, c5).
    measure_time : bool, optional
        If True, computation time is being measured. The default is False.
    compute_error : bool, optional
        If True, two different errors are being computed. The default is False.
    u_ana : tuple, optional
        Analytical solution as sympy Expr. Needs to be specified only if
        an analytical solution is known. The default is None.

    Returns
    -------
    u_hat : shenfun Function
        Solution of boundary value problem in spectral space.
    '''

    # assert input
    assert isinstance(N, (tuple, list))
    assert isinstance(dom, tuple)
    assert isinstance(boundary_conditions, tuple)
    assert isinstance(body_forces, tuple)
    assert isinstance(material_parameters, tuple)
    assert len(material_parameters) == 7
    if compute_error:
        assert u_ana is None or isinstance(u_ana, tuple)

    # some parameters
    dim = len(dom)

    # material_parameters
    lambd = material_parameters[0]
    mu = material_parameters[1]
    c1 = material_parameters[2]
    c2 = material_parameters[3]
    c3 = material_parameters[4]
    c4 = material_parameters[5]
    c5 = material_parameters[6]

    # create VectorSpace for displacement
    # check if nonhomogeneous boundary conditions are applied
    # check if only dirichlet-boundary conditions are applied
    vec_space = []
    only_dirichlet_bcs = True
    nonhomogeneous_bcs = False

    for i in range(dim):  # nb of displacement components
        tens_space = []
        for j in range(dim):  # nb of FunctionSpaces for each component
            basis = FunctionSpace(N[j], family='legendre',
                                  bc=boundary_conditions[i][j], domain=dom[j])
            tens_space.append(basis)
            if basis.has_nonhomogeneous_bcs:
                nonhomogeneous_bcs = True
            if not isinstance(basis, ShenBiharmonic):
                only_dirichlet_bcs = False
        vec_space.append(TensorProductSpace(comm, tuple(tens_space)))
    V = VectorSpace(vec_space)

    # body_forces on quadrature points
    V_none = V.get_orthogonal()
    body_forces_quad = Array(V_none, buffer=body_forces)

    # test and trial functions
    u = TrialFunction(V)
    v = TestFunction(V)

    # matrices
    matrices = []

    if c1 != 0.0:
        A = inner(c1*div(grad(u)), div(grad(v)))
    else:
        A = []
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
        # negative because added to right hand side
        b_add = BM.matvec(-uh_hat, b_add)

        # homogeneous part of solution
        u_hat = M.solve(b + b_add)

        # solution
        u_hat += uh_hat
    else:
        # BlockMatrix
        M = BlockMatrix(matrices)

        # solution
        u_hat = M.solve(b)

    # compute error using analytical solution if desired
    if compute_error:
        error = check_solution_gradient(
                u_hat=u_hat,
                material_parameters=(lambd, mu, c1, c2, c3, c4, c5),
                body_forces=body_forces)

        with open('N_errorBalanceLinMom.dat', 'a') as file:
            file.write(str(N[0]) + ' ' + str(error) + '\n')

        if u_ana is not None:
            # evaluate u_ana at quadrature points
            error_array = Array(V, buffer=u_ana)
            # subtract numerical solution
            error_array -= project(u_hat, V).backward()
            # compute integral error
            error = sqrt(inner((1, 1), error_array**2))
            # scale by magnitude of solution
            scale = sqrt(inner((1, 1), u_hat.backward()**2))

            with open('N_error_u_ana.dat', 'a') as file:
                file.write(str(N[0]) + ' ' + str(error/scale) + '\n')

    return u_hat
