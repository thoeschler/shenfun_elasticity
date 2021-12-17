import numpy as np
import sympy as sp
import shenfun as sf


def compute_numerical_error(u_ana, u_hat):
    assert isinstance(u_hat, sf.Function)

    V = u_hat.function_space()
    # evaluate u_ana at quadrature points
    error_array = sf.Array(V, buffer=u_ana)
    # subtract numerical solution
    error_array -= u_hat.backward()
    # compute integral error
    error = np.sqrt(sf.inner((1, 1), error_array ** 2))

    return error


def get_dimensionless_parameters(domain, boundary_conditions, body_forces,
                                 material_parameters, u_ref,
                                 l_ref, mat_param_ref, u_ana=None):
    domain_dl = get_dimensionless_domain(domain, l_ref)
    boundary_conditions_dl = get_dimensionless_boundary_conditions(
        boundary_conditions, l_ref, u_ref)
    body_forces_dl = get_dimensionless_body_forces(body_forces, l_ref, u_ref,
                                                   mat_param_ref)
    material_parameters_dl = get_dimensionless_material_parameters(
        material_parameters, mat_param_ref, l_ref)

    if u_ana is not None:
        u_ana_dl = get_dimensionless_displacement(u_ana, l_ref, u_ref)
        return domain_dl, boundary_conditions_dl, body_forces_dl, \
            material_parameters_dl, u_ana_dl

    else:
        return domain_dl, boundary_conditions_dl, body_forces_dl, \
            material_parameters_dl


def get_dimensional_displacement(u_hat_dimless, boundary_conditions,
                                 u_ref, l_ref):
    dim = len(u_hat_dimless)
    N = [u_hat_dimless.function_space().spaces[0].bases[i].N
         for i in range(dim)]

    domain_dimless = [u_hat_dimless.function_space().spaces[i].bases[i].domain
                      for i in range(dim)]
    domain = [[domain_dimless[i][j] * l_ref for j in range(2)]
              for i in range(dim)]

    # vector space for solution
    vec_space = []
    for i in range(dim):  # nb of displacement components
        tens_space = []
        for j in range(dim):  # nb of FunctionSpaces for each component
            basis = sf.FunctionSpace(N[j], domain=domain[j], family='legendre',
                                     bc=boundary_conditions[i][j])
            tens_space.append(basis)
        vec_space.append(sf.TensorProductSpace(sf.comm, tuple(tens_space)))

    V = sf.VectorSpace(vec_space)

    # actual solution (same coefficients, different vector space)
    u_hat = sf.Function(V)

    for i in range(dim):
        # u has the same expansions coefficients as u_dimless
        u_hat[i] = u_ref * u_hat_dimless[i]

    return u_hat


def get_dimensionless_domain(domain, l_ref):
    return tuple(np.array(domain) / l_ref)


def get_dimensionless_material_parameters(material_parameters,
                                          mat_param_ref, l_ref):
    if len(material_parameters) == 2:  # cauchy elasticity
        return tuple(np.array(material_parameters) / mat_param_ref)
    elif len(material_parameters) == 7:  # gradient elasticity
        lame_parameters = np.array(material_parameters)[slice(2)] / mat_param_ref
        gradient_parameters = np.array(material_parameters)[2:] \
            / mat_param_ref / l_ref ** 2
        # wrap material parameters in a tuple
        return tuple(np.hstack((lame_parameters, gradient_parameters)))


def get_dimensionless_boundary_conditions(boundary_conditions, l_ref,
                                          u_ref):
    # assign bc type to each boundary condition
    def assign_bc_type(bc):
        if bc is None:
            return 'ZERO_TRACTION'
        elif isinstance(bc, tuple):  # dirichlet-bcs are given as tuple
            return 'DIRICHLET'
        elif isinstance(bc, dict):  # all bcs can be defined via a dictionary
            return 'ARBITRARY'
        else:
            raise NotImplementedError()

    boundary_conditions_dimless = []

    for bcs_comp in boundary_conditions:
        bcs_comp_dimless = []
        for bc in bcs_comp:
            bc_type = assign_bc_type(bc)

            if bc_type == 'DIRICHLET':
                bc_dimless = []
                for component in bc:
                    if component is not None:  # coordinate transformation
                        tmp = component
                        if isinstance(component, sp.Expr):
                            for coord in component.free_symbols:
                                tmp = tmp.replace(
                                    coord, coord * l_ref)
                        bc_dimless.append(tmp / u_ref)
                    else:
                        bc_dimless.append(component)
                bcs_comp_dimless.append(tuple(bc_dimless))

            elif bc_type == 'ZERO_TRACTION':
                bcs_comp_dimless.append(bc)

            elif bc_type == 'ARBITRARY':
                bc_dimless = dict()

                for side, condition in bc.items():
                    assert side in ('left', 'right')
                    bc_dimless[side] = []  # initialize value with empty list

                    for kind, val in condition:
                        if isinstance(val, sp.Expr):
                            for coord in val.free_symbols:
                                val = val.replace(coord, coord * l_ref)
                        if kind == 'D':  # dirichlet bc
                            bc_dimless[side].append(('D', val / u_ref))
                        elif kind == 'N':  # neumann bc
                            bc_dimless[side].append(('N', val / u_ref * l_ref))
                        elif kind == 'N2':  # second deivative
                            bc_dimless[side].append(('N2', val / u_ref * l_ref ** 2))
                        elif kind == 'N3':  # third derivative
                            bc_dimless[side].append(('N3', val / u_ref * l_ref ** 3))
                        else:
                            raise NotImplementedError()

                bcs_comp_dimless.append(bc_dimless)

            else:
                raise NotImplementedError()

        # append bcs for each component to dimensionless bcs
        boundary_conditions_dimless.append(tuple(bcs_comp_dimless))

    return tuple(boundary_conditions_dimless)


def get_dimensionless_body_forces(body_forces, l_ref, u_ref,
                                  mat_param_ref):
    if body_forces is None:
        return None
    else:
        dim = len(body_forces)
        b = list(body_forces)
        for i in range(dim):
            if isinstance(b[i], sp.Expr):  # coordinate transformation
                for coord in b[i].free_symbols:
                    b[i] = b[i].replace(coord, coord * l_ref)
            b[i] *= l_ref ** 2 / u_ref / mat_param_ref

        return tuple(b)


def get_dimensionless_displacement(displacement, l_ref, u_ref):
    dim = len(displacement)
    ua = list(displacement)
    for i in range(dim):
        if isinstance(ua[i], sp.Expr):  # coordinate transformation
            for coord in ua[i].free_symbols:
                ua[i] = ua[i].replace(coord, coord * l_ref)
        ua[i] /= u_ref

    return tuple(ua)
