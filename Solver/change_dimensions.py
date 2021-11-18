import sympy
import numpy as np
from shenfun import comm, FunctionSpace, TensorProductSpace, VectorSpace, Function

def get_dimensionless_values(dom, boundary_conditions, body_forces, material_parameters, \
                            nondim_disp, nondim_length, nondim_mat_param, u_ana=None):
    '''
    Get dimensionless domain, boundary conditions, body forces and 
    material parameters for boundary value problems in Cauchy and 
    linear second order gradient elasticity.

    Parameters
    ----------
    dom : tuple
        Physical domain.
    boundary_conditions : tuple
        Boundary conditions of boundary value problem.
    body_forces : tuple
        Volumetric forces as sympy expression.
    material_parameters : tuple or list
        - for Cauchy elasticity: Lamé-parameters (lambda, mu)
        - for gradient elasticity: (lambda, mu, c1, c2, c3, c4, c5).
    nondim_disp : float
        Reference displacement value used to get dimensionless
        volumetric forces.
    nondim_length : float
        Reference length used to get dimensionless domain, volumetric 
        forces and material parameters (for gradient elasticity).
    nondim_mat_param : float
        Reference material parameter in MPa, e.g. one of the Lamé-
        parameters.
    u_ana : tuple, optional
        Analytical solution as sympy expression. Needs to be specified
        only if analytical solution is known. The default is None.

    Raises
    ------
    NotImplementedError
        Length of tuple or list containing material parameters has to
        be 2 (Cauchy elasticity) or 7 (gradient elasticity).

    Returns
    -------
    dom_dimless, boundary_conditions_dimless, body_forces_dimless, \
            material_parameters_dimless, u_ana_dimless
        Dimensionless domain, boundary conditions, volumetric forces
        and material parameters. Dimensionless analytical solution is
        returned only if u_ana is not None.
    '''
    # some parameters
    dim = len(dom)
    
    # dimensionless domain  
    dom_dimless = tuple([tuple([dom[i][j]/nondim_length for j in range(2)]) for i in  range(dim)])
    
    # dimensionless material_parameters
    if len(material_parameters) == 2: # cauchy elasticity
        lambd = material_parameters[0]/nondim_mat_param
        mu = material_parameters[1]/nondim_mat_param
        # wrap material parameters in a tuple
        material_parameters_dimless = (lambd, mu)
    elif len(material_parameters) == 7: # gradient elasticity
        lambd, mu = np.array(material_parameters)[[0, 1]] / nondim_mat_param
        c1, c2, c3, c4, c5 = np.array(material_parameters)[2:] / nondim_mat_param / nondim_length**2
        # wrap material parameters in a tuple
        material_parameters_dimless = lambd, mu, c1, c2, c3, c4, c5
    else:
        raise NotImplementedError()
    
    # assign bc type to each boundary condition
    def assign_bc_type(bc):
        if bc is None:
            return 'ZERO_TRACTION'
        elif isinstance(bc, str):
            # homogeneous mixed bc, e.g. 'upperdirichlet', 'lowerdirichlet'
            return 'MIXED_HOM'
        elif isinstance(bc, tuple):
            # dirichlet-bcs are given as tuple
            return 'DIRICHLET'
        elif isinstance(bc, dict):
            # all bcs can be defined via a dictionary
            return 'ARBITRARY'
        else:
            raise NotImplementedError() 


    # dimensionless boundary conditions
    boundary_conditions_dimless = []

    for bcs_comp in boundary_conditions: # bcs for each component
        # dimensionless bcs for each component
        bcs_comp_dimless = []           
        for bc in bcs_comp: # bc in one direction for one component
            # assign a bc type based on the data type
            bc_type = assign_bc_type(bc)
            
            # get dimensionless bc
            if bc_type == 'DIRICHLET':
                # dimensionless bc
                bc_dimless = []
                for component in bc: # coordinate transformation for each component
                    if component is not None:
                        if isinstance(component, sympy.Expr): # coordinate transformation
                            for coord in component.free_symbols:
                                component = component.replace(
                                    coord, coord*nondim_length
                            )
                        bc_dimless.append(component / nondim_disp)
                    else:
                        bc_dimless.append(component)
                # append to bcs_comp_dimless
                bcs_comp_dimless.append(tuple(bc_dimless))
            
            elif bc_type in ('ZERO_TRACTION', 'MIXED_HOM'):
                # nothing to change here
                bcs_comp_dimless.append(bc)
                
            elif bc_type == 'ARBITRARY':
                bc_dimless = dict()
                
                assert isinstance(bc, dict)
                for side, condition in bc.items():
                    assert side in ('left', 'right')
                    bc_dimless[side] = [] # initialize value with empty list
                    
                    for kind, val in condition:
                        if isinstance(val, sympy.Expr):
                            for coord in val.free_symbols:
                                val = val.replace(
                                        coord, coord*nondim_length
                                        )
                        if kind == 'D': # dirichlet bc
                            bc_dimless[side].append(
                                    ('D', val/nondim_disp)
                                )
                        elif kind == 'N': # neumann bc
                            bc_dimless[side].append(
                                    ('N', val / nondim_disp * nondim_length)
                                )
                        elif kind == 'N2': # second deivative
                            bc_dimless[side].append(
                                    ('N2', val / nondim_disp * nondim_length**2)
                                )
                        elif kind == 'N3': # third derivative
                            bc_dimless[side].append(
                                    ('N3', val / nondim_disp * nondim_length**3)
                                )         
                # append dimensionless bcs to list
                bcs_comp_dimless.append(bc_dimless) # change tuple, leave str as it is
                
            else:
                raise NotImplementedError()
        
        # append bcs for each component to dimensionless bcs
        boundary_conditions_dimless.append(tuple(bcs_comp_dimless))
        
    # convert bcs to tuple
    boundary_conditions_dimless = tuple(boundary_conditions_dimless)
    
    # dimensionless body forces
    b = list(body_forces)
    for i in range(dim):
        if isinstance(b[i], sympy.Expr): # coordinate transformation
            for coord in b[i].free_symbols:
                b[i] = b[i].replace(coord, coord*nondim_length)
        b[i] *= nondim_length**2 /nondim_disp/nondim_mat_param
    body_forces_dimless = tuple(b)
    
    if u_ana is not None:
        # transform analytical solution
        ua = list(u_ana)
        for i in range(dim):
            if isinstance(ua[i], sympy.Expr): # coordinate transformation
                for coord in ua[i].free_symbols:
                    ua[i] = ua[i].replace(coord, coord*nondim_length)
            ua[i] /= nondim_disp
        
        u_ana_dimless= tuple(ua)
        
        # return u_ana_dimless as well
        return dom_dimless, boundary_conditions_dimless, body_forces_dimless, \
            material_parameters_dimless, u_ana_dimless
    else: # no analytical solution
        return dom_dimless, boundary_conditions_dimless, body_forces_dimless, \
            material_parameters_dimless


def get_dimensionful_values(u_hat_dimless, boundary_conditions, nondim_disp, nondim_length):
    '''
    Get dimensionful values after solving the dimensionless
    boundary value problem.

    Parameters
    ----------
    u_hat_dimless : shenfun Function
        Displacement field in spectral space (expansion coefficients).
    boundary_conditions : tuple
        Boundary conditions of dimensionful problem.
    nondim_disp : float
        Reference displacement value used to get dimensionful displacement.
    nondim_length : float
        Reference length used to get physical domain.

    Returns
    -------
    u_hat : shenfun Function
        Dimensionful displacement field in spectral space.

    '''
    
    # some parameters (dimension)
    dim = len(u_hat_dimless)
    
        # get size of discretization
    N = tuple(u_hat_dimless.function_space().spaces[0].bases[i].N for i in range(dim))
    
    # get dimensionless domain from solution
    dom_dimless = tuple([u_hat_dimless.function_space().spaces[i].bases[i].domain for i in range(dim)])

    # dimensionful domain  
    dom = tuple([tuple([dom_dimless[i][j]*nondim_length for j in range(2)]) for i in range(dim)])

    # vector space for solution
    vec_space = []
    for i in range(dim): # nb of displacement components
        tens_space = []
        for j in range(dim): # nb of FunctionSpaces for each component
            basis = FunctionSpace(N[j], domain=dom[j], family='legendre', bc=boundary_conditions[i][j])
            tens_space.append(basis)
        vec_space.append(TensorProductSpace(comm, tuple(tens_space)))
    
    V = VectorSpace(vec_space)
        
    # actual solution (same coefficients, different vector space)
    u_hat = Function(V)
        
    for i in range(dim):
        u_hat[i] = nondim_disp*u_hat_dimless[i] # u has the same expansions coefficients as u_hat
            
    return u_hat