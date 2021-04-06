import sympy
from shenfun import comm, FunctionSpace, TensorProductSpace, VectorSpace, Function

def get_dimensionless_values(dom, boundary_conditions, body_forces, material_parameters, \
                            nondim_disp, nondim_length, nondim_mat_param, u_ana=None):
    
    # some parameters
    dim = len(dom)
    
    # dimensionless domain  
    dom_dimless = tuple([tuple([dom[i][j]/nondim_length for j in range(2)]) for i in  range(dim)])
    
    # dimensionless material_parameters
    lambd = material_parameters[0]/nondim_mat_param
    mu = material_parameters[1]/nondim_mat_param
    
    # assign bc type to each boundary condition
    def assign_bc_type(bc):
        if bc == None:
            return 'ZERO_TRACTION'
        elif isinstance(bc, str):
            # homogeneous mixed bc, e.g. 'upperdirichlet', 'lowerdirichlet'
            return 'MIXED_HOM'
        elif isinstance(bc, tuple):
            # dirichlet-bcs are given as tuple
            return 'DIRICHLET'
        elif isinstance(bc, list):
            # mixed inhomogeneous bcs are given as list
            return 'MIXED_INHOM'
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
                    if isinstance(component, sympy.Expr): # coordinate transformation
                        for coord in component.free_symbols:
                            component = component.replace(
                                coord, coord*nondim_length
                            )
                    bc_dimless.append(component / nondim_disp)
                # append to bcs_comp_dimless
                bcs_comp_dimless.append(tuple(bc_dimless))
            
            elif bc_type in ('ZERO_TRACTION', 'MIXED_HOM'):
                # nothing to change here
                bcs_comp_dimless.append(bc)
                
            elif bc_type == 'MIXED_INHOM':
                assert isinstance(bc[0], str)
                assert isinstance(bc[1], tuple) # inhomogeneous bcs are given in tuple
                bc_dimless = [] 
                for component in bc[1]:
                    if isinstance(component, sympy.Expr): # coordinate transformation
                        for coord in component.free_symbols:
                            component = component.replace(
                                coord, coord*nondim_length
                                )
                    # append dimensionless components to list
                    bc_dimless.append(component / nondim_disp)
                # append dimensionless bcs to list
                bcs_comp_dimless.append([bc[0], tuple(bc_dimless)]) # change tuple, leave str as it is
                
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
            ua[i] /= nondim_length
        
        u_ana_dimless= tuple(ua)
        
        # return u_ana_dimless as well
        return dom_dimless, boundary_conditions_dimless, body_forces_dimless, (lambd, mu), u_ana_dimless
        
    else: # no analytical solution
        return dom_dimless, boundary_conditions_dimless, body_forces_dimless, (lambd, mu)


def get_dimensionful_values(u_hat_dimless, boundary_conditions, nondim_disp, nondim_length):
    # get size of discretization
    N = u_hat_dimless.function_space().spaces[0].bases[0].N
    
    # some parameters
    dim = len(u_hat_dimless)
    
    # get dimensionless domain from solution
    dom_dimless = tuple([u_hat_dimless.function_space().spaces[i].bases[i].domain for i in range(dim)])

    # dimensionful domain  
    dom = tuple([tuple([dom_dimless[i][j]*nondim_length for j in range(2)]) for i in range(dim)])

    # vector space for solution
    vec_space = []
    for i in range(dim): # nb of displacement components
        tens_space = []
        for j in range(dim): # nb of FunctionSpaces for each component
            basis = FunctionSpace(N, domain=dom[j], family='legendre', bc=boundary_conditions[i][j])
            tens_space.append(basis)
        vec_space.append(TensorProductSpace(comm, tuple(tens_space)))
    
    V = VectorSpace(vec_space)
        
    # actual solution (same coefficients, different vector space)
    u_hat = Function(V)
        
    for i in range(dim):
        u_hat[i] = u_hat_dimless[i] # u has the same expansions coefficients as u_hat
            
    return u_hat