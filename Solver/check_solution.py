from shenfun import div, grad, Function, project, Array, inner
from math import sqrt

def check_solution_cauchy(u_hat, material_parameters, body_forces):
    '''
    Check whether the numerically computed solution fulfills the Lamé-Navier equation.
    
    Parameters
    ----------
    u_hat : shenfun Function
        Displacement in spectral space (expansion coefficients).
    material_parameters : tuple or list
        Lamé-parameters: (lambd, mu).
    body_forces : tuple
        Components of body forces as sympy expressions.

    Returns
    -------
    error
        L2 norm is computed to check whether the numerical solution fulfills the PDE.

    '''
    # assert input
    assert isinstance(u_hat, Function)
    assert isinstance(material_parameters, tuple)
    assert len(material_parameters) == 2
    for val in material_parameters:
        assert isinstance(val, float)
    
    # some parameters
    lambd = material_parameters[0]
    mu = material_parameters[1]
    
    # left hand side of lamé-navier-equation
    lhs = (lambd + mu)*grad(div(u_hat)) + mu*div(grad(u_hat))
    
    # space for volumetric forces
    V = u_hat.function_space().get_orthogonal()
    
    # evaluate volumetric forces at quadrature points (physical space)
    error_array = Array(V, buffer=body_forces)  
    
    # add left hand side of the Lamé-Navier equation
    error_array += project(lhs, V).backward()
    
    # compute integral error
    error = sqrt(inner((1, 1), error_array**2))
    
    return error
    

def check_solution_gradient(u_hat, material_parameters, body_forces):
    '''
    Check whether the numerically computed solution fulfills the 
    balance of linear momentum for linear gradient elasticity.
    
    Parameters
    ----------
    u_hat : shenfun Function
        Displacement in spectral space (expansion coefficients).
    material_parameters : tuple or list
        Lamé-parameters: (lambd, mu, c1, c2, c3, c4, c5).
    body_forces : tuple
        Components of body forces as sympy expressions.

    Returns
    -------
    error
        L2 norm is computed to check whether the numerical solution fulfills the PDE.

    '''
    # assert input
    assert isinstance(u_hat, Function)
    assert isinstance(material_parameters, tuple)
    assert len(material_parameters) == 7
    for val in material_parameters:
        assert isinstance(val, float)
    
    # some parameters
    lambd = material_parameters[0]
    mu = material_parameters[1]
    c1 = material_parameters[2]
    c2 = material_parameters[3]
    c3 = material_parameters[4]
    c4 = material_parameters[5]
    c5 = material_parameters[6]
    
    # left hand side of balance of linear momentum
    lhs = (lambd + mu)*grad(div(u_hat)) + mu*div(grad(u_hat)) - \
        (c1 + c4)*div(grad(div(grad(u_hat)))) - (c2 + c3 + c5)*grad(div(div(grad(u_hat))))
    
    # space for volumetric forces
    V = u_hat.function_space().get_orthogonal()
    
    # evaluate volumetric forces at quadrature points (physical space)
    error_array = Array(V, buffer=body_forces)  
    
    # add left hand side of the Lamé-Navier equation
    error_array += project(lhs, V).backward()
    
    # compute integral error
    error = sqrt(inner((1, 1), error_array**2))
    
    return error