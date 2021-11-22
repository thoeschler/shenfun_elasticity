from sympy import symbols


def body_forces_cauchy(u, material_parameters):
    '''
    Compute body forces via given displacement field
    for linear Cauchy elasticity.

    Parameters
    ----------
    u : tuple
        Displacement field as sympy expression.
    material_parameters : tuple or list
        Lamé-parameters: (lambd, mu).

    Returns
    -------
    body_forces: tuple(sympy.Expr)
        Volumetric forces as sympy expression.

    '''
    # assert inputs
    assert isinstance(u, tuple)
    assert isinstance(material_parameters, tuple)
    assert len(material_parameters) == 2

    # some parameters
    dim = len(u)
    lambd, mu = material_parameters

    x, y, z = symbols("x,y,z")
    coord = [x, y, z]

    # div(u)
    Divergence = 0.
    for i in range(dim):
        Divergence += u[i].diff(coord[i])

    # grad(div(u))
    GradDiv = [0. for _ in range(dim)]
    for i in range(dim):
        GradDiv[i] = Divergence.diff(coord[i])

    # laplace
    Laplace = [0. for _ in range(dim)]

    for i in range(dim):
        for j in range(dim):
            Laplace[i] += u[i].diff(coord[j], 2)

    # compute body forces
    body_forces = - (lambd + mu)*GradDiv - mu*Laplace

    return body_forces


def body_forces_gradient(u, material_parameters):
    '''
    Compute body forces via given displacement field
    for linear second order gradient elasticity.

    Parameters
    ----------
    u : tuple
        Displacement field as sympy expression.
    material_parameters : tuple or list
        Lamé-parameters: (lambd, mu, c1, c2, c3, c4, c5).

    Returns
    -------
    body_forces : tuple(sympy.Expr)
        Volumetric forces as sympy expression.

    '''
    # assert input
    assert isinstance(u, tuple)
    assert isinstance(material_parameters, tuple)
    assert len(material_parameters) == 7

    # some parameters
    dim = len(u)
    lambd, mu, c1, c2, c3, c4, c5 = material_parameters
    x, y, z = symbols("x,y,z")
    coord = [x, y, z]

    # div(u)
    Divergence = 0.
    for i in range(dim):
        Divergence += u[i].diff(coord[i])

    # div(grad(u))
    Laplace = [0. for _ in range(dim)]
    for i in range(dim):
        for j in range(dim):
            Laplace[i] += u[i].diff(coord[j], 2)

    # grad(div(u))
    GradDiv = [0. for _ in range(dim)]
    for i in range(dim):
        GradDiv[i] = Divergence.diff(coord[i])

    # DoubleLaplace
    DoubleLaplace = [0. for _ in range(dim)]
    for i in range(dim):
        for j in range(dim):
            DoubleLaplace[i] += Laplace[i].diff(coord[j], 2)

    # div(div(grad(u)))
    DivDivGrad = 0.
    for i in range(dim):
        DivDivGrad += Laplace[i].diff(coord[i])

    # grad(div(div(grad(u)))) / grad(div(laplace))
    GradDivDivGrad = [0. for _ in range(dim)]
    for i in range(dim):
        GradDivDivGrad[i] = DivDivGrad.diff(coord[i])

    # body forces
    body_forces = (c1 + c4)*DoubleLaplace + (c2 + c3 + c5)*GradDivDivGrad \
        - (lambd + mu)*GradDiv - mu*Laplace

    return body_forces
