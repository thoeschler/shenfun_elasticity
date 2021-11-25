from sympy import symbols
import sympy as sp
import numpy as np


def body_forces_cauchy(u, material_parameters):
    '''
    Compute body forces via given displacement field
    for linear Cauchy elasticity.

    Parameters
    ----------
    u : tuple
        Displacement field as sympy expression.
    material_parameters : tuple or list
        Lame-parameters: (lambd, mu).

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
    lmbda, mu = material_parameters

    x, y, z = symbols("x,y,z")
    coord = [x, y, z]

    # div(u)
    Divergence = 0.
    for i in range(dim):
        Divergence += u[i].diff(coord[i])

    # grad(div(u))
    GradDiv = np.empty(dim, dtype=sp.Expr)
    for i in range(dim):
        GradDiv[i] = Divergence.diff(coord[i])

    # laplace
    Laplace = np.zeros(dim, dtype=sp.Expr)
    for i in range(dim):
        for j in range(dim):
            Laplace[i] += u[i].diff(coord[j], 2)

    # compute body forces
    body_forces = - (lmbda + mu) * GradDiv - mu * Laplace

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
    lmbda, mu, c1, c2, c3, c4, c5 = material_parameters
    x, y, z = symbols("x,y,z")
    coord = [x, y, z]

    # div(u)
    Divergence = 0.
    for i in range(dim):
        Divergence += u[i].diff(coord[i])

    # div(grad(u))
    Laplace = np.zeros(dim, dtype=sp.Expr)
    for i in range(dim):
        for j in range(dim):
            Laplace[i] += u[i].diff(coord[j], 2)

    # grad(div(u))
    GradDiv = np.empty(dim, dtype=sp.Expr)
    for i in range(dim):
        GradDiv[i] = Divergence.diff(coord[i])

    # DoubleLaplace
    DoubleLaplace = np.zeros(dim, dtype=sp.Expr)
    for i in range(dim):
        for j in range(dim):
            DoubleLaplace[i] += Laplace[i].diff(coord[j], 2)

    # div(div(grad(u)))
    DivDivGrad = 0.
    for i in range(dim):
        DivDivGrad += Laplace[i].diff(coord[i])

    # grad(div(div(grad(u)))) / grad(div(laplace))
    GradDivDivGrad = np.empty(dim, dtype=sp.Expr)
    for i in range(dim):
        GradDivDivGrad[i] = DivDivGrad.diff(coord[i])

    # body forces
    body_forces = (c1 + c4) * DoubleLaplace + (c2 + c3 + c5) * GradDivDivGrad \
        - (lmbda + mu) * GradDiv - mu * Laplace

    return body_forces
