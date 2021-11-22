from shenfun import Function, project, Dx, VectorSpace
import numpy as np


def cauchy_stresses(material_parameters, u_hat):
    '''
    Compute components of Cauchy stress tensor.

    Parameters
    ----------
    material_parameters : tuple or list
        Lamé-parameters: (lambd, mu).
    u_hat : shenfun Function
        Displacement field in spectral space.

    Returns
    -------
    T : list
        Cauchy stress Tensor in spectral space.

    '''
    # input assertion
    assert isinstance(material_parameters, tuple)
    assert len(material_parameters) == 2
    assert isinstance(u_hat, Function)

    # some parameters
    T_none = u_hat[0].function_space().get_orthogonal()
    dim = len(T_none.bases)

    lambd, mu = material_parameters

    # displacement gradient
    H = np.empty(shape=(dim, dim))
    for i in range(dim):
        for j in range(dim):
            H[i, j] = project(Dx(u_hat[i], j), T_none)

    # linear strain tensor
    E = 0.5 * (H + H.T)

    # trace of linear strain tensor
    trE = np.trace(E)

    # Cauchy stress tensor
    T = 2.0 * mu * E + lambd * trE * np.identity(dim)

    return T


def hyper_stresses(material_parameters, u_hat):
    '''
    Compute components of hyper stress tensor.

    Parameters
    ----------
    material_parameters : tuple or list
        (c1, c2, c3, c4, c5).
    u_hat : shenfun Function
        Displacement field in spectral space.

    Returns
    -------
    T : list
        Hyper stress Tensor in spectral space.

    '''
    # input assertion
    assert isinstance(material_parameters, tuple)
    assert len(material_parameters) == 5
    assert isinstance(u_hat, Function)

    # some parameters
    T_none = u_hat[0].function_space()
    dim = len(T_none.bases)
    c1, c2, c3, c4, c5 = material_parameters

    # Divergence
    Div = np.sum([project(Dx(u_hat[i], i), T_none)] for i in range(dim))

    # Laplace
    Laplace = np.empty(dim)
    for i in range(dim):
        for j in range(dim):
            Laplace[i] += project(Dx(u_hat[i], j, 2), T_none)

    # grad(div(u))
    GradDiv = np.empty(dim)
    for i in range(dim):
        GradDiv[i] += project(Dx(Div, i), T_none)

    # hyper stresses
    T = [ [ [0. for _ in range(dim)] for _ in range(dim)] for _ in range(dim)]
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                if i==j:
                    if c2 != 0.:
                        T[i][j][k] += 0.5*c2*Laplace[k]
                    if c3 != 0.:
                        T[i][j][k] += 0.5*c3*GradDiv[k]
                if i==k:
                    if c2 != 0.:
                        T[i][j][k] += 0.5*c2*Laplace[j]
                    if c3 != 0.:
                        T[i][j][k] += 0.5*c3*GradDiv[j]
                if j==k:
                    if c1 != 0.:
                        T[i][j][k] += c1*Laplace[i]
                if c4 != 0.:
                    T[i][j][k] += project(c4*Dx(Dx(u_hat[i], j), k), T_none)
                if c5 != 0.:
                    T[i][j][k] += project(0.5*c5*Dx(Dx(u_hat[j], i), k), T_none) + project(0.5*c5*Dx(Dx(u_hat[k], i), j), T_none)

    return T

def traction_vector_gradient(cauchy_stresses, hyper_stresses, normal_vector):
    '''
    Compute traction vector for linear second order gradient elasticity.

    Parameters
    ----------
    cauchy_stresses : list
        Components of Cauchy stress tensor in spectral space.
    hyper_stresses : list
        Components of hyper stress tensor in spectral space.
    normal_vector : tuple
        Normal vector used to compute the traction.

    Returns
    -------
    t : list
        Traction vector in spectral space.

    '''
    # some paramaters
    dim = len(normal_vector)
    T2 = cauchy_stresses
    T3 = hyper_stresses
    n = normal_vector
    T_none = T2[0][0].function_space().get_orthogonal()
    tol = 1e-10
    assert normal_vector[0] ** 2 +  normal_vector[1] ** 2 - 1 < tol
    
    # compute traction vector
    t = [0. for _ in range(dim)]
    # check if there are nonzero values in hyper stresses
    has_nonzero_hyper_stresses = any([val.any() for row in T3 for comp in row for val in np.array(comp)])
    if has_nonzero_hyper_stresses:
        # div(T3)
        divT3 = [[0. for _ in range(dim)] for _ in range(dim)]
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    # for some reason divT3[i][j] += project(Dx(T3[i][j][k], k), T_none) raises an error --> transform back and forth once
                    work = project(T3[i][j][k].copy().backward(), T_none)
                    divT3[i][j] += project(Dx(work, k), T_none)
                    
        # divn(T3), divt(T3)
        divnT3 = [[0. for _ in range(dim)] for _ in range(dim)]
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    for l in range(dim):
                        work = project(T3[i][j][k].copy().backward(), T_none)
                        divnT3[i][j] += project(Dx(work, l), T_none) * n[k] * n[l]
                        
        divtT3 = np.array(divT3.copy()) - np.array(divnT3.copy())
                        
    # traction vector
    t = Function(VectorSpace([T_none, T_none]))
    for i in range(dim):
        for j in range(dim):
            for k in range(T_none.bases[0].N):
                for m in range(T_none.bases[1].N):
                    t[i][k][m] += T2[i][j][k][m] * n[j]
                    if has_nonzero_hyper_stresses:
                        t[i][k][m] -= (divnT3[i][j][k][m] + 2 * divtT3[i][j][k][m]) * n[j]
            
    return t