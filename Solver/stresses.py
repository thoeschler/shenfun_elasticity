from shenfun import Function, project, Dx, VectorSpace
import numpy as np
import itertools as it


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
    T, space : tuple[numpy.ndarray,
                         shenfun.tensorproductspace.TensorProductSpace]
        T: stress values in physical space
        space: space corresponding to the components of T

    '''
    # input assertion
    assert isinstance(material_parameters, tuple)
    assert len(material_parameters) == 2
    assert isinstance(u_hat, Function)

    # some parameters
    space = u_hat[0].function_space().get_orthogonal()
    dim = len(space.bases)

    # size of discretization
    N = [u_hat.function_space().spaces[0].bases[i].N for i in range(dim)]

    lmbda, mu = material_parameters

    # displacement gradient
    H = np.empty(shape=(dim, dim, *N))
    for i in range(dim):
        for j in range(dim):
            H[i, j] = project(Dx(u_hat[i], j), space).backward()

    # linear strain tensor, transpose first to indices of array
    E = 0.5 * (H + np.transpose(
            H, axes=np.hstack(
                    ((1, 0), range(2, 2 + dim)))
            )
            )

    # trace of linear strain tensor
    trE = np.trace(E)

    # create block with identity matrices on diagonal
    identity = np.zeros_like(H)
    for i in range(dim):
        identity[i, i] = np.ones(N)

    # Cauchy stress tensor
    T = 2.0 * mu * E + lmbda * trE * identity

    # return stresses and the space as sigma is now just a numpy.ndarray
    return T, space


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
    space = u_hat[0].function_space()
    dim = len(space.bases)
    c1, c2, c3, c4, c5 = material_parameters

    # size of discretization
    N = [u_hat.function_space().spaces[0].bases[i].N for i in range(dim)]

    # Laplace
    Laplace = np.zeros(shape=(dim, *N))
    for i in range(dim):
        for j in range(dim):
            Laplace[i] += project(Dx(u_hat[i], j, 2), space).backward()

    # grad(div(u))
    GradDiv = np.zeros(shape=(dim, *N))
    for i in range(dim):
        for j in range(dim):
            GradDiv[i] += project(Dx(Dx(u_hat[j], j), i), space).backward()

    # grad(grad(u))
    GradGrad = np.empty(shape=(dim, dim, dim, *N))
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                GradGrad[i, j, k] = project(
                        Dx(Dx(u_hat[i], j), k), space
                        ).backward()

    # create block with identity matrices on diagonal
    identity = np.identity(dim)

    # define axes for transposition
    # use np.transpose(..., axes=axes[0, 2, 1]) to transpose axes 1 and 2
    ax = [np.hstack((val, range(3, 3 + dim))) for val
          in it.product(range(3), repeat=3)]
    axes = np.reshape(ax, (3, 3, 3, 3 + dim))

    # hyper stresses
    T = c1 * np.transpose(
            np.tensordot(identity, Laplace, axes=0), axes=axes[2, 1, 0]
            ) \
        + c2 * (np.tensordot(identity, Laplace, axes=0) +
                np.transpose(np.tensordot(identity, Laplace, axes=0),
                             axes=axes[0, 2, 1])
                ) \
        + c3 * (np.tensordot(identity, GradDiv, axes=0) +
                np.transpose(np.tensordot(identity, GradDiv, axes=0),
                             axes=axes[0, 2, 1])
                ) \
        + c4 * GradGrad \
        + c5 / 2 * (
                np.transpose(GradGrad, axes=axes[1, 0, 2]) +
                np.transpose(GradGrad, axes=axes[2, 1, 0])
                )

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