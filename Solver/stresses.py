from shenfun import FunctionSpace, VectorSpace, TensorProductSpace, Function, project, Dx, comm

def cauchy_stresses(material_parameters, u_hat):
    
    # input assertion
    assert isinstance(material_parameters, tuple)
    for val in material_parameters:
        assert isinstance(val, float)
    assert len(material_parameters) == 2
    assert isinstance(u_hat, Function)
    
    # some parameters
    V = u_hat.function_space()
    dim = len(V.spaces)
    N = V.spaces[0].bases[0].N
    dom = tuple([
        V.spaces[i].bases[i].domain for i in range(dim)
        ])

    lambd = material_parameters[0]
    mu = material_parameters[1]
    
    # space for stresses
    tens_space = tuple([
        FunctionSpace(N, family='legendre', domain=dom[i], bc=None) for i in range(dim)
        ])
    T_none = TensorProductSpace(comm, tens_space)
    
    # displacement gradient
    H = [ [None for _ in range(dim)] for _ in range(dim)]
    for i in range(dim):
        for j in range(dim):
            H[i][j] = project(Dx(u_hat[i], j), T_none)
    
    # linear strain tensor
    E = [ [None for _ in range(dim)] for _ in range(dim)]
    for i in range(dim):
        for j in range(dim):
            E[i][j] = 0.5 * (H[i][j] + H[j][i])
    
    # trace of linear strain tensor
    trE = 0.
    for i in range(dim):
        trE += E[i][i]
    
    # Cauchy stress tensor
    T = [ [None for _ in range(dim)] for _ in range(dim)]
    for i in range(dim):
        for j in range(dim):
            T[i][j] = 2.0 * mu * E[i][j] 
            if i==j:
                T[i][j] += lambd * trE

    return T


def hyper_stresses(material_parameters, u_hat):    
    # input assertion
    assert isinstance(material_parameters, tuple)
    for val in material_parameters:
        assert isinstance(val, float)
    assert len(material_parameters) == 5
    assert isinstance(u_hat, Function)
    
    # some parameters
    V = u_hat.function_space()

    dim = len(V.spaces)
    N = V.spaces[0].bases[0].N
    dom = []
    for i in range(len(V.spaces[0])):
        dom.append(V.spaces[0].bases[i].domain)
    c1 = material_parameters[0]
    c2 = material_parameters[1]
    c3 = material_parameters[2]
    c4 = material_parameters[3]
    c5 = material_parameters[4]
    
    # space for stresses
    tens_space = tuple([
        FunctionSpace(N, family='legendre', domain=dom[i], bc=None) for i in range(dim)
        ])
    T_none = TensorProductSpace(comm, tens_space)
    
    # Laplace
    Laplace = [0. for _ in range(dim)]
    for i in range(dim):
        for j in range(dim):
            Laplace[i] += project(Dx(u_hat[i], j, 2), T_none)
            
    # grad(div(u))
    GradDiv = [0. for _ in range(dim)]
    for i in range(dim):
        for j in range(dim):
            GradDiv[i] += project(Dx(Dx(u_hat[j], j), i), T_none)
    
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
    # some paramaters
    dim = len(normal_vector)
    T2 = cauchy_stresses
    T3 = hyper_stresses
    n = normal_vector
    T_none = T2[0][0].function_space()
    
    # compute traction vector
    t = [0. for _ in range(dim)]
    
    # div(T3)
    divT3 = [[0. for _ in range(dim)] for _ in range(dim)]
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                divT3[i][j] += project(Dx(T3[i][j][k], k), T_none)
                
    # divn(T3), divt(T3)
    divnT3 = [[0. for _ in range(dim)] for _ in range(dim)]
    divtT3 = divT3.copy()
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for l in range(dim):
                    divnT3[i][j] += ( project(Dx(T3[i][j][k], l), T_none) )*n[k]*n[l]
                    divtT3[i][j] -= ( project(Dx(T3[i][j][k], l), T_none) )*n[k]*n[l]
                    
    # traction vector
    t = Function(VectorSpace([T_none, T_none]))
    for i in range(dim):
        for j in range(dim):
            for k in range(T_none.bases[0].N):
                for m in range(T_none.bases[0].N):
                    t[i][k][m] += (T2[i][j][k][m] - divnT3[i][j][k][m] - 2*divtT3[i][j][k][m])*n[j]
            
    return t