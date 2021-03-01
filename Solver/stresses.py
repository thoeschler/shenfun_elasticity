from shenfun import FunctionSpace, VectorSpace, TensorProductSpace, Function, project, Dx, comm
from mpi4py import MPI
from mpi4py_fft.pencil import Subcomm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

def cauchy_stresses(material_parameters, u_hat, plot=False):
    
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
    dom = []
    for i in range(len(V.spaces[0])):
        dom.append(V.spaces[0].bases[i].domain)
    lambd = material_parameters[0]
    mu = material_parameters[1]
    
    # space for stresses
    B = []
    for i in range(dim):
        B.append(FunctionSpace(N, family='legendre', domain=dom[i], bc=None))
    T_none = TensorProductSpace(comm, tuple(B))
    
    # displacement gradient
    H = [ [None for _ in range(dim)] for _ in range(dim)]
    for i in range(dim):
        for j in range(dim):
            H[i][j] = project(Dx(u_hat[i], j), T_none).backward()
    
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
    if plot:
        x_, y_ = T_none.local_mesh()
        X, Y = np.meshgrid(x_, y_, indexing='ij')
        for k in range(dim):
            for l in range(dim):
                fig = plt.figure()
                title = 'T' + str(k + 1) + str(l + 1)
                ax = fig.gca(projection='3d')
                ax.plot_surface(X, Y, T[k][l], cmap=cm.coolwarm)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_title(title)
                plt.show()
    return T


def hyper_stresses(material_parameters, u_hat, plot=False):    
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
    B = []
    for i in range(dim):
        B.append(FunctionSpace(N, family='legendre', domain=dom[i], bc=None))
    T_none = TensorProductSpace(comm, tuple(B))
    
    # zero function
    zero_func = Function(T_none).backward()
    
    # Laplace
    Laplace = [zero_func for _ in range(dim)]
    for i in range(dim):
        for j in range(dim):
            Laplace[i] += project(Dx(u_hat[i], j, 2), T_none).backward()
            
    # grad(div(u))
    GradDiv = [zero_func for _ in range(dim)]
    for i in range(dim):
        for j in range(dim):
            GradDiv[i] += project(Dx(Dx(u_hat[j], j), i), T_none).backward()
    
    # hyper stresses
    T = [ [ [zero_func for _ in range(dim)] for _ in range(dim)] for _ in range(dim)]
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
                    T[i][j][k] += project(c4*Dx(Dx(u_hat[i], j), k), T_none).backward()
                if c5 != 0.:
                    T[i][j][k] += project(0.5*c5*Dx(Dx(u_hat[j], i), k), T_none).backward() + project(0.5*c5*Dx(Dx(u_hat[k], i), j), T_none).backward()
                
    if plot:
        x_, y_ = T_none.local_mesh()
        X, Y = np.meshgrid(x_, y_, indexing='ij')
        for k in range(dim):
            for l in range(dim):
                for m in range(dim):
                    fig = plt.figure()
                    title = 'T' + str(k + 1) + str(l + 1) + str(m + 1)
                    ax = fig.gca(projection='3d')
                    ax.plot_surface(X, Y, T[k][l][m], cmap=cm.coolwarm)
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                    ax.set_title(title)
                    plt.show()
    return T

def traction_vector_gradient(cauchy_stresses, hyper_stresses, normal_vector, plot=False):
    # some paramaters
    dim = len(normal_vector)
    T2 = cauchy_stresses
    T3 = hyper_stresses
    n = normal_vector
    T_none = T2[0][0].function_space()
    
    zero_func = Function(T_none).backward()
    
    # compute traction vector
    t = [zero_func for _ in range(dim)]
    
    # div(T3)
    divT3 = [[zero_func for _ in range(dim)] for _ in range(dim)]
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                divT3[i][j] += project(Dx(T3[i][j][k].forward(), k), T_none).backward()
    # divn(T3), divt(T3)
    divnT3 = [[zero_func for _ in range(dim)] for _ in range(dim)]
    divtT3 = divT3
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for l in range(dim):
                    divnT3[i][j] += ( project(Dx(T3[i][j][k].forward(), l), T_none).backward() )*n[k]*n[l]
                    divtT3[i][j] -= ( project(Dx(T3[i][j][k].forward(), l), T_none).backward() )*n[k]*n[l]
    # traction vector
    t = Function(VectorSpace([T_none, T_none])).backward()
    for i in range(dim):
        for j in range(dim):
            for k in range(T_none.bases[0].N):
                for m in range(T_none.bases[0].N):
                    t[i][k][m] += (T2[i][j][k][m] - divnT3[i][j][k][m] - 2*divtT3[i][j][k][m])*n[j]
        
    if plot:
        x_, y_ = T_none.local_mesh()
        X, Y = np.meshgrid(x_, y_, indexing='ij')
        for k in range(dim):
            fig = plt.figure()
            title = 't' + str(k + 1)
            ax = fig.gca(projection='3d')
            ax.plot_surface(X, Y, t[i], cmap=cm.coolwarm)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(title)
            plt.show()
            
    return t

def double_tractions():
    pass

def edge_forces():
    pass
