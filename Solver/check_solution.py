from shenfun import div, grad, Function, project, FunctionSpace, TensorProductSpace, Array, VectorSpace, inner, comm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math

def check_solution_cauchy(u_hat, material_parameters, body_forces):
    assert isinstance(u_hat, Function)
    assert isinstance(material_parameters, tuple)
    assert len(material_parameters) == 2
    for val in material_parameters:
        assert isinstance(val, float)
    
    # some parameters
    dim = len(u_hat)
    lambd = material_parameters[0]
    mu = material_parameters[1]
    
    # left hand side
    lhs = (lambd + mu)*grad(div(u_hat)) + mu*div(grad(u_hat))
    
    # space for volumetric forces
    bases = []
    for i in range(dim):
        bases.append(FunctionSpace(N=u_hat.function_space().spaces[i].bases[i].N, \
                                   domain=u_hat.function_space().spaces[i].bases[i].domain, \
                                       family='legendre', bc=None))
    T = TensorProductSpace(comm, tuple(bases))
    V = VectorSpace([T, T])
    
    # test solution using lame-navier-equation
    error_array = Array(V, buffer=body_forces)  
    for i in range(dim):
        error_array[i] += project(lhs[i], T).backward()
    
    for i in range(dim):
        for j in range(len(error_array[i])):
            for k in range(len(error_array[i][j])):
                error_array[i][j][k] = error_array[i][j][k]**2
    error = inner((1, 1), error_array)
    return math.sqrt(error)


def check_solution_gradient(u_hat, material_parameters, body_forces):
    assert isinstance(u_hat, Function)
    assert isinstance(material_parameters, tuple)
    assert len(material_parameters) == 7
    for val in material_parameters:
        assert isinstance(val, float)
    
    # some parameters
    dim = len(u_hat)
    lambd = material_parameters[0]
    mu = material_parameters[1]
    c1 = material_parameters[2]
    c2 = material_parameters[3]
    c3 = material_parameters[4]
    c4 = material_parameters[5]
    c5 = material_parameters[6]
    
    # left hand side
    lhs = (lambd + mu)*grad(div(u_hat)) + mu*div(grad(u_hat)) - \
        (c1 + c4)*div(grad(div(grad(u_hat)))) - (c2 + c3 + c5)*grad(div(div(grad(u_hat))))
    
    # space for volumetric forces
    bases = []
    for i in range(dim):
        bases.append(FunctionSpace(N=u_hat.function_space().spaces[i].bases[i].N, \
                                   domain=u_hat.function_space().spaces[i].bases[i].domain, \
                                       family='legendre', bc=None))
    T = TensorProductSpace(comm, tuple(bases))
    V = VectorSpace([T, T])
    
    # test solution using lame-navier-equation
    error_array = Array(V, buffer=body_forces)  
    for i in range(dim):
        error_array[i] += project(lhs[i], T).backward()
    
    for i in range(dim):
        for j in range(len(error_array[i])):
            for k in range(len(error_array[i][j])):
                error_array[i][j][k] = error_array[i][j][k]**2
    error = inner((1, 1), error_array)
    return math.sqrt(error)