from sympy import symbols

def body_forces_cauchy(u, material_parameters):
    
    # assert inputs
    assert isinstance(u, tuple)
    assert isinstance(material_parameters, tuple)
    assert len(material_parameters) == 2
    for val in material_parameters:
        assert isinstance(val, float)
    
    # some parameters
    dim = len(u)
    lambd = material_parameters[0]
    mu = material_parameters[1]
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
    body_forces = []
    for i in range(dim):
        body_forces.append(- ( (lambd + mu)*GradDiv[i] + mu*Laplace[i] ))
        
    return tuple(body_forces)



def body_forces_gradient(u, material_parameters):   
    # assert input
    assert isinstance(u, tuple)
    assert isinstance(material_parameters, tuple)
    assert len(material_parameters) == 7
    
    # some parameters
    dim = len(u)
    lambd = material_parameters[0]
    mu = material_parameters[1]
    c1 = material_parameters[2]
    c2 = material_parameters[3]
    c3 = material_parameters[4]
    c4 = material_parameters[5]
    c5 = material_parameters[6]
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
    body_forces = []
    for i in range(dim):
        body_forces.append((c1 + c4)*DoubleLaplace[i] + (c2 + c3 + c5)*GradDivDivGrad[i] \
                           - (lambd + mu)*GradDiv[i] - mu*Laplace[i])
        
    return tuple(body_forces)