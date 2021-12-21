import shenfun as sf
import sympy as sp
import numpy as np
from shenfun import inner, div, grad, Dx


class LinearCauchyElasticity:
    def __init__(self):
        self._name = "LinearCauchyElasticity"
        self._n_material_parameters = 2

    def check_with_pde(self, u_hat, material_parameters, body_forces):
        assert isinstance(u_hat, sf.Function)
        assert len(material_parameters) == self._n_material_parameters
        for comp in body_forces:
            assert isinstance(comp, (sp.Expr, float, int))

        lambd, mu = material_parameters
        V = u_hat.function_space().get_orthogonal()
        # left hand side of pde
        lhs = (lambd + mu) * grad(div(u_hat)) + mu * div(grad(u_hat))

        error_array = sf.Array(V, buffer=body_forces)
        error_array += sf.project(lhs, V).backward()

        error = np.sqrt(inner((1, 1), error_array ** 2))
        # scale by magnitude of solution
        scale = np.sqrt(inner((1, 1), u_hat.backward() ** 2))

        return error / scale

    def compute_body_forces(self, u):
        for comp in u:
            assert isinstance(comp, (sp.Expr, float, int))
        assert hasattr(self, "_material_parameters")
        assert len(self._material_parameters) == self._n_material_parameters

        dim = len(u)
        lmbda, mu = self._material_parameters
        x, y, z = sp.symbols("x,y,z")
        coord = [x, y, z]

        Divergence = 0.
        for i in range(dim):
            Divergence += u[i].diff(coord[i])

        GradDiv = np.empty(dim, dtype=sp.Expr)
        for i in range(dim):
            GradDiv[i] = Divergence.diff(coord[i])

        Laplace = np.zeros(dim, dtype=sp.Expr)
        for i in range(dim):
            for j in range(dim):
                Laplace[i] += u[i].diff(coord[j], 2)

        body_forces = - (lmbda + mu) * GradDiv - mu * Laplace

        return tuple(body_forces)

    def compute_cauchy_stresses(self, u_hat):
        assert isinstance(u_hat, sf.Function)

        space = u_hat[0].function_space().get_orthogonal()
        dim = len(space.bases)
        # number of dofs for each component
        N = [u_hat.function_space().spaces[0].bases[i].N for i in range(dim)]
        lmbda, mu = self._material_parameters

        # displacement gradient
        H = np.empty(shape=(dim, dim, *N))
        for i in range(dim):
            for j in range(dim):
                H[i, j] = sf.project(Dx(u_hat[i], j), space).backward()

        # linear strain tensor
        E = 0.5 * (H + np.swapaxes(H, 0, 1))

        # trace of linear strain tensor
        trE = np.trace(E)

        # create block with identity matrices on diagonal
        identity = np.zeros_like(H)
        for i in range(dim):
            identity[i, i] = np.ones(N)

        # Cauchy stress tensor
        T = 2.0 * mu * E + lmbda * trE * identity

        return T, space

    def dw_int(self, u, v):
        assert isinstance(u, sf.TrialFunction)
        assert isinstance(v, sf.TestFunction)
        assert hasattr(self, "_material_parameters")
        assert len(self._material_parameters) == self._n_material_parameters

        self.dim = u.dimensions
        lmbda, mu = self._material_parameters

        A = inner(mu * grad(u), grad(v))
        B = []
        for i in range(self.dim):
            for j in range(self.dim):
                mat = inner(mu * Dx(u[i], j), Dx(v[j], i))
                if isinstance(mat, list):
                    B += mat
                else:
                    B += [mat]
        C = inner(lmbda * div(u), div(v))
        dw_int = A + B + C

        return dw_int

    def set_material_parameters(self, material_parameters):
        assert len(material_parameters) == self._n_material_parameters
        self._material_parameters = material_parameters

    @property
    def name(self):
        return self._name

    @property
    def material_parameters(self):
        assert hasattr(self, "_material_parameters")
        return self._material_parameters


class LinearGradientElasticity:
    def __init__(self):
        self._name = "LinearGradientElasticity"
        self._n_material_parameters = 7

    def check_with_pde(self, u_hat, material_parameters, body_forces):
        assert isinstance(u_hat, sf.Function)
        assert len(material_parameters) == self._n_material_parameters
        for comp in body_forces:
            assert isinstance(comp, (sp.Expr, float, int))

        lambd, mu, c1, c2, c3, c4, c5 = material_parameters
        V = u_hat.function_space().get_orthogonal()
        # left hand side of pde
        lhs = (lambd + mu) * grad(div(u_hat)) + mu * div(grad(u_hat)) \
            - (c1 + c4) * div(grad(div(grad(u_hat)))) \
            - (c2 + c3 + c5) * grad(div(div(grad(u_hat))))

        error_array = sf.Array(V, buffer=body_forces)
        error_array += sf.project(lhs, V).backward()

        error = np.sqrt(inner((1, 1), error_array ** 2))
        # scale by magnitude of solution
        scale = np.sqrt(inner((1, 1), u_hat.backward() ** 2))

        return error / scale

    def compute_cauchy_stresses(self, u_hat):
        assert isinstance(u_hat, sf.Function)

        space = u_hat[0].function_space().get_orthogonal()
        dim = len(space.bases)
        # number of dofs for each component
        N = [u_hat.function_space().spaces[0].bases[i].N for i in range(dim)]
        lmbda, mu = self._material_parameters[slice(2)]

        # displacement gradient
        H = np.empty(shape=(dim, dim, *N))
        for i in range(dim):
            for j in range(dim):
                H[i, j] = sf.project(Dx(u_hat[i], j), space).backward()

        # linear strain tensor
        E = 0.5 * (H + np.swapaxes(H, 0, 1))

        # trace of linear strain tensor
        trE = np.trace(E)

        # create block with identity matrices on diagonal
        identity = np.zeros_like(H)
        for i in range(dim):
            identity[i, i] = np.ones(N)

        # Cauchy stress tensor
        T = 2.0 * mu * E + lmbda * trE * identity

        return T, space

    def compute_hyper_stresses(self, u_hat):
        assert isinstance(u_hat, sf.Function)

        space = u_hat[0].function_space()
        dim = len(space.bases)
        c1, c2, c3, c4, c5 = self._material_parameters[2:]
        N = [u_hat.function_space().spaces[0].bases[i].N for i in range(dim)]

        Laplace = np.zeros(shape=(dim, *N))
        for i in range(dim):
            for j in range(dim):
                Laplace[i] += sf.project(Dx(u_hat[i], j, 2), space).backward()

        GradDiv = np.zeros(shape=(dim, *N))
        for i in range(dim):
            for j in range(dim):
                GradDiv[i] += sf.project(
                        Dx(Dx(u_hat[j], j), i), space
                                        ).backward()

        GradGrad = np.empty(shape=(dim, dim, dim, *N))
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    GradGrad[i, j, k] = sf.project(
                            Dx(Dx(u_hat[i], j), k), space
                            ).backward()

        identity = np.identity(dim)

        # hyper stresses
        T = c1 * np.swapaxes(np.tensordot(identity, Laplace, axes=0), 0, 2) \
            + c2 / 2.0 * (np.tensordot(identity, Laplace, axes=0) +
                          np.swapaxes(np.tensordot(identity, Laplace, axes=0), 1, 2)
                          ) \
            + c3 / 2.0 * (np.tensordot(identity, GradDiv, axes=0) +
                          np.swapaxes(np.tensordot(identity, GradDiv, axes=0), 1, 2)
                          ) \
            + c4 * GradGrad \
            + c5 / 2.0 * (np.swapaxes(GradGrad, 0, 1) + np.swapaxes(GradGrad, 0, 2))

        return T, space

    def dw_int(self, u, v):
        assert isinstance(u, sf.TrialFunction)
        assert isinstance(v, sf.TestFunction)
        assert hasattr(self, "_material_parameters")
        assert len(self._material_parameters) == self._n_material_parameters

        self.dim = u.dimensions
        lmbda, mu, c1, c2, c3, c4, c5 = self._material_parameters
        dw_int = []

        if c1 != 0.0:
            dw_int += inner(c1 * div(grad(u)), div(grad(v)))
        if c2 != 0.0:
            dw_int += inner(c2 * div(grad(u)), grad(div(v)))
        if c3 != 0.0:
            dw_int += inner(c3 * grad(div(u)), grad(div(v)))
        if c4 != 0.0:
            for i in range(self.dim):
                for j in range(self.dim):
                    for k in range(self.dim):
                        mat = inner(c4 * Dx(Dx(u[i], j), k),
                                    Dx(Dx(v[i], j), k))
                        if isinstance(mat, list):
                            dw_int += mat
                        else:
                            dw_int += [mat]
        if c5 != 0.0:
            for i in range(self.dim):
                for j in range(self.dim):
                    for k in range(self.dim):
                        mat = inner(c5 * Dx(Dx(u[j], i), k),
                                    Dx(Dx(v[i], j), k))
                        if isinstance(mat, list):
                            dw_int += mat
                        else:
                            dw_int += [mat]

        # add the classical cauchy-terms
        dw_int += inner(mu * grad(u), grad(v))

        for i in range(self.dim):
            for j in range(self.dim):
                mat = inner(mu * Dx(u[i], j), Dx(v[j], i))
                if isinstance(mat, list):
                    dw_int += mat
                else:
                    dw_int += [mat]
        dw_int += inner(lmbda * div(u), div(v))

        return dw_int

    def compute_body_forces(self, u):
        for comp in u:
            assert isinstance(comp, (sp.Expr, float, int))
        assert hasattr(self, "_material_parameters")
        assert len(self._material_parameters) == self._n_material_parameters

        dim = len(u)
        lmbda, mu, c1, c2, c3, c4, c5 = self._material_parameters

        x, y, z = sp.symbols("x,y,z")
        coord = [x, y, z]

        Divergence = 0.
        for i in range(dim):
            Divergence += u[i].diff(coord[i])

        Laplace = np.zeros(dim, dtype=sp.Expr)
        for i in range(dim):
            for j in range(dim):
                Laplace[i] += u[i].diff(coord[j], 2)

        GradDiv = np.empty(dim, dtype=sp.Expr)
        for i in range(dim):
            GradDiv[i] = Divergence.diff(coord[i])

        DoubleLaplace = np.zeros(dim, dtype=sp.Expr)
        for i in range(dim):
            for j in range(dim):
                DoubleLaplace[i] += Laplace[i].diff(coord[j], 2)

        DivDivGrad = 0.
        for i in range(dim):
            DivDivGrad += Laplace[i].diff(coord[i])

        GradDivDivGrad = np.empty(dim, dtype=sp.Expr)
        for i in range(dim):
            GradDivDivGrad[i] = DivDivGrad.diff(coord[i])

        body_forces = (c1 + c4) * DoubleLaplace + \
            (c2 + c3 + c5) * GradDivDivGrad - \
            (lmbda + mu) * GradDiv - mu * Laplace

        return tuple(body_forces)

    def set_material_parameters(self, material_parameters):
        assert len(material_parameters) == self._n_material_parameters
        self._material_parameters = material_parameters

    @property
    def name(self):
        return self._name

    @property
    def material_parameters(self):
        return self._material_parameters
