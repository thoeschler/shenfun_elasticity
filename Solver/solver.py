import shenfun as sf
from shenfun import inner, comm, Function
from enum import auto, Enum


class DisplacementBCType(Enum):
    fixed = auto()
    fixed_component = auto()
    constant = auto()
    constant_component = auto()
    function = auto()
    function_component = auto()


class TractionBCType(Enum):
    constant = auto()
    constant_component = auto()
    function = auto()
    function_component = auto()


class ElasticSolver:
    def __init__(self, N, domain, bcs, material_parameters, body_forces,
                 elastic_law):
        self._dim = len(N)
        assert self._dim == 2, 'Solver only works for 2D-problems'
        self._N = N
        self._domain = domain
        self._bcs = bcs
        self._material_parameters = material_parameters
        self._elastic_law = elastic_law
        self._body_forces = body_forces
        self._setup_variational_problem()

    def _setup_function_space(self):
        self._nonhomogeneous_bcs = False
        vec_space = []
        for i in range(self._dim):
            tens_space = []
            for j in range(self._dim):
                basis = sf.FunctionSpace(self._N[j], family='legendre',
                                         bc=self._bcs[i][j],
                                         domain=tuple(self._domain[j]))
                if basis.has_nonhomogeneous_bcs:
                    self._nonhomogeneous_bcs = True
                tens_space.append(basis)
            vec_space.append(sf.TensorProductSpace(comm, tuple(tens_space)))

        self._V = sf.VectorSpace(vec_space)

    def _setup_variational_problem(self):
        self._setup_function_space()
        u = sf.TrialFunction(self._V)
        v = sf.TestFunction(self._V)

        self._elastic_law.set_material_parameters(self._material_parameters)
        self._dw_int = self._elastic_law.dw_int(u, v)

        self._dw_ext = inner(v, sf.Array(self._V.get_orthogonal(),
                                         buffer=(0, ) * self._dim))

        if self._body_forces is not None:
            V_body_forces = self._V.get_orthogonal()
            body_forces_quad = sf.Array(V_body_forces,
                                        buffer=self._body_forces)
            self._dw_ext = inner(v, body_forces_quad)

    def solve(self):
        if self._nonhomogeneous_bcs:
            # get boundary matrices
            bc_mats = sf.extract_bc_matrices([self._dw_int])
            # BlockMatrix for homogeneous part
            M = sf.BlockMatrix(self._dw_int)
            # BlockMatrix for inhomogeneous part
            BM = sf.BlockMatrix(bc_mats)
            # inhomogeneous part of solution
            uh_hat = Function(self._V).set_boundary_dofs()
            # pass boundary_matrices to rhs
            add_to_rhs = Function(self._V)
            add_to_rhs = BM.matvec(-uh_hat, add_to_rhs)
            self._dw_ext += add_to_rhs
            # homogeneous part of solution
            u_hat = M.solve(self._dw_ext)
            # solution
            u_hat += uh_hat
        else:
            # BlockMatrix
            M = sf.BlockMatrix(self._dw_int)
            # solution
            u_hat = M.solve(self._dw_ext)

        return u_hat
