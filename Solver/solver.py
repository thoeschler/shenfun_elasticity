import shenfun as sf
from shenfun import inner, comm, Function


class ElasticSolver:
    def __init__(self, N, domain, bc, material_parameters, body_forces,
                 elastic_law):
        self.dim = len(N)
        self.N = N
        self.domain = domain
        self.bc = bc
        self.material_parameters = material_parameters
        self.elastic_law = elastic_law
        self.body_forces = body_forces

    def setup_function_space(self):
        self.nonhomogeneous_bcs = False
        vec_space = []
        for i in range(self.dim):
            tens_space = []
            for j in range(self.dim):
                basis = sf.FunctionSpace(self.N[j], family='legendre',
                                         bc=self.bc[i][j],
                                         domain=tuple(self.domain[j]))
                if basis.has_nonhomogeneous_bcs:
                    self.nonhomogeneous_bcs = True
                tens_space.append(basis)
            vec_space.append(sf.TensorProductSpace(comm, tuple(tens_space)))

        V = sf.VectorSpace(vec_space)

        return V

    def setup_variational_problem(self):
        self.V = self.setup_function_space()
        u = sf.TrialFunction(self.V)
        v = sf.TestFunction(self.V)

        self.elastic_law.set_material_parameters(self.material_parameters)
        self.dw_int = self.elastic_law.dw_int(u, v)

        self.dw_ext = inner(v, sf.Array(self.V.get_orthogonal(),
                                        buffer=(0, ) * self.dim
                                        )
                            )

        if self.body_forces is not None:
            V_body_forces = self.V.get_orthogonal()
            body_forces_quad = sf.Array(V_body_forces,
                                        buffer=self.body_forces)
            self.dw_ext = inner(v, body_forces_quad)

    def solve(self):
        if self.nonhomogeneous_bcs:
            # get boundary matrices
            bc_mats = sf.extract_bc_matrices([self.dw_int])
            # BlockMatrix for homogeneous part
            M = sf.BlockMatrix(self.dw_int)
            # BlockMatrix for inhomogeneous part
            BM = sf.BlockMatrix(bc_mats)
            # inhomogeneous part of solution
            uh_hat = Function(self.V).set_boundary_dofs()
            # pass boundary_matrices to rhs
            add_to_rhs = Function(self.V)
            add_to_rhs = BM.matvec(-uh_hat, add_to_rhs)
            self.dw_ext += add_to_rhs
            # homogeneous part of solution
            u_hat = M.solve(self.dw_ext)
            # solution
            u_hat += uh_hat
        else:
            # BlockMatrix
            M = sf.BlockMatrix(self.dw_int)
            # solution
            u_hat = M.solve(self.dw_ext)
        return u_hat
