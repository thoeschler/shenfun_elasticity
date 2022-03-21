import shenfun as sf
import numpy as np
from shenfun import inner, comm, Function, Array


class ElasticSolver:
    def __init__(self, N, domain, bcs, traction_bcs, material_parameters,
                 body_forces, elastic_law):
        self._dim = len(N)
        assert self._dim == 2, 'Solver only works for 2D-problems'
        self._N = N
        self._domain = domain
        self._bcs = bcs
        self._traction_bcs = traction_bcs
        self._material_parameters = material_parameters
        self._elastic_law = elastic_law
        self._body_forces = body_forces
        self._setup_variational_problem()

    def _setup_function_space(self):
        self._nonhomogeneous_bcs = False
        self._function_spaces = []
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
            self._function_spaces.append(tens_space)
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
        if self._dim == 2:
            map_boundary_to_component_index = {'right': 0, 'top': 1, 'left': 0,
                                               'bottom': 1}
            map_boundary_to_start_end_index = {'right': 1, 'top': 1,
                                               'left': -1, 'bottom': -1}
        if self._traction_bcs:
            boundary_traction_term = Function(self._V.get_orthogonal())

            # b: boundary, c: component
            for b, c, value in self._traction_bcs:
                if self._dim == 2:
                    side_index = map_boundary_to_component_index[b]
                    bdry_basis_index = 1 if side_index == 0 else 0
                    boundary_basis = self._function_spaces[c][bdry_basis_index]
                    start_or_end_index = map_boundary_to_start_end_index[b]
                    # test function for boundary integral
                    v_boundary = sf.TestFunction(boundary_basis)
                    if isinstance(value, (float, int)):
                        trac = Array(boundary_basis, val=value)
                    else:
                        trac = Array(boundary_basis, buffer=value)
                    evaluate_on_boundary = self._function_spaces[c][
                        side_index].evaluate_basis_all(start_or_end_index)
                    project_traction = inner(trac, v_boundary)
                    if side_index == 0:
                        boundary_traction_term[c] += np.outer(
                            evaluate_on_boundary, project_traction)
                    elif side_index == 1:
                        boundary_traction_term[c] += np.outer(
                            project_traction, evaluate_on_boundary)
                    else:
                        raise ValueError()

        M = sf.BlockMatrix(self._dw_int)
        if self._traction_bcs:
            self._dw_ext += boundary_traction_term
        u_hat = M.solve(self._dw_ext)


        return u_hat
