from shenfun_elasticity.Solver.solver import ElasticSolver
from shenfun_elasticity.Solver.utilities import get_dimensionless_parameters


class ElasticProblem:
    def __init__(self, N, domain, elastic_law):
        self.N = N
        self.domain = domain
        self.elastic_law = elastic_law
        self.setup_problem()

    def get_solver(self):
        return ElasticSolver(self.N, self.domain_dl, self.bc_dl,
                             self.material_parameters_dl, self.body_forces_dl,
                             self.elastic_law)

    def setup_problem(self):
        assert hasattr(self, "set_boundary_conditions")
        assert hasattr(self, "set_material_parameters")
        if hasattr(self, "set_analytical_solution"):
            self.set_analytical_solution()

        self.bc = self.set_boundary_conditions()
        self.material_parameters = self.set_material_parameters()
        self.elastic_law.set_material_parameters(self.material_parameters)

        if hasattr(self, "set_body_forces"):
            self.body_forces = self.set_body_forces()
        else:
            self.body_forces = None

        if hasattr(self, "set_nondim_parameters"):
            self.l_ref, self.u_ref, self.mat_param_ref = \
                self.set_nondim_parameters()
        else:
            self.l_ref = self.domain[0][1]
            self.u_ref = 1
            self.mat_param_ref = self.material_parameters[0]

        # dl means dimensionless
        self.domain_dl, self.bc_dl, self.body_forces_dl, \
            self.material_parameters_dl = get_dimensionless_parameters(
                    self.domain, self.bc, self.body_forces,
                    self.material_parameters, self.u_ref, self.l_ref,
                    self.mat_param_ref)

    def solve(self):
        solver = self.get_solver()
        solver.setup_variational_problem()

        self.solution = solver.solve()

        return self.solution
