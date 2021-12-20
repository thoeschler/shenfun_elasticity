from shenfun_elasticity.solver.solver import ElasticSolver
from shenfun_elasticity.solver.utilities import get_dimensionless_parameters
from mpi4py_fft import generate_xdmf
import shenfun as sf
import os


class ElasticProblem:
    def __init__(self, N, domain, elastic_law):
        self._N = N
        self._body_forces = None
        self._dim = len(N)
        self._domain = domain
        self._elastic_law = elastic_law
        self._setup_problem()

    def get_dimensional_solution(self):
        assert hasattr(self, "_solution")
        vec_space = list()
        for i in range(self._dim):
            tens_space = []
            for j in range(self._dim):
                basis = sf.FunctionSpace(self._N[j], family='legendre',
                                         bc=self._bcs[i][j],
                                         domain=tuple(self._domain[j]))
                tens_space.append(basis)
            vec_space.append(sf.TensorProductSpace(sf.comm, tuple(tens_space)))

        V = sf.VectorSpace(vec_space)
        u_hat = sf.Function(V)
        for i in range(self._dim):
            # u has the same expansions coefficients as u_dimless
            u_hat[i] = self._u_ref * self._solution[i]
        return u_hat

    def _get_solver(self):
        return ElasticSolver(self._N, self._dimless_domain, self._dimless_bcs,
                             self._dimless_material_parameters,
                             self._dimless_body_forces, self._elastic_law)

    def postprocess(self):
        dirs = ['results', str(self.name), str(self.elastic_law.name)]
        for d in dirs:
            if not os.path.exists(d):
                os.mkdir(d)
            os.chdir(d)

        output = []
        # displacement
        u = self.get_dimensional_solution()

        V = u.function_space()
        fl_disp_name = 'displacement'
        fl_disp = sf.ShenfunFile(fl_disp_name, V, backend='hdf5', mode='w',
                                 uniform=True)
        output.append(fl_disp_name + '.h5')

        for i in range(self.dim):
            fl_disp.write(i, {'u': [u.backward(kind='uniform')]},
                          as_scalar=True)

        # stress
        stress, space = self.elastic_law.compute_cauchy_stresses(u)
        fl_stress_name = 'cauchy_stress'
        fl_stress = sf.ShenfunFile(fl_stress_name, space,
                                   backend='hdf5', mode='w', uniform=True)
        for i in range(self.dim):
            for j in range(self.dim):
                s = sf.Array(space, buffer=stress[i, j])
                fl_stress.write(0, {f'S{i}{j}': [s]}, as_scalar=True)
        output.append(fl_stress_name + '.h5')

        # hyper stress
        if self.elastic_law.name == 'LinearGradientElasticity':
            stress, space = self.elastic_law.compute_hyper_stresses(u)
            fl_stress_name = 'hyper_stress'
            fl_stress = sf.ShenfunFile(fl_stress_name, space,
                                       backend='hdf5', mode='w', uniform=True)
            for i in range(self.dim):
                for j in range(self.dim):
                    for k in range(self.dim):
                        s = sf.Array(space, buffer=stress[i, j])
                        fl_stress.write(0, {f'S{i}{j}{k}': [s]},
                                        as_scalar=True)
        output.append(fl_stress_name + '.h5')
        self.write_xdmf_file(output)
        os.chdir('../../..')

    def _setup_problem(self):
        assert hasattr(self, "set_boundary_conditions")
        assert hasattr(self, "set_material_parameters")

        self.set_material_parameters()
        if hasattr(self, "set_analytical_solution"):
            self.set_analytical_solution()

        self.set_boundary_conditions()
        self._elastic_law.set_material_parameters(self._material_parameters)

        if hasattr(self, "set_body_forces"):
            self.set_body_forces()

        if hasattr(self, "set_nondim_parameters"):
            self.set_nondim_parameters()
        else:
            self._l_ref = self._domain[0][1]
            self._u_ref = 1
            self._mat_param_ref = self._material_parameters[0]

        # dl means dimensionless
        self._dimless_domain, self._dimless_bcs, self._dimless_body_forces, \
            self._dimless_material_parameters = get_dimensionless_parameters(
                    self._domain, self._bcs, self._body_forces,
                    self._material_parameters, self._u_ref, self._l_ref,
                    self._mat_param_ref)

    def solve(self):
        self._solver = self._get_solver()
        self._solution = self._solver.solve()

    def write_xdmf_file(self, files):
        if not isinstance(files, (list, tuple)):
            generate_xdmf(files)
        else:
            for f in files:
                generate_xdmf(f)

    @property
    def dim(self):
        return self._dim

    @property
    def elastic_law(self):
        return self._elastic_law

    @property
    def material_parameters(self):
        return self._material_parameters

    def name(self):
        assert hasattr(self, "_name")
        return self._name

    @property
    def solution(self):
        assert hasattr(self, "_solution")
        return self._solution

    @property
    def solver(self):
        assert hasattr(self, "_solver")
        return self._solver
