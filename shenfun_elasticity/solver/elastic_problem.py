from shenfun_elasticity.solver.solver import ElasticSolver
from shenfun_elasticity.solver.utilities import get_dimensionless_parameters
from mpi4py_fft import generate_xdmf
import shenfun as sf
import os
from copy import copy
from enum import auto, Enum


class DisplacementBC(Enum):
    fixed = auto()
    fixed_component = auto()
    fixed_gradient = auto()
    fixed_gradient_component = auto()
    function = auto()
    function_component = auto()


class TractionBC(Enum):
    function = auto()
    function_component = auto()


class DoubleTractionBC(Enum):
    function = auto()
    function_component = auto()


class ElasticProblem:
    def __init__(self, N, domain, elastic_law):
        self._N = N
        self._body_forces = None
        self._dim = len(N)
        self._domain = domain
        self._elastic_law = elastic_law
        self._setup_problem()

    def _organize_boundary_conditions(self):
        assert hasattr(self, '_processed_bcs')
        for component in range(self._dim):
            for direction in range(self._dim):
                # check for bcs that fix a single value: those can not be
                # implemented via a dictionary
                bc = copy(self._processed_bcs[component][direction])
                assert isinstance(bc, dict)
                if sum([len(val) for val in bc.values()]) == 1:
                    if len(bc['left']) > 0:
                        single_bc = bc['left'][0]
                        index = 0
                    elif len(bc['right']) > 0:
                        single_bc = bc['right'][0]
                        index = 1
                    else:
                        raise ValueError()
                    new_bc_format = [None, None]
                    # for now the boundary condition for only one side
                    # needs to be of dirichlet type
                    assert single_bc[0] == 'D'
                    new_bc_format[index] = single_bc[1]
                    self._processed_bcs[component][direction] = tuple(new_bc_format)
        # remove empty lists
        for component in range(self._dim):
            for direction in range(self._dim):
                bc = copy(self._processed_bcs[component][direction])
                if not isinstance(bc, dict):
                    continue
                for side, value in bc.items():
                    if value == []:
                        self._processed_bcs[component][direction].pop(side)

        # replace empty dictionaries
        for component in range(self._dim):
            for direction in range(self._dim):
                bc = copy(self._processed_bcs[component][direction])
                if not isinstance(bc, dict):
                    continue
                # check whether dictionary is empty
                if not bc:
                    self._processed_bcs[component][direction] = None
        self._bcs = self._processed_bcs

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
                             self._dimless_traction_bcs,
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

    def _process_boundary_conditions(self):
        assert hasattr(self, "_bcs")
        if self._dim == 2:
            for boundary, bc in self._bcs.items():
                assert boundary in ('right', 'top', 'left', 'bottom')
                assert isinstance(bc, list)
            map_boundary_to_direction_index = {'left': 0, 'right': 0,
                                               'bottom': 1, 'top': 1}
            map_boundary_to_side = {'right': 'right', 'top': 'right',
                                    'left': 'left', 'bottom': 'left'}
        elif self._dim == 3:
            for boundary, bc in self._bcs.items():
                assert boundary in ('right', 'top', 'left', 'bottom', 'back', 'front')
                assert isinstance(bc, list)
            map_boundary_to_direction_index = {'left': 0, 'right': 0,
                                               'back': 1, 'front': 1,
                                               'bottom': 2, 'top': 2}
            map_boundary_to_side = {'left': 'left', 'right': 'right',
                                    'bottom': 'left', 'top': 'right',
                                    'back': 'left', 'front': 'right'}
        else:
            raise ValueError()
        # repeat base dictionary for every component and direction
        self._processed_bcs = [[{'left': list(), 'right': list()} for _ in range(self._dim)]
                               for _ in range(self._dim)]
        self._traction_bcs = []
        for boundary, bcs in self._bcs.items():
            # direction
            d = map_boundary_to_direction_index[boundary]
            side = map_boundary_to_side[boundary]
            for bc in bcs:
                if len(bc) == 2:
                    bc_type, value = bc
                elif len(bc) == 3:
                    bc_type, c, value = bc
                # create bcs in shenfun style
                if bc_type is DisplacementBC.fixed:
                    for c in range(self._dim):
                        self._processed_bcs[c][d][side].append(('D', 0.))
                elif bc_type is DisplacementBC.fixed_gradient:
                    for c in range(self._dim):
                        self._processed_bcs[c][d][side].append(('N', 0.))
                elif bc_type is DisplacementBC.fixed_component:
                    self._processed_bcs[c][d][side].append(('D', 0.))
                elif bc_type is DisplacementBC.fixed_gradient_component:
                    self._processed_bcs[c][d][side].append(('N', 0.))
                elif bc_type is DisplacementBC.function:
                    for c, val in enumerate(value):
                        self._processed_bcs[c][d][side].append(('D', val))
                elif bc_type is DisplacementBC.function_component:
                    self._processed_bcs[c][d][side].append(('D', value))
                elif bc_type is TractionBC.function:
                    for c in range(self._dim):
                        self._traction_bcs.append((boundary, c, value[c]))
                elif bc_type is TractionBC.function_component:
                    self._traction_bcs.append((boundary, c, value))
        self._organize_boundary_conditions()

    def _setup_problem(self):
        assert hasattr(self, "set_boundary_conditions")
        assert hasattr(self, "set_material_parameters")

        self.set_material_parameters()
        if hasattr(self, "set_analytical_solution"):
            self.set_analytical_solution()

        self.set_boundary_conditions()
        self._process_boundary_conditions()
        self._elastic_law.set_material_parameters(self._material_parameters)

        if hasattr(self, "set_body_forces"):
            self.set_body_forces()

        if hasattr(self, "set_nondim_parameters"):
            self.set_nondim_parameters()
        else:
            self._l_ref = self._domain[0][1]
            self._u_ref = 1
            self._mat_param_ref = self._material_parameters[0]

        # get dimensionless values for solver
        self._dimless_domain, self._dimless_bcs, self._dimless_traction_bcs, \
            self._dimless_body_forces, self._dimless_material_parameters = \
            get_dimensionless_parameters(
                    self._domain, self._bcs, self._traction_bcs,
                    self._body_forces, self._material_parameters, self._u_ref,
                    self._l_ref, self._mat_param_ref)

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

    @property
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
