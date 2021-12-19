from shenfun_elasticity.Solver.elastic_problem import ElasticProblem, \
    DisplacementBC, TractionBC
from shenfun_elasticity.Solver.elastic_law import LinearCauchyElasticity, \
    LinearGradientElasticity
from shenfun_elasticity.Solver.utilities import compute_numerical_error, \
    get_dimensionless_displacement
import numpy as np
import sympy as sp
import shenfun as sf
from sympy import cos, sin, pi, sinh, cosh


class DirichletTest(ElasticProblem):
    def __init__(self, N, domain, elastic_law):
        self._name = 'DirichletTest'
        self.ell, self.h = domain[0][1], domain[1][1]
        self.u0 = self.ell / 100
        super().__init__(N, domain, elastic_law)

    def set_analytical_solution(self):
        x, y = sp.symbols("x, y")
        u0, l, h = self.u0, self.ell, self.h
        if self.elastic_law.name == "LinearCauchyElasticity":
            self.u_ana = (
                    u0 * ((1 + x / l) * (y / h) ** 2 * (1 - y / h) ** 2
                          * sin(2 * pi * x / l) * cos(3 * pi * y / h) +
                          x / l * 4 * y / h * (1 - y / h)),
                    u0 * x / l * (1 - x / l) * sin(2 * pi * y / h))
        elif self.elastic_law.name == "LinearGradientElasticity":
            self.u_ana = (
                    u0 * (50 * (1 - x / l) ** 2 * (x / l) ** 2 * (y / h) ** 2
                          * (1 - y / h) ** 2 * sin(2 * pi * x / l) *
                          cos(3 * pi * y / h) + 16 * (1 - y / h) ** 2
                          * (y / h) ** 2 * (3 * (x / l) ** 2 - 2 *
                          (x / l) ** 3)),
                    u0 * 50 * (1 - x / l) ** 2 * (x / l) ** 2 * (y / h) ** 2 *
                    (1 - y / h) ** 2 * sin(2 * pi * y / h) *
                    cos(3 * pi * x / l))

    def set_body_forces(self):
        self._body_forces = self._elastic_law.compute_body_forces(self.u_ana)

    def set_boundary_conditions(self):
        x, y = sp.symbols("x, y", real=True)
        u0, h = self.u0, self.h
        if self.elastic_law.name == "LinearCauchyElasticity":
            self._bcs = {'right': [(DisplacementBC.function_component, 0,
                                    u0 * 4 * y / h * (1 - y / h)),
                                   (DisplacementBC.fixed_component, 1, None)],
                         'top': [(DisplacementBC.fixed, None)],
                         'left': [(DisplacementBC.fixed, None)],
                         'bottom': [(DisplacementBC.fixed, None)]}
        elif self.elastic_law.name == "LinearGradientElasticity":
            self._bcs = {'right': [(DisplacementBC.function_component, 0,
                                    u0 * 16 * (1 - y / h) ** 2 * (y / h) ** 2),
                                   (DisplacementBC.fixed_component, 1, None),
                                   (DisplacementBC.fixed_gradient, None)],
                         'top': [(DisplacementBC.fixed, None),
                                 (DisplacementBC.fixed_gradient, None)],
                         'left': [(DisplacementBC.fixed, None),
                                  (DisplacementBC.fixed_gradient, None)],
                         'bottom': [(DisplacementBC.fixed, None),
                                    (DisplacementBC.fixed_gradient, None)]}

    def set_material_parameters(self):
        if self.elastic_law.name == "LinearCauchyElasticity":
            E = 400.
            nu = 0.4
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            mu = E / (2 * (1 + nu))
            self._material_parameters = lmbda, mu

        elif self.elastic_law.name == "LinearGradientElasticity":
            E = 400.
            nu = 0.4
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            mu = E / (2 * (1 + nu))
            c1 = c2 = c3 = c4 = c5 = 0.1
            self._material_parameters = lmbda, mu, c1, c2, c3, c4, c5

    def set_nondim_parameters(self):
        self._l_ref = self.ell
        self._u_ref = self.u0
        self._mat_param_ref = self.material_parameters[0]


class TensileTestOneDimensional(ElasticProblem):
    def __init__(self, N, domain, elastic_law):
        self._name = 'TensileTestOneDimensional'
        self.ell, self.h = domain[0][1], domain[1][1]
        self.u0 = self.ell / 100
        super().__init__(N, domain, elastic_law)

    def set_analytical_solution(self):
        assert hasattr(self, "material_parameters")
        lmbda, mu = self.material_parameters[slice(2)]
        nu = lmbda / (2 * (lmbda + mu))
        x, y = sp.symbols("x, y")
        u0, ell = self.u0, self.ell
        self.u_ana = (x / ell * u0, - nu / (1 - nu) * u0 / ell * y)

    def set_boundary_conditions(self):
        self._bcs = {'right': [(DisplacementBC.function_component, 0,
                                self.u0)],
                     'left': [(DisplacementBC.fixed_component, 0, None)],
                     'bottom': [(DisplacementBC.fixed_component, 1, None)]}

    def set_material_parameters(self):
        if self.elastic_law._name == "LinearCauchyElasticity":
            E = 400.
            nu = 0.4
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            mu = E / (2 * (1 + nu))
            self._material_parameters = lmbda, mu

        elif self.elastic_law._name == "LinearGradientElasticity":
            E = 400.
            nu = 0.4
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            mu = E / (2 * (1 + nu))
            c1 = c2 = c3 = c4 = c5 = 0.1
            self._material_parameters = lmbda, mu, c1, c2, c3, c4, c5

    def set_nondim_parameters(self):
        self._l_ref = self.ell
        self._u_ref = self.u0
        self._mat_param_ref = self.material_parameters[0]


class TensileTestClamped(ElasticProblem):
    def __init__(self, N, domain, elastic_law):
        self._name = 'TensileTestClamped'
        self.ell, self.h = domain[0][1], domain[1][1]
        self.u0 = self.ell / 100
        super().__init__(N, domain, elastic_law)

    def set_boundary_conditions(self):
        x, y = sp.symbols("x, y", real=True)
        if self.elastic_law.name == "LinearCauchyElasticity":
            self._bcs = {'right': [(DisplacementBC.function, (self.u0, 0.))],
                         'left': [(DisplacementBC.fixed, None)]}
        elif self.elastic_law.name == "LinearGradientElasticity":
            self._bcs = {'right': [(DisplacementBC.function, (self.u0, 0.))],
                         'left': [(DisplacementBC.fixed, None),
                                  (DisplacementBC.fixed_gradient_component, 0,
                                   None)]}

    def set_material_parameters(self):
        if self.elastic_law._name == "LinearCauchyElasticity":
            E = 400.
            nu = 0.4
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            mu = E / (2 * (1 + nu))
            self._material_parameters = lmbda, mu

        elif self.elastic_law._name == "LinearGradientElasticity":
            E = 400.
            nu = 0.4
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            mu = E / (2 * (1 + nu))
            c1 = c2 = c3 = c4 = c5 = 0.1
            self._material_parameters = lmbda, mu, c1, c2, c3, c4, c5

    def set_nondim_parameters(self):
        self._l_ref = self.ell
        self._u_ref = self.u0
        self._mat_param_ref = self.material_parameters[0]


class ShearTest(ElasticProblem):
    def __init__(self, N, domain, elastic_law):
        self._name = 'ShearTest'
        self.ell, self.h = domain[0][1], domain[1][1]
        self.u0 = self.ell / 10
        super().__init__(N, domain, elastic_law)

    def set_analytical_solution(self):
        assert hasattr(self, "material_parameters")
        x, y = sp.symbols("x, y")
        u0, h = self.u0, self.h
        if self.elastic_law.name == "LinearCauchyElasticity":
            lmbda, mu = self.material_parameters
            self.u_ana = (y / h * u0, 0.)
        elif self.elastic_law.name == "LinearGradientElasticity":
            lmbda, mu, c1, c2, c3, c4, c5 = self.material_parameters
            zeta = sp.sqrt((c1 + c4) / mu)
            A1 = u0 * sinh(h / zeta) / \
                (sinh(h / zeta) - h / zeta * cosh(h / zeta))
            A2 = - u0 / zeta * cosh(h / zeta) / \
                (sinh(h / zeta) - h / zeta * cosh(h / zeta))
            A3 = - zeta * A2
            A4 = - A1
            self.u_ana = (A1 + A2 * y + A3 * sinh(y / zeta) +
                          A4 * cosh(y / zeta), 0.)

    def set_boundary_conditions(self):
        if self.elastic_law.name == "LinearCauchyElasticity":
            self._bcs = {'top': [(DisplacementBC.function_component, 0,
                                  self.u0),
                                 (DisplacementBC.function_component, 1, 0.)],
                         'bottom': [(DisplacementBC.fixed, None)]}
        elif self.elastic_law.name == "LinearGradientElasticity":
            self._bcs = {'top': [(DisplacementBC.function_component, 0,
                                  self.u0),
                                 (DisplacementBC.function_component, 1, 0.)],
                         'bottom': [(DisplacementBC.fixed, None),
                                    (DisplacementBC.fixed_gradient_component,
                                     0, None)]}

    def set_material_parameters(self):
        if self.elastic_law._name == "LinearCauchyElasticity":
            E = 400.
            nu = 0.4
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            mu = E / (2 * (1 + nu))
            self._material_parameters = lmbda, mu

        elif self.elastic_law._name == "LinearGradientElasticity":
            E = 400.
            nu = 0.4
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            mu = E / (2 * (1 + nu))
            c1 = c2 = c3 = c4 = c5 = 0.1
            self._material_parameters = lmbda, mu, c1, c2, c3, c4, c5

    def set_nondim_parameters(self):
        self._l_ref = self.ell
        self._u_ref = self.u0
        self._mat_param_ref = self.material_parameters[0]


def test_dirichlet():
    N = (30, 30)
    domain = ((0., 10.), (0., 5.))
    print("Starting Dirichlet test ...")
    for elastic_law in (LinearCauchyElasticity(), LinearGradientElasticity()):
        DirichletProblem = DirichletTest(N, domain, elastic_law)

        DirichletProblem.solve()
        u_ana_dl = get_dimensionless_displacement(DirichletProblem.u_ana,
                                                  DirichletProblem._l_ref,
                                                  DirichletProblem._u_ref)

        error = compute_numerical_error(u_ana_dl, DirichletProblem.solution)
        DirichletProblem.postprocess()
        print(f'Error {elastic_law._name}:\t {error}\t N = {N}')
    print("Finished Dirichlet test!")


def test_tensile_test_one_dimensional():
    N = (30, 30)
    domain = ((0., 10.), (0., 5))
    print("Starting tensile test (one dimensional) ...")
    for elastic_law in (LinearCauchyElasticity(), LinearGradientElasticity()):
        TensileTest = TensileTestOneDimensional(N, domain, elastic_law)
        TensileTest.solve()
        u_ana_dl = get_dimensionless_displacement(TensileTest.u_ana,
                                                  TensileTest._l_ref,
                                                  TensileTest._u_ref)

        error = compute_numerical_error(u_ana_dl, TensileTest.solution)
        TensileTest.postprocess()
        print(f'Error {elastic_law._name}:\t {error}\t N = {N}')
    print("Finished tensile test (one dimensional)!")


def test_tensile_test_clamped():
    N = (30, 30)
    domain = ((0., 10.), (0., 5))
    print("Starting tensile test (clamped) ...")
    for elastic_law in (LinearCauchyElasticity(), LinearGradientElasticity()):
        TensileTest = TensileTestClamped(N, domain, elastic_law)
        TensileTest.solve()
        TensileTest.postprocess()
    print("Finished tensile test (clamped)!")


def test_shear_test():
    h = 0.1
    length_ratio = 20
    ell = h * length_ratio
    N = (round(length_ratio / 5) * 30, 30)
    domain = ((0., ell), (0., h))
    print("Starting shear test ...")
    for elastic_law in (LinearCauchyElasticity(), LinearGradientElasticity()):
        Shear = ShearTest(N, domain, elastic_law)
        Shear.solve()
        u_ana_dl = get_dimensionless_displacement(Shear.u_ana,
                                                  Shear._l_ref,
                                                  Shear._u_ref)

        error_center = sf.Array(Shear.solution.function_space(),
                                buffer=u_ana_dl)[0, round(N[0] / 2), :] - \
            Shear.solution.backward()[0, round(N[0] / 2), :]

        error = np.linalg.norm(error_center)
        Shear.postprocess()

        print(f'Error {elastic_law._name}:\t {error}\t N = {N}')
    print("Finished tensile test (clamped)!")


if __name__ == "__main__":
    test_dirichlet()
    test_tensile_test_one_dimensional()
    test_tensile_test_clamped()
    test_shear_test()
