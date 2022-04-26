from shenfun_elasticity.solver.elastic_problem import ElasticProblem
from shenfun_elasticity.solver.elastic_problem import DisplacementBC, TractionBC
from shenfun_elasticity.solver.elastic_law import LinearCauchyElasticity, \
    LinearGradientElasticity
from shenfun_elasticity.solver.utilities import compute_numerical_error, \
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
        x, y = sp.symbols('x, y')
        u0, l, h = self.u0, self.ell, self.h
        if self.elastic_law.name == 'LinearCauchyElasticity':
            self.u_ana = (
                    u0 * ((1 + x / l) * (y / h) ** 2 * (1 - y / h) ** 2
                          * sin(2 * pi * x / l) * cos(3 * pi * y / h) +
                          x / l * 4 * y / h * (1 - y / h)),
                    u0 * x / l * (1 - x / l) * sin(2 * pi * y / h))
        elif self.elastic_law._name == 'LinearGradientElasticity':
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
        x, y = sp.symbols('x, y', real=True)
        u0, h = self.u0, self.h
        if self.elastic_law.name == 'LinearCauchyElasticity':
            self._bcs = {'right': [(DisplacementBC.function_component, 0,
                                    u0 * 4 * y / h * (1 - y / h)),
                                   (DisplacementBC.fixed_component, 1, None)],
                         'top': [(DisplacementBC.fixed, None)],
                         'left': [(DisplacementBC.fixed, None)],
                         'bottom': [(DisplacementBC.fixed, None)]}
        elif self.elastic_law.name == 'LinearGradientElasticity':
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
        if self.elastic_law.name == 'LinearCauchyElasticity':
            E = 400.
            nu = 0.4
            lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            mu = E / (2.0 * (1.0 + nu))
            self._material_parameters = lmbda, mu

        elif self.elastic_law.name == 'LinearGradientElasticity':
            E = 400.
            nu = 0.4
            lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            mu = E / (2.0 * (1.0 + nu))
            c1 = c2 = c3 = c4 = c5 = 0.1
            self._material_parameters = lmbda, mu, c1, c2, c3, c4, c5

    def set_nondim_parameters(self):
        self._l_ref = self.ell
        self._u_ref = self.u0
        self._mat_param_ref = self.material_parameters[0]


class TensileTestOneDimensional(ElasticProblem):
    def __init__(self, N, domain, elastic_law, name_suffix):
        self._name_suffix = name_suffix
        self._name = 'TensileTestOneDimensional' + self._name_suffix
        self.ell, self.h = domain[0][1], domain[1][1]
        if self._name_suffix == 'DisplacementControlled':
            self.u0 = self.ell / 100
        elif self._name_suffix == 'TractionControlled':
            self.sigma0 = 10
        else:
            raise ValueError()
        super().__init__(N, domain, elastic_law)

    def set_analytical_solution(self):
        assert hasattr(self, 'material_parameters')
        lmbda, mu = self.material_parameters[slice(2)]
        nu = lmbda / (2 * (lmbda + mu))
        x, y = sp.symbols('x, y')
        if self._name_suffix == 'DisplacementControlled':
            assert hasattr(self, 'u0')
            assert hasattr(self, 'ell')
            u0, ell = self.u0, self.ell
            nu_hat = nu / (1 - nu)
            self.u_ana = (x / ell * u0, - nu_hat * u0 / ell * y)
        elif self._name_suffix == 'TractionControlled':
            assert hasattr(self, 'sigma0')
            E = mu * (3 * lmbda + 2 * mu) / (lmbda + mu)
            E_hat = E / (1 - nu ** 2)
            nu_hat = nu / (1 - nu)
            self.u_ana = (self.sigma0 / E_hat * x,
                          - nu_hat * self.sigma0 / E_hat * y)

    def set_boundary_conditions(self):
        if self._name_suffix == 'DisplacementControlled':
            self._bcs = {'right': [(DisplacementBC.function_component, 0,
                                    self.u0)],
                         'left': [(DisplacementBC.fixed_component, 0, None)],
                         'bottom': [(DisplacementBC.fixed_component, 1, None)]}
        elif self._name_suffix == 'TractionControlled':
            self._bcs = {'right': [(TractionBC.function_component, 0,
                                    self.sigma0)],
                         'left': [(DisplacementBC.fixed_component, 0, None)],
                         'bottom': [(DisplacementBC.fixed_component, 1, None)]}

    def set_material_parameters(self):
        if self.elastic_law.name == 'LinearCauchyElasticity':
            E = 400.
            nu = 0.4
            lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            mu = E / (2.0 * (1.0 + nu))
            self._material_parameters = lmbda, mu

        elif self.elastic_law.name == 'LinearGradientElasticity':
            E = 400.
            nu = 0.4
            lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            mu = E / (2.0 * (1.0 + nu))
            c1 = c2 = c3 = c4 = c5 = 0.1
            self._material_parameters = lmbda, mu, c1, c2, c3, c4, c5

    def set_nondim_parameters(self):
        self._l_ref = self.ell
        if hasattr(self, 'u0'):
            self._u_ref = self.u0
        else:
            self._u_ref = 1
        self._mat_param_ref = self.material_parameters[0]


class TensileTestClamped(ElasticProblem):
    def __init__(self, N, domain, elastic_law):
        self._name = 'TensileTestClamped'
        self.ell, self.h = domain[0][1], domain[1][1]
        self.u0 = self.ell / 100
        super().__init__(N, domain, elastic_law)

    def set_boundary_conditions(self):
        x, y = sp.symbols('x, y', real=True)
        if self.elastic_law.name == 'LinearCauchyElasticity':
            self._bcs = {'right': [(DisplacementBC.function, (self.u0, 0.))],
                         'left': [(DisplacementBC.fixed, None)]}
        elif self.elastic_law.name == 'LinearGradientElasticity':
            self._bcs = {'right': [(DisplacementBC.function, (self.u0, 0.))],
                         'left': [(DisplacementBC.fixed, None),
                                  (DisplacementBC.fixed_gradient_component, 0,
                                   None)]}

    def set_material_parameters(self):
        if self.elastic_law.name == 'LinearCauchyElasticity':
            E = 400.
            nu = 0.4
            lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            mu = E / (2.0 * (1.0 + nu))
            self._material_parameters = lmbda, mu

        elif self.elastic_law.name == 'LinearGradientElasticity':
            E = 400.
            nu = 0.4
            lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            mu = E / (2.0 * (1.0 + nu))
            c1 = c2 = c3 = c4 = c5 = 0.1
            self._material_parameters = lmbda, mu, c1, c2, c3, c4, c5

    def set_nondim_parameters(self):
        self._l_ref = self.ell
        self._u_ref = self.u0
        self._mat_param_ref = self.material_parameters[0]


class ShearTest(ElasticProblem):
    def __init__(self, N, domain, elastic_law, name_suffix):
        self._name_suffix = name_suffix
        self._name = 'ShearTest' + self._name_suffix
        self.ell, self.h = domain[0][1], domain[1][1]
        if self._name_suffix == 'DisplacementControlled':
            self.u0 = self.ell / 10
        elif self._name_suffix == 'TractionControlled':
            self.tau0 = 10
        else:
            raise NotImplementedError()
        super().__init__(N, domain, elastic_law)

    def set_analytical_solution(self):
        assert hasattr(self, 'material_parameters')
        x, y = sp.symbols('x, y')
        if self.elastic_law.name == 'LinearCauchyElasticity':
            lmbda, mu = self.material_parameters
            if self._name_suffix == 'DisplacementControlled':
                self.u_ana = (y / self.h * self.u0, 0.)
            elif self._name_suffix == 'TractionControlled':
                assert hasattr(self, 'tau0')
                self.u_ana = (self.tau0 / mu * y, 0.)
            else:
                raise NotImplementedError()
        elif self.elastic_law.name == 'LinearGradientElasticity':
            lmbda, mu, c1, c2, c3, c4, c5 = self.material_parameters
            zeta = sp.sqrt((c1 + c4) / mu)
            h = self.h
            if self._name_suffix == 'DisplacementControlled':
                u0 = self.u0
                A1 = u0 * sinh(h / zeta) / \
                    (sinh(h / zeta) - h / zeta * cosh(h / zeta))
                A2 = - u0 / zeta * cosh(h / zeta) / \
                    (sinh(h / zeta) - h / zeta * cosh(h / zeta))
                A3 = - zeta * A2
                A4 = - A1
                self.u_ana = (A1 + A2 * y + A3 * sinh(y / zeta) +
                              A4 * cosh(y / zeta), 0.)
            elif self._name_suffix == 'TractionControlled':
                assert hasattr(self, 'tau0')
                xi = 1. / (mu * cosh(h / zeta) * zeta ** 2 \
                           - cosh(h / zeta) ** 2 * (mu * zeta ** 2 - (c1 + c4)) \
                               + sinh(h / zeta) ** 2 * (mu * zeta ** 2 - (c1 + c4)))
                A1 = - xi * zeta ** 3 * self.tau0 * sinh(h / zeta)
                A2 = xi * zeta ** 2 * self.tau0 * cosh(h / zeta)
                A3 = - A2 * zeta
                A4 = - A1
                self.u_ana = (A1 + A2 * y + A3 * sinh(y / zeta) +
                              A4 * cosh(y / zeta), 0.)
            else:
                raise NotImplementedError()

    def set_boundary_conditions(self):
        if self.elastic_law.name == 'LinearCauchyElasticity':
            if self._name_suffix == 'DisplacementControlled':
                self._bcs = {'top': [(DisplacementBC.function_component, 0, self.u0),
                                     (DisplacementBC.function_component, 1, 0.)],
                             'bottom': [(DisplacementBC.fixed, None)]}
            elif self._name_suffix == 'TractionControlled':
                self._bcs = {'top': [(TractionBC.function_component, 0, self.tau0),
                                     (DisplacementBC.function_component, 1, 0.)],
                             'bottom': [(DisplacementBC.fixed, None)]}
        elif self.elastic_law.name == 'LinearGradientElasticity':
            if self._name_suffix == 'DisplacementControlled':
                self._bcs = {'top': [(DisplacementBC.function_component, 0, self.u0),
                                     (DisplacementBC.function_component, 1, 0.)],
                             'bottom': [(DisplacementBC.fixed, None),
                                        (DisplacementBC.fixed_gradient_component,
                                         0, None)]}
            elif self._name_suffix == 'TractionControlled':
                self._bcs = {'top': [(TractionBC.function_component, 0, self.tau0),
                                     (DisplacementBC.function_component, 1, 0.)],
                             'bottom': [(DisplacementBC.fixed, None),
                                        (DisplacementBC.fixed_gradient_component,
                                         0, None)]}
            else:
                raise NotImplementedError()

    def set_material_parameters(self):
        if self.elastic_law.name == 'LinearCauchyElasticity':
            E = 400.
            nu = 0.4
            lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            mu = E / (2.0 * (1.0 + nu))
            self._material_parameters = lmbda, mu

        elif self.elastic_law.name == 'LinearGradientElasticity':
            E = 400.
            nu = 0.4
            lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            mu = E / (2.0 * (1.0 + nu))
            c1 = c2 = c3 = c4 = c5 = 0.1
            self._material_parameters = lmbda, mu, c1, c2, c3, c4, c5

    def set_nondim_parameters(self):
        self._l_ref = self.ell
        if hasattr(self, 'u0'):
            self._u_ref = self.u0
        else:
            self._u_ref = 1
        self._mat_param_ref = self.material_parameters[0]


def test_dirichlet():
    N = (30, 30)
    domain = ((0., 10.), (0., 5.))
    print('Starting Dirichlet test ...')
    for elastic_law in (LinearCauchyElasticity(), LinearGradientElasticity()):
        DirichletProblem = DirichletTest(N, domain, elastic_law)

        DirichletProblem.solve()
        u_ana_dl = get_dimensionless_displacement(DirichletProblem.u_ana,
                                                  DirichletProblem._l_ref,
                                                  DirichletProblem._u_ref)

        error = compute_numerical_error(u_ana_dl, DirichletProblem.solution)
        assert error < 1e-15, 'Error tolerance not achieved'
        DirichletProblem.postprocess()
        print(f'Error {elastic_law._name}:\t {error}\t N = {N}')
    print('Finished Dirichlet test!')


def test_tensile_test_one_dimensional():
    N = (5, 5)
    domain = ((0., 10.), (0., 10))
    print('Starting tensile test (one dimensional) ...')
    for elastic_law in (LinearCauchyElasticity(), LinearGradientElasticity()):
        for name_suffix in ('DisplacementControlled', 'TractionControlled'):
            TensileTest = TensileTestOneDimensional(N, domain, elastic_law,
                                                    name_suffix)
            TensileTest.solve()
            u_ana_dl = get_dimensionless_displacement(TensileTest.u_ana,
                                                      TensileTest._l_ref,
                                                      TensileTest._u_ref)
            error = compute_numerical_error(u_ana_dl, TensileTest.solution)
            assert error < 1e-15, 'Error tolerance not achieved'
            TensileTest.postprocess()
            print(f'Error {elastic_law._name} ({name_suffix}):\t {error}\t N = {N}')
    print('Finished tensile test (one dimensional)!')


def test_tensile_test_clamped():
    N = (30, 30)
    domain = ((0., 10.), (0., 5))
    print('Starting tensile test (clamped) ...')
    for elastic_law in (LinearCauchyElasticity(), LinearGradientElasticity()):
        TensileTest = TensileTestClamped(N, domain, elastic_law)
        TensileTest.solve()
        TensileTest.postprocess()
    print('Finished tensile test (clamped)!')


def test_shear_test():
    h = 0.1
    length_ratio = 20
    ell = h * length_ratio
    N = (round(length_ratio / 5) * 30, 30)
    domain = ((0., ell), (0., h))
    print('Starting shear test ...')
    for elastic_law in (LinearCauchyElasticity(), LinearGradientElasticity()):
        for name_suffix in ('DisplacementControlled', 'TractionControlled'):
            Shear = ShearTest(N, domain, elastic_law, name_suffix)
            Shear.solve()
            u_ana_dl = get_dimensionless_displacement(Shear.u_ana,
                                                      Shear._l_ref,
                                                      Shear._u_ref)

            error_center = sf.Array(Shear.solution.function_space(),
                                    buffer=u_ana_dl)[0, round(N[0] / 2), :] - \
                Shear.solution.backward()[0, round(N[0] / 2), :]

            error = np.linalg.norm(error_center)
            assert error < 1e-5, 'Error tolerance not achieved'
            Shear.postprocess()

            print(f'Error {elastic_law._name} ({name_suffix}):\t {error}\t N = {N}')
    print('Finished tensile test (clamped)!')


if __name__ == '__main__':
    test_dirichlet()
    test_tensile_test_one_dimensional()
    test_tensile_test_clamped()
    test_shear_test()
