"""
    test_distribution
    ~~~~~~~~~~~~~~~~~
"""
import numpy as np
import pytest
from mrtool import Gaussian, Laplace, Uniform


@pytest.mark.parametrize('params', [[0.0, 1.0],
                                    (0.0, 1.0),
                                    np.array([0.0, 1.0]),
                                    np.array([[0.0, 1.0]])])
def test_gaussian(params):
    distr = Gaussian(params)
    assert np.allclose(distr.params, np.array([[0.0, 1.0]]))


@pytest.mark.parametrize(('params', 'result'), [([0.0, 1.0], True),
                                                ([[0.0, 1.0], [0.0, np.inf]], True),
                                                ([0.0, np.inf], False),
                                                ([[0.0, np.inf], [1.0, np.inf]], False)])
def test_gaussian_is_informative(params, result):
    distr = Gaussian(params)
    assert distr.is_informative() == result


@pytest.mark.parametrize(('params', 'result'), [([[0.0, 1.0], [0.0, 1.0]], True),
                                                ([[0.0, 1.0]], True),
                                                ([[0.0, np.inf], [1.0, np.inf]], True),
                                                ([[0.0, 1.0], [1.0, 1.0]], False),
                                                ([[0.0, 1.0], [0.0, 2.0]], False)])
def test_gaussian_is_identical(params, result):
    distr = Gaussian(params)
    assert distr.is_identical() == result


def test_gaussian_change_dim():
    distr = Gaussian(np.array([0.0, 1.0]))
    distr.change_dim(5)
    assert np.all(distr.mean == 0.0)
    assert np.all(distr.std == 1)
    assert distr.params.shape == (5, 2)

    distr = Gaussian(np.array([[0.0, np.inf],
                               [1.0, np.inf]]))
    distr.change_dim(5)
    assert np.all(distr.mean == np.array([0.0, 1.0, 1.0, 1.0, 1.0]))
    assert np.all(np.isinf(distr.std))
    assert distr.params.shape == (5, 2)

    distr = Gaussian(np.array([[0.0, 1.0],
                               [1.0, 1.0]]))
    with pytest.raises(ValueError):
        distr.change_dim(5)


@pytest.mark.parametrize('params', [[0.0, 1.0],
                                    (0.0, 1.0),
                                    np.array([0.0, 1.0]),
                                    np.array([[0.0, 1.0]])])
def test_laplace(params):
    distr = Laplace(params)
    assert np.allclose(distr.params, np.array([[0.0, 1.0]]))


@pytest.mark.parametrize(('params', 'result'), [([0.0, 1.0], True),
                                                ([[0.0, 1.0], [0.0, np.inf]], True),
                                                ([0.0, np.inf], False),
                                                ([[0.0, np.inf], [1.0, np.inf]], False)])
def test_laplace_is_informative(params, result):
    distr = Laplace(params)
    assert distr.is_informative() == result


@pytest.mark.parametrize(('params', 'result'), [([[0.0, 1.0], [0.0, 1.0]], True),
                                                ([[0.0, 1.0]], True),
                                                ([[0.0, np.inf], [1.0, np.inf]], True),
                                                ([[0.0, 1.0], [1.0, 1.0]], False),
                                                ([[0.0, 1.0], [0.0, 2.0]], False)])
def test_laplace_is_identical(params, result):
    distr = Laplace(params)
    assert distr.is_identical() == result


def test_laplace_change_dim():
    distr = Laplace(np.array([0.0, 1.0]))
    distr.change_dim(5)
    assert np.all(distr.mean == 0.0)
    assert np.all(distr.std == 1)
    assert distr.params.shape == (5, 2)

    distr = Laplace(np.array([[0.0, np.inf],
                              [1.0, np.inf]]))
    distr.change_dim(5)
    assert np.all(distr.mean == np.array([0.0, 1.0, 1.0, 1.0, 1.0]))
    assert np.all(np.isinf(distr.std))
    assert distr.params.shape == (5, 2)

    distr = Laplace(np.array([[0.0, 1.0],
                              [1.0, 1.0]]))
    with pytest.raises(ValueError):
        distr.change_dim(5)


@pytest.mark.parametrize('params', [[0.0, 1.0],
                                    (0.0, 1.0),
                                    np.array([0.0, 1.0]),
                                    np.array([[0.0, 1.0]])])
def test_uniform(params):
    distr = Uniform(params)
    assert np.allclose(distr.params, np.array([[0.0, 1.0]]))


@pytest.mark.parametrize(('params', 'result'), [([0.0, 1.0], True),
                                                ([[0.0, 1.0], [-np.inf, np.inf]], True),
                                                ([-np.inf, np.inf], False),
                                                ([[-np.inf, np.inf], [-np.inf, np.inf]], False)])
def test_uniform_is_informative(params, result):
    distr = Uniform(params)
    assert distr.is_informative() == result


@pytest.mark.parametrize(('params', 'result'), [([[0.0, 1.0], [0.0, 1.0]], True),
                                                ([[0.0, 1.0]], True),
                                                ([[-np.inf, np.inf], [-np.inf, np.inf]], True),
                                                ([[0.0, 1.0], [1.0, 1.0]], False),
                                                ([[0.0, 1.0], [0.0, 2.0]], False)])
def test_uniform_is_identical(params, result):
    distr = Uniform(params)
    assert distr.is_identical() == result


def test_uniform_change_dim():
    distr = Uniform(np.array([0.0, 1.0]))
    distr.change_dim(5)
    assert np.all(distr.lb == 0.0)
    assert np.all(distr.ub == 1)
    assert distr.params.shape == (5, 2)

    distr = Uniform(np.array([[-np.inf, np.inf],
                              [-np.inf, np.inf]]))
    distr.change_dim(5)
    assert np.all(np.isneginf(distr.lb))
    assert np.all(np.isposinf(distr.ub))
    assert distr.params.shape == (5, 2)

    distr = Uniform(np.array([[0.0, np.inf],
                              [1.0, np.inf]]))
    with pytest.raises(ValueError):
        distr.change_dim(5)
