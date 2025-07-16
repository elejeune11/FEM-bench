import pytest
import numpy as np
from typing import List, Dict, Callable, Optional

def test_no_load_self_contained(fcn):
    """Test zero displacement and zero reaction in a fixed-free bar with no external load.

    A 2-element bar with uniform material, zero body force, and one fixed end should return:
    - Zero displacement at all nodes
    - Zero reaction at the fixed node
    - Correct output shapes and boundary condition enforcement
    """
    x_min = 0.0
    x_max = 2.0
    num_elements = 2
    material_regions = [{"coord_min": 0.0, "coord_max": 2.0, "E": 1.0, "A": 1.0}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{"x_location": 0.0, "u_prescribed": 0.0}]
    neumann_bc_list = None
    n_gauss = 2

    results = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)

    assert np.allclose(results["displacements"], np.array([0.0, 0.0, 0.0]))
    assert np.allclose(results["reactions"], np.array([0.0]))
    assert np.allclose(results["node_coords"], np.array([0.0, 1.0, 2.0]))
    assert np.allclose(results["reaction_nodes"], np.array([0]))


def test_uniform_extension_analytical_self_contained(fcn):
    """Test displacement field against a known analytical solution."""
    x_min = 0.0
    x_max = 1.0
    num_elements = 10
    material_regions = [{"coord_min": 0.0, "coord_max": 1.0, "E": 10.0, "A": 1.0}]
    body_force_fn = lambda x: 1.0  # Constant body force
    dirichlet_bc_list = [{"x_location": 0.0, "u_prescribed": 0.0}]
    neumann_bc_list = None
    n_gauss = 2

    results = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)

    node_coords = results["node_coords"]
    displacements = results["displacements"]

    # Analytical solution: u(x) = (1/2E) * x^2 + C.  BC: u(0) = 0  => C = 0.
    analytical_displacements = (1.0 / (2.0 * 10.0)) * node_coords**2

    assert np.allclose(displacements, analytical_displacements, rtol=1e-5)
    assert results["reactions"].shape == (1,)
    assert results["node_coords"].shape == (num_elements + 1,)