import numpy as np
import pytest

def test_basic_mesh_creation(fcn):
    """Test basic mesh creation with simple parameters."""
    x_min = 0.0
    x_max = 1.0
    num_elements = 4
    node_coords, element_connectivity = fcn(x_min, x_max, num_elements)

    assert np.allclose(node_coords, np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
    assert np.array_equal(element_connectivity, np.array([[0, 1], [1, 2], [2, 3], [3, 4]]))

def test_single_element_mesh(fcn):
    """Test edge case with only one element."""
    x_min = 0.0
    x_max = 1.0
    num_elements = 1
    node_coords, element_connectivity = fcn(x_min, x_max, num_elements)

    assert np.allclose(node_coords, np.array([0.0, 1.0]))
    assert np.array_equal(element_connectivity, np.array([[0, 1]]))