import numpy as np

def test_basic_mesh_creation(fcn):
    """
    Test basic mesh creation with simple parameters.
    """
    x_min = 0.0
    x_max = 1.0
    num_elements = 2

    node_coords, element_connectivity = fcn(x_min, x_max, num_elements)

    assert isinstance(node_coords, np.ndarray)
    assert isinstance(element_connectivity, np.ndarray)
    assert node_coords.shape == (num_elements + 1,)
    assert element_connectivity.shape == (num_elements, 2)

    expected_nodes = np.linspace(x_min, x_max, num_elements + 1)
    assert np.allclose(node_coords, expected_nodes)

    expected_connectivity = np.array([[0,1],[1,2]])
    assert np.array_equal(element_connectivity, expected_connectivity)

def test_single_element_mesh(fcn):
    """
    Test edge case with only one element.
    """
    x_min = 0.0
    x_max = 1.0
    num_elements = 1

    node_coords, element_connectivity = fcn(x_min, x_max, num_elements)

    assert isinstance(node_coords, np.ndarray)
    assert isinstance(element_connectivity, np.ndarray)
    assert node_coords.shape == (2,)
    assert element_connectivity.shape == (1, 2)

    expected_nodes = np.array([x_min, x_max])
    assert np.allclose(node_coords, expected_nodes)

    expected_connectivity = np.array([[0,1]])
    assert np.array_equal(element_connectivity, expected_connectivity)