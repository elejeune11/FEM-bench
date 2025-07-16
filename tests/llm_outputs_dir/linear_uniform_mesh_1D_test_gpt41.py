def test_basic_mesh_creation(fcn):
    """
    Test basic mesh creation with simple parameters.
    """
    import numpy as np
    x_min = 0.0
    x_max = 1.0
    num_elements = 4
    node_coords, element_connectivity = fcn(x_min, x_max, num_elements)
    assert isinstance(node_coords, np.ndarray)
    assert isinstance(element_connectivity, np.ndarray)
    assert node_coords.shape == (num_elements + 1,)
    expected_coords = np.linspace(x_min, x_max, num_elements + 1)
    assert np.allclose(node_coords, expected_coords)
    assert element_connectivity.shape == (num_elements, 2)
    for i in range(num_elements):
        assert np.array_equal(element_connectivity[i], [i, i + 1])

def test_single_element_mesh(fcn):
    """
    Test edge case with only one element.
    """
    import numpy as np
    x_min = -2.5
    x_max = 3.5
    num_elements = 1
    node_coords, element_connectivity = fcn(x_min, x_max, num_elements)
    assert isinstance(node_coords, np.ndarray)
    assert isinstance(element_connectivity, np.ndarray)
    assert node_coords.shape == (2,)
    expected_coords = np.array([x_min, x_max])
    assert np.allclose(node_coords, expected_coords)
    assert element_connectivity.shape == (1, 2)
    assert np.array_equal(element_connectivity[0], [0, 1])