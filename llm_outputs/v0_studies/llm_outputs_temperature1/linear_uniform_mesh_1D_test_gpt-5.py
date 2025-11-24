def test_basic_mesh_creation(fcn):
    """
    Test basic mesh creation with simple parameters.
    """
    import numpy as np
    (x_min, x_max, num_elements) = (0.0, 1.0, 4)
    result = fcn(x_min, x_max, num_elements)
    assert isinstance(result, tuple) and len(result) == 2
    (node_coords, element_connectivity) = result
    assert isinstance(node_coords, np.ndarray)
    assert isinstance(element_connectivity, np.ndarray)
    expected_nodes = np.linspace(x_min, x_max, num_elements + 1)
    assert node_coords.ndim == 1
    assert node_coords.shape == (num_elements + 1,)
    assert np.allclose(node_coords, expected_nodes)
    assert np.all(np.diff(node_coords) > 0)
    expected_conn = np.column_stack((np.arange(num_elements), np.arange(1, num_elements + 1)))
    assert element_connectivity.shape == (num_elements, 2)
    assert np.array_equal(element_connectivity, expected_conn)
    assert element_connectivity.dtype.kind in ('i', 'u')
    assert element_connectivity.min() == 0
    assert element_connectivity.max() == num_elements

def test_single_element_mesh(fcn):
    """
    Test edge case with only one element.
    """
    import numpy as np
    (x_min, x_max, num_elements) = (-2.5, 3.5, 1)
    result = fcn(x_min, x_max, num_elements)
    assert isinstance(result, tuple) and len(result) == 2
    (node_coords, element_connectivity) = result
    assert isinstance(node_coords, np.ndarray)
    assert node_coords.shape == (2,)
    assert np.allclose(node_coords, np.array([x_min, x_max]))
    assert isinstance(element_connectivity, np.ndarray)
    assert element_connectivity.shape == (1, 2)
    assert np.array_equal(element_connectivity, np.array([[0, 1]]))
    assert element_connectivity.dtype.kind in ('i', 'u')