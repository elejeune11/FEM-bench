def test_basic_mesh_creation(fcn):
    (x_min, x_max, num_elements) = (0.0, 1.0, 4)
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    assert len(node_coords) == num_elements + 1
    assert element_connectivity.shape == (num_elements, 2)
    assert np.allclose(node_coords, [0.0, 0.25, 0.5, 0.75, 1.0])
    assert np.array_equal(element_connectivity, [[0, 1], [1, 2], [2, 3], [3, 4]])

def test_single_element_mesh(fcn):
    (x_min, x_max, num_elements) = (0.0, 1.0, 1)
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    assert len(node_coords) == num_elements + 1
    assert element_connectivity.shape == (num_elements, 2)
    assert np.allclose(node_coords, [0.0, 1.0])
    assert np.array_equal(element_connectivity, [[0, 1]])