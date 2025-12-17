def test_basic_mesh_creation(fcn):
    """Test basic 1D uniform mesh creation for correctness.
    This test verifies that the provided mesh-generation function produces
    the expected node coordinates and element connectivity for a simple
    1D domain with uniform spacing.
    """
    x_min = 0.0
    x_max = 1.0
    num_elements = 4
    result = fcn(x_min, x_max, num_elements)
    assert isinstance(result, tuple) and len(result) == 2
    node_coords, element_connectivity = result
    tol = 1e-12
    assert len(node_coords) == num_elements + 1
    assert abs(node_coords[0] - x_min) <= tol
    assert abs(node_coords[-1] - x_max) <= tol
    spacing = (x_max - x_min) / num_elements
    for k in range(len(node_coords)):
        expected = x_min + k * spacing
        assert abs(node_coords[k] - expected) <= tol
        if k < len(node_coords) - 1:
            assert node_coords[k] < node_coords[k + 1]
    assert len(element_connectivity) == num_elements
    for i in range(num_elements):
        pair = element_connectivity[i]
        assert len(pair) == 2
        a, b = (int(pair[0]), int(pair[1]))
        assert a == i
        assert b == i + 1
        assert 0 <= a < len(node_coords)
        assert 0 <= b < len(node_coords)
        assert a < b

def test_single_element_mesh(fcn):
    """Test mesh generation for the edge case of a single 1D element.
    This test checks that the mesh-generation function correctly handles
    the minimal valid case of one linear element spanning a domain from
    x_min to x_max. It ensures the function properly computes both node
    coordinates and connectivity for this degenerate case.
    """
    x_min = -2.0
    x_max = 3.0
    num_elements = 1
    result = fcn(x_min, x_max, num_elements)
    assert isinstance(result, tuple) and len(result) == 2
    node_coords, element_connectivity = result
    tol = 1e-12
    assert len(node_coords) == 2
    assert abs(node_coords[0] - x_min) <= tol
    assert abs(node_coords[1] - x_max) <= tol
    assert node_coords[0] < node_coords[1]
    assert len(element_connectivity) == 1
    pair = element_connectivity[0]
    assert len(pair) == 2
    a, b = (int(pair[0]), int(pair[1]))
    assert a == 0
    assert b == 1
    assert 0 <= a < len(node_coords)
    assert 0 <= b < len(node_coords)
    assert a < b