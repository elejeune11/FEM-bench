import numpy as np


def test_basic_mesh_creation(fcn: callable):
    """Test basic mesh creation with simple parameters."""
    x_min, x_max, num_elements = 0.0, 1.0, 4
    
    node_coords, element_connectivity = fcn(x_min, x_max, num_elements)
    
    # Check shapes
    assert node_coords.shape == (5,)  # num_elements + 1
    assert element_connectivity.shape == (4, 2)
    
    # Check node coordinates
    expected_nodes = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_array_almost_equal(node_coords, expected_nodes)
    
    # Check connectivity
    expected_connectivity = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    np.testing.assert_array_equal(element_connectivity, expected_connectivity)


def test_single_element_mesh(fcn: callable):
    """Test edge case with only one element."""
    x_min, x_max, num_elements = -1.0, 3.0, 1
    
    node_coords, element_connectivity = fcn(x_min, x_max, num_elements)
    
    # Check shapes
    assert node_coords.shape == (2,)
    assert element_connectivity.shape == (1, 2)
    
    # Check values
    expected_nodes = np.array([-1.0, 3.0])
    np.testing.assert_array_almost_equal(node_coords, expected_nodes)
    
    expected_connectivity = np.array([[0, 1]])
    np.testing.assert_array_equal(element_connectivity, expected_connectivity)


def fail_basic_mesh_creation(x_min: float, x_max: float, num_elements: int) -> np.ndarray:
    num_nodes = num_elements + 1
    node_coords = np.zeros((num_nodes))
    element_connectivity = np.zeros((num_elements, 2))
    return (node_coords, element_connectivity)


def fail_single_element_mesh(x_min: float, x_max: float, num_elements: int) -> np.ndarray:
    node_coords = np.asarray([x_min, x_min])
    element_connectivity = np.asarray([0, 1])
    return (node_coords, element_connectivity)


def linear_uniform_mesh_1D(x_min: float, x_max: float, num_elements: int) -> np.ndarray:
    """
    Generate a 1D linear mesh.

    Parameters:
        x_min (float): Start coordinate of the domain.
        x_max (float): End coordinate of the domain.
        num_elements (int): Number of linear elements.

    Returns:
        (node_coords, element_connectivity):
            - node_coords: 1D numpy array of node coordinates (shape: [num_nodes])
            - element_connectivity: 2D numpy array of element connectivity 
              (shape: [num_elements, 2]) with node indices per element
    """
    num_nodes = num_elements + 1
    node_coords = np.linspace(x_min, x_max, num_nodes)
    element_connectivity = np.zeros((num_elements, 2), dtype=int)

    for e in range(num_elements):
        element_connectivity[e, 0] = e
        element_connectivity[e, 1] = e + 1

    return (node_coords, element_connectivity)


def task_info():
    task_id = "linear_uniform_mesh_1D"
    task_short_description = "creates a 1D uniform mesh with node coordiantes and element connectivity"
    created_date = "2025-07-08"
    created_by = "elejeune11"
    main_fcn = linear_uniform_mesh_1D
    required_imports = ["import numpy as np", "import pytest"]
    fcn_dependencies = []
    reference_verification_inputs = [[0.0, 10.0, 10],
                                     [-5.0, 5.0, 7],
                                     [3.0, -1.0, 6]]
    test_cases = [{"test_code": test_basic_mesh_creation, "expected_failures": [fail_basic_mesh_creation]}, {"test_code": test_single_element_mesh, "expected_failures": [fail_single_element_mesh]}]
    return task_id, task_short_description, created_date, created_by, main_fcn, required_imports, fcn_dependencies, reference_verification_inputs, test_cases
