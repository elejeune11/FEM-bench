import numpy as np


def FEM_1D_uniform_mesh_CC0_H0_T0(x_min: float, x_max: float, num_elements: int) -> np.ndarray:
    """
    Generate a 1D linear uniform mesh with evenly spaced nodes.

    Parameters:
        x_min (float): Start coordinate of the domain.
        x_max (float): End coordinate of the domain.
        num_elements (int): Number of linear elements.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            node_coords: 1D array of node coordinates (shape: [num_nodes]).
            element_connectivity: 2D array of element connectivity 
                (shape: [num_elements, 2]), where each row lists the two node indices of an element.

    Notes:
        - Node coordinates increase uniformly from x_min to x_max.
        - Elements are numbered sequentially from 0 to num_elements - 1.
        - Node indices increase monotonically along the domain.
    """
    num_nodes = num_elements + 1
    node_coords = np.linspace(x_min, x_max, num_nodes)
    element_connectivity = np.zeros((num_elements, 2), dtype=int)

    for e in range(num_elements):
        element_connectivity[e, 0] = e
        element_connectivity[e, 1] = e + 1

    return (node_coords, element_connectivity)


def test_basic_mesh_creation(fcn: callable):
    """
    Test basic 1D uniform mesh creation for correctness.

    This test verifies that the provided mesh-generation function produces
    the expected node coordinates and element connectivity for a simple
    1D domain with uniform spacing.
    """
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
    """
    Test mesh generation for the edge case of a single 1D element.

    This test checks that the mesh-generation function correctly handles
    the minimal valid case of one linear element spanning a domain from
    x_min to x_max. It ensures the function properly computes both node
    coordinates and connectivity for this degenerate case.
    """
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


def task_info():
    task_id = "FEM_1D_uniform_mesh_CC0_H0_T0"
    task_short_description = "creates a 1D uniform mesh with node coordiantes and element connectivity"
    created_date = "2025-07-08"
    created_by = "elejeune11"
    main_fcn = FEM_1D_uniform_mesh_CC0_H0_T0
    required_imports = ["import numpy as np", "import pytest"]
    fcn_dependencies = []
    reference_verification_inputs = [[0.0, 10.0, 10],
                                     [-5.0, 5.0, 7],
                                     [3.0, -1.0, 6]]
    test_cases = [{"test_code": test_basic_mesh_creation, "expected_failures": [fail_basic_mesh_creation]}, {"test_code": test_single_element_mesh, "expected_failures": [fail_single_element_mesh]}]
    return {
        "task_id": task_id,
        "task_short_description": task_short_description,
        "created_date": created_date,
        "created_by": created_by,
        "main_fcn": main_fcn,
        "required_imports": required_imports,
        "fcn_dependencies": fcn_dependencies,
        "reference_verification_inputs": reference_verification_inputs,
        "test_cases": test_cases,
        # "python_version": "version_number",
        # "package_versions": {"numpy": "version_number", },
    }