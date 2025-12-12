def test_no_load_self_contained(fcn):
    """Test zero displacement and zero reaction in a fixed-free bar with no external load.
    A 2-element bar with uniform material, zero body force, and one fixed end should return:
    """
    import numpy as np
    x_min = 0.0
    x_max = 1.0
    num_elements = 2
    material_regions = [{'coord_min': x_min, 'coord_max': x_max, 'E': 100000.0, 'A': 1.0}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': x_min, 'u_prescribed': 0.0}]
    neumann_bc_list = None
    n_gauss = 2
    result = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
    assert isinstance(result, dict)
    assert 'displacements' in result and 'reactions' in result and ('node_coords' in result) and ('reaction_nodes' in result)
    displacements = result['displacements']
    reactions = result['reactions']
    node_coords = result['node_coords']
    reaction_nodes = result['reaction_nodes']
    expected_coords = np.linspace(x_min, x_max, num_elements + 1)
    assert isinstance(displacements, np.ndarray)
    assert isinstance(reactions, np.ndarray)
    assert isinstance(node_coords, np.ndarray)
    assert isinstance(reaction_nodes, np.ndarray)
    assert displacements.shape == expected_coords.shape
    assert node_coords.shape == expected_coords.shape
    assert reactions.shape == (len(dirichlet_bc_list),)
    assert reaction_nodes.shape == (len(dirichlet_bc_list),)
    assert np.allclose(node_coords, expected_coords)
    assert np.allclose(displacements, np.zeros_like(expected_coords), atol=1e-12)
    assert np.allclose(reactions, np.zeros_like(reactions), atol=1e-12)
    idx = int(reaction_nodes[0])
    assert np.isclose(node_coords[idx], x_min)
    assert np.isclose(displacements[idx], 0.0)

def test_analytical_solution(fcn):
    """Test a non-zero displacement field against a known analytical solution.
    For a uniform bar on [0,1] with E=1, A=1, left end fixed (u=0) and a point load
    P=1 applied at the right end, the analytical displacement is u(x) = P*x/(E*A) = x.
    The reaction at the fixed left end should be -P to balance equilibrium.
    """
    import numpy as np
    x_min = 0.0
    x_max = 1.0
    num_elements = 4
    E = 1.0
    A = 1.0
    P = 1.0
    material_regions = [{'coord_min': x_min, 'coord_max': x_max, 'E': E, 'A': A}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': x_min, 'u_prescribed': 0.0}]
    neumann_bc_list = [{'x_location': x_max, 'load_mag': P}]
    n_gauss = 2
    result = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
    assert isinstance(result, dict)
    displacements = result['displacements']
    reactions = result['reactions']
    node_coords = result['node_coords']
    reaction_nodes = result['reaction_nodes']
    expected_coords = np.linspace(x_min, x_max, num_elements + 1)
    assert node_coords.shape == expected_coords.shape
    assert displacements.shape == expected_coords.shape
    assert np.allclose(node_coords, expected_coords)
    expected_displacements = P / (E * A) * expected_coords
    assert np.allclose(displacements, expected_displacements, atol=1e-09)
    total_point_load = sum((item['load_mag'] for item in neumann_bc_list)) if neumann_bc_list else 0.0
    total_body = 0.0
    assert np.allclose(np.sum(reactions) + total_point_load + total_body, 0.0, atol=1e-09)
    idx = int(reaction_nodes[0])
    assert np.isclose(node_coords[idx], x_min)
    assert np.isclose(reactions[0], -P, atol=1e-09)