def test_no_load_self_contained(fcn):
    """
    Test zero displacement and zero reaction in a fixed-free bar with no external load.
    A 2-element bar with uniform material, zero body force, and one fixed end should return:
    """
    x_min = 0.0
    x_max = 1.0
    num_elements = 2
    E = 100.0
    A = 1.0
    material_regions = [{'coord_min': x_min, 'coord_max': x_max, 'E': E, 'A': A}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': x_min, 'u_prescribed': 0.0}]
    neumann_bc_list = None
    n_gauss = 2
    result = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
    assert isinstance(result, dict)
    assert 'displacements' in result and 'reactions' in result
    assert 'node_coords' in result and 'reaction_nodes' in result
    n_nodes = num_elements + 1
    displacements = result['displacements']
    reactions = result['reactions']
    node_coords = result['node_coords']
    reaction_nodes = result['reaction_nodes']
    assert displacements.shape == (n_nodes,)
    assert node_coords.shape == (n_nodes,)
    assert reactions.shape == (1,)
    assert reaction_nodes.shape == (1,)
    idx_fixed_arr = np.where(np.isclose(node_coords, x_min))[0]
    assert idx_fixed_arr.size == 1
    idx_fixed = int(idx_fixed_arr[0])
    assert reaction_nodes[0] == idx_fixed
    assert np.allclose(displacements, 0.0)
    assert np.allclose(reactions, 0.0)
    assert np.isclose(displacements[idx_fixed], 0.0)

def test_analytical_solution(fcn):
    """
    Test a non-zero displacement field against a known analytical solution.
    Fixed-left bar with a point load P at the right end (Neumann) and zero body force.
    Analytical solution for uniform EA: u(x) = (P/(E*A)) * x. With E=2, A=5, P=10,
    u(x) = x. The FE solution should match nodal displacements exactly.
    """
    x_min = 0.0
    x_max = 3.0
    num_elements = 6
    E = 2.0
    A = 5.0
    P = 10.0
    material_regions = [{'coord_min': x_min, 'coord_max': x_max, 'E': E, 'A': A}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': x_min, 'u_prescribed': 0.0}]
    neumann_bc_list = [{'x_location': x_max, 'load_mag': P}]
    n_gauss = 2
    result = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
    node_coords = result['node_coords']
    displacements = result['displacements']
    slope = P / (E * A)
    expected_u = slope * node_coords
    assert displacements.shape == node_coords.shape
    assert np.allclose(displacements, expected_u, rtol=1e-12, atol=1e-12)
    assert np.any(np.abs(displacements) > 0.0)