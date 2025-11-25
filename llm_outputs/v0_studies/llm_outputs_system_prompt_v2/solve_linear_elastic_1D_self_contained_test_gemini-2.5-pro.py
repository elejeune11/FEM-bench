def test_no_load_self_contained(fcn: Callable):
    """Test zero displacement and zero reaction in a fixed-free bar with no external load.
    A 2-element bar with uniform material, zero body force, and one fixed end should return:
    """
    x_min = 0.0
    x_max = 1.0
    num_elements = 2
    n_nodes = num_elements + 1
    material_regions = [{'coord_min': x_min, 'coord_max': x_max, 'E': 1.0, 'A': 1.0}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': x_min, 'u_prescribed': 0.0}]
    neumann_bc_list = []
    n_gauss = 2
    result = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
    expected_displacements = np.zeros(n_nodes)
    expected_reactions = np.zeros(len(dirichlet_bc_list))
    expected_node_coords = np.linspace(x_min, x_max, n_nodes)
    expected_reaction_nodes = np.array([0])
    assert 'displacements' in result
    assert 'reactions' in result
    assert 'node_coords' in result
    assert 'reaction_nodes' in result
    assert result['displacements'].shape == (n_nodes,)
    assert result['reactions'].shape == (len(dirichlet_bc_list),)
    assert result['node_coords'].shape == (n_nodes,)
    assert result['reaction_nodes'].shape == (len(dirichlet_bc_list),)
    np.testing.assert_allclose(result['displacements'], expected_displacements, atol=1e-12)
    np.testing.assert_allclose(result['reactions'], expected_reactions, atol=1e-12)
    np.testing.assert_allclose(result['node_coords'], expected_node_coords)
    np.testing.assert_array_equal(result['reaction_nodes'], expected_reaction_nodes)

def test_uniform_extension_analytical_self_contained(fcn: Callable):
    """Test displacement field against a known analytical solution."""
    x_min = 0.0
    x_max = 2.0
    num_elements = 4
    n_nodes = num_elements + 1
    E = 200000000000.0
    A = 0.0001
    P = 10000.0
    material_regions = [{'coord_min': x_min, 'coord_max': x_max, 'E': E, 'A': A}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': x_min, 'u_prescribed': 0.0}]
    neumann_bc_list = [{'x_location': x_max, 'load_mag': P}]
    n_gauss = 2
    result = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
    expected_node_coords = np.linspace(x_min, x_max, n_nodes)
    expected_displacements = P / (E * A) * expected_node_coords
    expected_reactions = np.array([-P])
    expected_reaction_nodes = np.array([0])
    assert 'displacements' in result
    assert 'reactions' in result
    assert 'node_coords' in result
    assert 'reaction_nodes' in result
    assert result['displacements'].shape == (n_nodes,)
    assert result['reactions'].shape == (1,)
    assert result['node_coords'].shape == (n_nodes,)
    assert result['reaction_nodes'].shape == (1,)
    np.testing.assert_allclose(result['displacements'], expected_displacements)
    np.testing.assert_allclose(result['reactions'], expected_reactions)
    np.testing.assert_allclose(result['node_coords'], expected_node_coords)
    np.testing.assert_array_equal(result['reaction_nodes'], expected_reaction_nodes)