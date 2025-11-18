def test_no_load_self_contained(fcn: Callable):
    """
    Test zero displacement and zero reaction in a fixed-free bar with no external load.
    A 2-element bar with uniform material, zero body force, and one fixed end should return:
    """
    x_min = 0.0
    x_max = 1.0
    num_elements = 2
    n_nodes = num_elements + 1
    material_regions = [{'coord_min': 0.0, 'coord_max': 1.0, 'E': 1.0, 'A': 1.0}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    neumann_bc_list = None
    n_gauss = 2
    expected_displacements = np.zeros(n_nodes)
    expected_reactions = np.array([0.0])
    expected_node_coords = np.linspace(x_min, x_max, n_nodes)
    expected_reaction_nodes = np.array([0])
    results = fcn(x_min=x_min, x_max=x_max, num_elements=num_elements, material_regions=material_regions, body_force_fn=body_force_fn, dirichlet_bc_list=dirichlet_bc_list, neumann_bc_list=neumann_bc_list, n_gauss=n_gauss)
    assert 'displacements' in results
    assert 'reactions' in results
    assert 'node_coords' in results
    assert 'reaction_nodes' in results
    assert results['displacements'].shape == (n_nodes,)
    assert results['reactions'].shape == (len(dirichlet_bc_list),)
    assert results['node_coords'].shape == (n_nodes,)
    assert results['reaction_nodes'].shape == (len(dirichlet_bc_list),)
    np.testing.assert_allclose(results['node_coords'], expected_node_coords)
    np.testing.assert_allclose(results['displacements'], expected_displacements, atol=1e-09)
    np.testing.assert_allclose(results['reactions'], expected_reactions, atol=1e-09)
    np.testing.assert_array_equal(results['reaction_nodes'], expected_reaction_nodes)
    assert np.isclose(results['displacements'][0], dirichlet_bc_list[0]['u_prescribed'])

def test_uniform_extension_analytical_self_contained(fcn: Callable):
    """Test displacement field against a known analytical solution."""
    L = 10.0
    E = 2.0
    A = 0.5
    P = 5.0
    num_elements = 4
    n_nodes = num_elements + 1
    u_analytical = lambda x: P / (E * A) * x
    x_min = 0.0
    x_max = L
    material_regions = [{'coord_min': x_min, 'coord_max': x_max, 'E': E, 'A': A}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    neumann_bc_list = [{'x_location': L, 'load_mag': P}]
    n_gauss = 2
    expected_node_coords = np.linspace(x_min, x_max, n_nodes)
    expected_displacements = u_analytical(expected_node_coords)
    expected_reactions = np.array([-P])
    expected_reaction_nodes = np.array([0])
    results = fcn(x_min=x_min, x_max=x_max, num_elements=num_elements, material_regions=material_regions, body_force_fn=body_force_fn, dirichlet_bc_list=dirichlet_bc_list, neumann_bc_list=neumann_bc_list, n_gauss=n_gauss)
    assert 'displacements' in results
    assert 'reactions' in results
    assert 'node_coords' in results
    assert 'reaction_nodes' in results
    np.testing.assert_allclose(results['node_coords'], expected_node_coords)
    np.testing.assert_allclose(results['displacements'], expected_displacements)
    np.testing.assert_allclose(results['reactions'], expected_reactions)
    np.testing.assert_array_equal(results['reaction_nodes'], expected_reaction_nodes)