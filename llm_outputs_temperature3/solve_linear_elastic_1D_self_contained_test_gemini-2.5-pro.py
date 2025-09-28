def test_no_load_self_contained(fcn: Callable):
    """
    Test zero displacement and zero reaction in a fixed-free bar with no external load.
    A 2-element bar with uniform material, zero body force, and one fixed end should return:
    """
    x_min = 0.0
    x_max = 2.0
    num_elements = 2
    num_nodes = num_elements + 1
    material_regions = [{'coord_min': x_min, 'coord_max': x_max, 'E': 1.0, 'A': 1.0}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    neumann_bc_list = []
    n_gauss = 1
    results = fcn(x_min=x_min, x_max=x_max, num_elements=num_elements, material_regions=material_regions, body_force_fn=body_force_fn, dirichlet_bc_list=dirichlet_bc_list, neumann_bc_list=neumann_bc_list, n_gauss=n_gauss)
    expected_displacements = np.zeros(num_nodes)
    expected_reactions = np.zeros(len(dirichlet_bc_list))
    expected_node_coords = np.linspace(x_min, x_max, num_nodes)
    expected_reaction_nodes = np.array([0])
    assert results['displacements'].shape == (num_nodes,)
    assert results['reactions'].shape == (len(dirichlet_bc_list),)
    assert results['node_coords'].shape == (num_nodes,)
    assert results['reaction_nodes'].shape == (len(dirichlet_bc_list),)
    np.testing.assert_allclose(results['node_coords'], expected_node_coords)
    np.testing.assert_allclose(results['displacements'], expected_displacements, atol=1e-09)
    np.testing.assert_allclose(results['reactions'], expected_reactions, atol=1e-09)
    np.testing.assert_allclose(results['reaction_nodes'], expected_reaction_nodes)
    assert results['displacements'][expected_reaction_nodes[0]] == dirichlet_bc_list[0]['u_prescribed']

def test_uniform_extension_analytical_self_contained(fcn: Callable):
    """Test displacement field against a known analytical solution."""
    L = 10.0
    E = 2.0
    A = 0.5
    P = 5.0
    EA = E * A
    x_min = 0.0
    x_max = L
    num_elements = 5
    num_nodes = num_elements + 1
    material_regions = [{'coord_min': x_min, 'coord_max': x_max, 'E': E, 'A': A}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    neumann_bc_list = [{'x_location': L, 'load_mag': P}]
    n_gauss = 1
    results = fcn(x_min=x_min, x_max=x_max, num_elements=num_elements, material_regions=material_regions, body_force_fn=body_force_fn, dirichlet_bc_list=dirichlet_bc_list, neumann_bc_list=neumann_bc_list, n_gauss=n_gauss)
    node_coords = results['node_coords']
    expected_displacements = P / EA * node_coords
    expected_reaction = -P
    expected_reaction_node = 0
    np.testing.assert_allclose(results['node_coords'], np.linspace(x_min, x_max, num_nodes))
    np.testing.assert_allclose(results['displacements'], expected_displacements, atol=1e-09)
    np.testing.assert_allclose(results['reactions'], np.array([expected_reaction]), atol=1e-09)
    np.testing.assert_allclose(results['reaction_nodes'], np.array([expected_reaction_node]))
    assert results['displacements'][expected_reaction_node] == dirichlet_bc_list[0]['u_prescribed']