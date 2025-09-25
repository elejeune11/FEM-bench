def test_no_load_self_contained(fcn: Callable) -> None:
    """
    Test zero displacement and zero reaction in a fixed-free bar with no external load.
    A 2-element bar with uniform material, zero body force, and one fixed end should return:
    """
    x_min = 0.0
    x_max = 1.0
    num_elements = 2
    material_regions = [{'coord_min': 0.0, 'coord_max': 1.0, 'E': 1.0, 'A': 1.0}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    neumann_bc_list = None
    n_gauss = 2
    results = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
    n_nodes = num_elements + 1
    expected_displacements = np.zeros(n_nodes)
    expected_reactions = np.array([0.0])
    expected_node_coords = np.linspace(x_min, x_max, n_nodes)
    expected_reaction_nodes = np.array([0])
    assert np.allclose(results['displacements'], expected_displacements, atol=1e-09)
    assert np.allclose(results['reactions'], expected_reactions, atol=1e-09)
    assert np.allclose(results['node_coords'], expected_node_coords, atol=1e-09)
    assert np.array_equal(results['reaction_nodes'], expected_reaction_nodes)
    assert results['displacements'].shape == (n_nodes,)
    assert results['reactions'].shape == (len(dirichlet_bc_list),)
    assert results['node_coords'].shape == (n_nodes,)
    assert results['reaction_nodes'].shape == (len(dirichlet_bc_list),)

def test_uniform_extension_analytical_self_contained(fcn: Callable) -> None:
    """
    Test displacement field against a known analytical solution.
    A bar fixed at one end and subjected to a point load at the other end
    should exhibit a linear displacement field.
    """
    x_min = 0.0
    x_max = 1.0
    num_elements = 10
    E = 100.0
    A = 1.0
    P = 10.0
    material_regions = [{'coord_min': 0.0, 'coord_max': 1.0, 'E': E, 'A': A}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    neumann_bc_list = [{'x_location': 1.0, 'load_mag': P}]
    n_gauss = 2
    results = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
    n_nodes = num_elements + 1
    node_coords = results['node_coords']
    expected_displacements = P * node_coords / (E * A)
    expected_reactions = np.array([-P])
    assert np.allclose(results['displacements'], expected_displacements, atol=1e-09)
    assert np.allclose(results['reactions'], expected_reactions, atol=1e-09)
    assert np.allclose(results['node_coords'], np.linspace(x_min, x_max, n_nodes), atol=1e-09)
    assert np.array_equal(results['reaction_nodes'], np.array([0]))
    assert results['displacements'].shape == (n_nodes,)
    assert results['reactions'].shape == (len(dirichlet_bc_list),)
    assert results['node_coords'].shape == (n_nodes,)
    assert results['reaction_nodes'].shape == (len(dirichlet_bc_list),)