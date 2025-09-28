def test_no_load_self_contained(fcn: Callable):
    """Test zero displacement and zero reaction in a fixed-free bar with no external load.
    A 2-element bar with uniform material, zero body force, and one fixed end should return:
    """
    x_min = 0.0
    x_max = 1.0
    num_elements = 2
    n_nodes = num_elements + 1
    material_regions = [{'coord_min': 0.0, 'coord_max': 1.0, 'E': 10.0, 'A': 1.0}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    neumann_bc_list = []
    n_gauss = 1
    results = fcn(x_min=x_min, x_max=x_max, num_elements=num_elements, material_regions=material_regions, body_force_fn=body_force_fn, dirichlet_bc_list=dirichlet_bc_list, neumann_bc_list=neumann_bc_list, n_gauss=n_gauss)
    assert results['displacements'].shape == (n_nodes,)
    assert results['reactions'].shape == (len(dirichlet_bc_list),)
    assert results['node_coords'].shape == (n_nodes,)
    assert results['reaction_nodes'].shape == (len(dirichlet_bc_list),)
    expected_displacements = np.zeros(n_nodes)
    assert np.allclose(results['displacements'], expected_displacements)
    expected_reactions = np.zeros(len(dirichlet_bc_list))
    assert np.allclose(results['reactions'], expected_reactions)
    expected_node_coords = np.linspace(x_min, x_max, n_nodes)
    assert np.allclose(results['node_coords'], expected_node_coords)
    expected_reaction_nodes = np.array([0])
    assert np.array_equal(results['reaction_nodes'], expected_reaction_nodes)
    assert results['displacements'][0] == dirichlet_bc_list[0]['u_prescribed']

def test_uniform_extension_analytical_self_contained(fcn: Callable):
    """Test displacement field against a known analytical solution."""
    x_min = 0.0
    x_max = 2.0
    num_elements = 4
    n_nodes = num_elements + 1
    E = 10.0
    A = 2.0
    P = 5.0
    material_regions = [{'coord_min': x_min, 'coord_max': x_max, 'E': E, 'A': A}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': x_min, 'u_prescribed': 0.0}]
    neumann_bc_list = [{'x_location': x_max, 'load_mag': P}]
    n_gauss = 1
    results = fcn(x_min=x_min, x_max=x_max, num_elements=num_elements, material_regions=material_regions, body_force_fn=body_force_fn, dirichlet_bc_list=dirichlet_bc_list, neumann_bc_list=neumann_bc_list, n_gauss=n_gauss)
    node_coords = np.linspace(x_min, x_max, n_nodes)
    expected_displacements = P / (E * A) * node_coords
    expected_reaction = np.array([-P])
    assert np.allclose(results['node_coords'], node_coords)
    assert np.allclose(results['displacements'], expected_displacements)
    assert np.allclose(results['reactions'], expected_reaction)
    expected_reaction_nodes = np.array([0])
    assert np.array_equal(results['reaction_nodes'], expected_reaction_nodes)