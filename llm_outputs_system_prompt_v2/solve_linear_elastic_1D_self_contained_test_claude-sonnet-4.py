def test_no_load_self_contained(fcn):
    """Test zero displacement and zero reaction in a fixed-free bar with no external load.
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
    result = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
    assert 'displacements' in result
    assert 'reactions' in result
    assert 'node_coords' in result
    assert 'reaction_nodes' in result
    displacements = result['displacements']
    reactions = result['reactions']
    node_coords = result['node_coords']
    reaction_nodes = result['reaction_nodes']
    assert displacements.shape == (3,)
    assert reactions.shape == (1,)
    assert node_coords.shape == (3,)
    assert reaction_nodes.shape == (1,)
    assert np.allclose(displacements, 0.0)
    assert np.allclose(reactions, 0.0)
    assert np.allclose(node_coords, [0.0, 0.5, 1.0])
    assert reaction_nodes[0] == 0

def test_uniform_extension_analytical_self_contained(fcn):
    """Test displacement field against a known analytical solution."""
    x_min = 0.0
    x_max = 1.0
    num_elements = 4
    E = 200000000000.0
    A = 0.01
    P = 1000.0
    material_regions = [{'coord_min': 0.0, 'coord_max': 1.0, 'E': E, 'A': A}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    neumann_bc_list = [{'x_location': 1.0, 'load_mag': P}]
    n_gauss = 2
    result = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
    displacements = result['displacements']
    node_coords = result['node_coords']
    analytical_displacements = P * node_coords / (E * A)
    assert np.allclose(displacements, analytical_displacements, rtol=1e-10)
    assert displacements[0] == 0.0
    assert displacements[-1] == P / (E * A)