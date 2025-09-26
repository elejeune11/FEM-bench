def test_no_load_self_contained(fcn):
    """Test zero displacement and zero reaction in a fixed-free bar with no external load."""
    x_min = 0.0
    x_max = 1.0
    num_elements = 2
    material_regions = [{'coord_min': 0.0, 'coord_max': 1.0, 'E': 210000000000.0, 'A': 0.01}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    neumann_bc_list = None
    n_gauss = 2
    result = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
    displacements = result['displacements']
    reactions = result['reactions']
    node_coords = result['node_coords']
    reaction_nodes = result['reaction_nodes']
    assert np.allclose(displacements, 0.0), 'Displacements should be zero at all nodes.'
    assert np.allclose(reactions, 0.0), 'Reactions should be zero at the fixed node.'
    assert displacements.shape == (num_elements + 1,), 'Displacement array shape mismatch.'
    assert reactions.shape == (1,), 'Reactions array shape mismatch.'
    assert node_coords.shape == (num_elements + 1,), 'Node coordinates array shape mismatch.'
    assert reaction_nodes.shape == (1,), 'Reaction nodes array shape mismatch.'
    assert reaction_nodes[0] == 0, 'Reaction node should be the first node.'

def test_uniform_extension_analytical_self_contained(fcn):
    """Test displacement field against a known analytical solution."""
    x_min = 0.0
    x_max = 1.0
    num_elements = 2
    E = 210000000000.0
    A = 0.01
    material_regions = [{'coord_min': 0.0, 'coord_max': 1.0, 'E': E, 'A': A}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    neumann_bc_list = [{'x_location': 1.0, 'load_mag': 1000.0}]
    n_gauss = 2
    result = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
    displacements = result['displacements']
    node_coords = result['node_coords']
    analytical_displacements = 1000.0 * node_coords / (E * A)
    assert np.allclose(displacements, analytical_displacements, atol=1e-05), 'Displacements do not match analytical solution.'
    assert displacements.shape == (num_elements + 1,), 'Displacement array shape mismatch.'
    assert node_coords.shape == (num_elements + 1,), 'Node coordinates array shape mismatch.'