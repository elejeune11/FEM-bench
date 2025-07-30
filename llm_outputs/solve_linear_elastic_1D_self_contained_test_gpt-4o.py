def test_no_load_self_contained(fcn):
    """
    Test zero displacement and zero reaction in a fixed-free bar with no external load.
    A 2-element bar with uniform material, zero body force, and one fixed end should return:
    """
    x_min = 0.0
    x_max = 2.0
    num_elements = 2
    material_regions = [{'coord_min': 0.0, 'coord_max': 2.0, 'E': 1.0, 'A': 1.0}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    neumann_bc_list = None
    n_gauss = 2
    result = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
    assert np.allclose(result['displacements'], np.zeros(3)), 'Displacements should be zero at all nodes'
    assert np.allclose(result['reactions'], np.zeros(1)), 'Reactions should be zero at the fixed node'
    assert result['displacements'].shape == (3,), 'Displacement array shape should be (3,)'
    assert result['reactions'].shape == (1,), 'Reactions array shape should be (1,)'
    assert result['node_coords'].shape == (3,), 'Node coordinates array shape should be (3,)'
    assert result['reaction_nodes'].shape == (1,), 'Reaction nodes array shape should be (1,)'
    assert result['reaction_nodes'][0] == 0, 'Reaction node should be at the fixed end'

def test_uniform_extension_analytical_self_contained(fcn):
    """
    Test displacement field against a known analytical solution.
    A bar with uniform material properties and a uniform body force should match the analytical solution.
    """
    x_min = 0.0
    x_max = 2.0
    num_elements = 2
    E = 1.0
    A = 1.0
    body_force_value = 1.0
    material_regions = [{'coord_min': 0.0, 'coord_max': 2.0, 'E': E, 'A': A}]
    body_force_fn = lambda x: body_force_value
    dirichlet_bc_list = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    neumann_bc_list = [{'x_location': 2.0, 'load_mag': 0.0}]
    n_gauss = 2
    result = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
    analytical_displacements = np.array([0.0, 1.0, 2.0]) * body_force_value / (2 * E * A)
    assert np.allclose(result['displacements'], analytical_displacements), 'Displacements should match the analytical solution'
    assert result['displacements'].shape == (3,), 'Displacement array shape should be (3,)'
    assert result['reactions'].shape == (1,), 'Reactions array shape should be (1,)'
    assert result['node_coords'].shape == (3,), 'Node coordinates array shape should be (3,)'
    assert result['reaction_nodes'].shape == (1,), 'Reaction nodes array shape should be (1,)'
    assert result['reaction_nodes'][0] == 0, 'Reaction node should be at the fixed end'