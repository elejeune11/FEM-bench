def test_no_load_self_contained(fcn):
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
    result = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
    assert np.allclose(result['displacements'], 0.0), 'Displacements should be zero'
    assert np.allclose(result['reactions'], 0.0), 'Reactions should be zero'
    assert result['node_coords'].shape == (num_elements + 1,), 'Incorrect node coordinates shape'
    assert result['reaction_nodes'].shape == (len(dirichlet_bc_list),), 'Incorrect reaction nodes shape'

def test_uniform_extension_analytical_self_contained(fcn):
    """
    Test displacement field against a known analytical solution.
    A 2-element bar with uniform material and a linear body force should match the analytical solution.
    """
    x_min = 0.0
    x_max = 1.0
    num_elements = 2
    E = 1.0
    A = 1.0
    material_regions = [{'coord_min': 0.0, 'coord_max': 1.0, 'E': E, 'A': A}]
    body_force_fn = lambda x: 1.0
    dirichlet_bc_list = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    neumann_bc_list = [{'x_location': 1.0, 'load_mag': 0.0}]
    n_gauss = 2
    result = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
    node_coords = result['node_coords']
    expected_displacements = 1 / (2 * E * A) * (node_coords ** 2 - node_coords)
    assert np.allclose(result['displacements'], expected_displacements), 'Displacements do not match analytical solution'
    assert result['node_coords'].shape == (num_elements + 1,), 'Incorrect node coordinates shape'
    assert result['reaction_nodes'].shape == (len(dirichlet_bc_list),), 'Incorrect reaction nodes shape'