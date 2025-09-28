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
    assert np.allclose(result['displacements'], 0.0), 'Displacements should be zero at all nodes.'
    assert np.allclose(result['reactions'], 0.0), 'Reactions should be zero at the fixed node.'
    assert result['displacements'].shape == (num_elements + 1,), 'Displacement array shape mismatch.'
    assert result['reactions'].shape == (1,), 'Reactions array shape mismatch.'
    assert np.array_equal(result['reaction_nodes'], [0]), 'Reaction nodes should match Dirichlet BC nodes.'

def test_uniform_extension_analytical_self_contained(fcn):
    """Test displacement field against a known analytical solution."""
    x_min = 0.0
    x_max = 1.0
    num_elements = 2
    material_regions = [{'coord_min': 0.0, 'coord_max': 1.0, 'E': 1.0, 'A': 1.0}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    neumann_bc_list = [{'x_location': 1.0, 'load_mag': 1.0}]
    n_gauss = 2
    result = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
    expected_displacements = np.array([0.0, 0.5, 1.0])
    assert np.allclose(result['displacements'], expected_displacements), 'Displacements do not match analytical solution.'
    assert result['displacements'].shape == (num_elements + 1,), 'Displacement array shape mismatch.'
    assert result['reactions'].shape == (1,), 'Reactions array shape mismatch.'
    assert np.array_equal(result['reaction_nodes'], [0]), 'Reaction nodes should match Dirichlet BC nodes.'