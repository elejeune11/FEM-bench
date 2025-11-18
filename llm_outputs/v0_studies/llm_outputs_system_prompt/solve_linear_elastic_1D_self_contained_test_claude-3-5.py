def test_no_load_self_contained(fcn):
    """Test zero displacement and zero reaction in a fixed-free bar with no external load."""
    (x_min, x_max) = (0.0, 1.0)
    num_elements = 2
    material_regions = [{'coord_min': 0.0, 'coord_max': 1.0, 'E': 200000000000.0, 'A': 0.01}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    neumann_bc_list = None
    n_gauss = 2
    result = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
    assert 'displacements' in result
    assert 'reactions' in result
    assert 'node_coords' in result
    assert 'reaction_nodes' in result
    assert len(result['node_coords']) == num_elements + 1
    assert len(result['displacements']) == num_elements + 1
    assert len(result['reactions']) == len(dirichlet_bc_list)
    assert len(result['reaction_nodes']) == len(dirichlet_bc_list)
    assert np.allclose(result['displacements'], 0.0)
    assert np.allclose(result['reactions'], 0.0)
    assert np.allclose(result['node_coords'], np.linspace(x_min, x_max, num_elements + 1))
    assert result['reaction_nodes'][0] == 0

def test_uniform_extension_analytical_self_contained(fcn):
    """Test displacement field against a known analytical solution."""
    (x_min, x_max) = (0.0, 2.0)
    num_elements = 4
    (E, A) = (200000000000.0, 0.01)
    material_regions = [{'coord_min': 0.0, 'coord_max': 2.0, 'E': E, 'A': A}]
    body_force_fn = lambda x: 0.0
    F = 1000.0
    dirichlet_bc_list = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    neumann_bc_list = [{'x_location': 2.0, 'load_mag': F}]
    n_gauss = 2
    result = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
    x = result['node_coords']
    u_analytical = F * x / (E * A)
    R_analytical = -F
    assert np.allclose(result['displacements'], u_analytical, rtol=1e-10)
    assert np.allclose(result['reactions'], [R_analytical], rtol=1e-10)
    assert np.allclose(result['node_coords'], np.linspace(x_min, x_max, num_elements + 1))
    assert result['reaction_nodes'][0] == 0