def test_no_load_self_contained(fcn):
    """Test zero displacement and zero reaction in a fixed-free bar with no external load.
    A 2-element bar with uniform material, zero body force, and one fixed end should return:
    """
    material_regions = [{'coord_min': 0.0, 'coord_max': 1.0, 'E': 1000.0, 'A': 0.01}]
    dirichlet_bc_list = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    body_force_fn = lambda x: 0.0
    result = fcn(x_min=0.0, x_max=1.0, num_elements=2, material_regions=material_regions, body_force_fn=body_force_fn, dirichlet_bc_list=dirichlet_bc_list, neumann_bc_list=None, n_gauss=2)
    assert 'displacements' in result
    assert 'reactions' in result
    assert 'node_coords' in result
    assert 'reaction_nodes' in result
    assert len(result['node_coords']) == 3
    assert len(result['displacements']) == 3
    assert len(result['reactions']) == 1
    assert len(result['reaction_nodes']) == 1
    assert np.allclose(result['displacements'], 0.0)
    assert np.allclose(result['reactions'], 0.0)
    assert np.allclose(result['node_coords'], np.linspace(0.0, 1.0, 3))
    assert result['reaction_nodes'][0] == 0

def test_uniform_extension_analytical_self_contained(fcn):
    """Test displacement field against a known analytical solution."""
    L = 2.0
    E = 1000.0
    A = 0.01
    P = 10.0
    material_regions = [{'coord_min': 0.0, 'coord_max': L, 'E': E, 'A': A}]
    dirichlet_bc_list = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    neumann_bc_list = [{'x_location': L, 'load_mag': P}]
    body_force_fn = lambda x: 0.0
    result = fcn(x_min=0.0, x_max=L, num_elements=4, material_regions=material_regions, body_force_fn=body_force_fn, dirichlet_bc_list=dirichlet_bc_list, neumann_bc_list=neumann_bc_list, n_gauss=2)
    x = result['node_coords']
    u_numerical = result['displacements']
    u_analytical = P * x / (E * A)
    assert np.allclose(u_numerical, u_analytical, rtol=1e-10)
    assert np.allclose(result['reactions'], [-P], rtol=1e-10)