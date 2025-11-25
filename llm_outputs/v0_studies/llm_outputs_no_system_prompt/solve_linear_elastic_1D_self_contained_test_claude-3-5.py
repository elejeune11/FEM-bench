def test_no_load_self_contained(fcn):
    """Test zero displacement and zero reaction in a fixed-free bar with no external load.
    A 2-element bar with uniform material, zero body force, and one fixed end should return:
    """
    material = [{'coord_min': 0.0, 'coord_max': 2.0, 'E': 1000.0, 'A': 0.1}]
    dirichlet = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    body_force = lambda x: 0.0
    result = fcn(x_min=0.0, x_max=2.0, num_elements=2, material_regions=material, body_force_fn=body_force, dirichlet_bc_list=dirichlet, neumann_bc_list=None, n_gauss=2)
    assert 'displacements' in result
    assert 'reactions' in result
    assert 'node_coords' in result
    assert 'reaction_nodes' in result
    assert len(result['node_coords']) == 3
    assert len(result['displacements']) == 3
    assert len(result['reactions']) == 1
    assert len(result['reaction_nodes']) == 1
    assert numpy.allclose(result['displacements'], 0.0)
    assert numpy.allclose(result['reactions'], 0.0)
    assert numpy.allclose(result['node_coords'], [0.0, 1.0, 2.0])
    assert result['reaction_nodes'][0] == 0

def test_uniform_extension_analytical_self_contained(fcn):
    """Test displacement field against a known analytical solution."""
    L = 1.0
    E = 1000.0
    A = 0.1
    P = 10.0
    material = [{'coord_min': 0.0, 'coord_max': L, 'E': E, 'A': A}]
    dirichlet = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    neumann = [{'x_location': L, 'load_mag': P}]
    body_force = lambda x: 0.0
    result = fcn(x_min=0.0, x_max=L, num_elements=4, material_regions=material, body_force_fn=body_force, dirichlet_bc_list=dirichlet, neumann_bc_list=neumann, n_gauss=2)
    x = result['node_coords']
    u_numerical = result['displacements']
    u_analytical = P * x / (E * A)
    assert numpy.allclose(u_numerical, u_analytical, rtol=1e-10)
    assert numpy.allclose(result['reactions'], [-P], rtol=1e-10)