def test_no_load_self_contained(fcn):
    """Test zero displacement and zero reaction in a fixed-free bar with no external load.
    A 2-element bar with uniform material, zero body force, and one fixed end should return:
    """
    import numpy as np
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
    n_nodes = num_elements + 1
    assert result['displacements'].shape == (n_nodes,)
    assert result['reactions'].shape == (1,)
    assert result['node_coords'].shape == (n_nodes,)
    assert result['reaction_nodes'].shape == (1,)
    assert np.allclose(result['displacements'], 0.0, atol=1e-12)
    assert np.allclose(result['reactions'], 0.0, atol=1e-12)
    assert result['reaction_nodes'][0] == 0

def test_uniform_extension_analytical_self_contained(fcn):
    """Test displacement field against a known analytical solution."""
    import numpy as np
    x_min = 0.0
    x_max = 2.0
    num_elements = 4
    E = 200.0
    A = 1.0
    f = 10.0
    material_regions = [{'coord_min': 0.0, 'coord_max': 2.0, 'E': E, 'A': A}]
    body_force_fn = lambda x: f
    dirichlet_bc_list = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    neumann_bc_list = None
    n_gauss = 2
    result = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
    L = x_max - x_min
    x_coords = result['node_coords']
    analytical_displacements = f * x_coords * (L - x_coords / 2) / (E * A)
    assert np.allclose(result['displacements'], analytical_displacements, rtol=1e-10)
    expected_reaction = -f * L * A
    assert np.allclose(result['reactions'][0], expected_reaction, rtol=1e-10)