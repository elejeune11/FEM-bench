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
    assert np.allclose(result['displacements'], 0.0)
    assert np.allclose(result['reactions'], 0.0)
    assert result['reaction_nodes'][0] == 0
    assert np.isclose(result['node_coords'][0], 0.0)

def test_uniform_extension_analytical_self_contained(fcn):
    """Test displacement field against a known analytical solution."""
    import numpy as np
    x_min = 0.0
    x_max = 1.0
    num_elements = 4
    E = 200000000000.0
    A = 0.01
    material_regions = [{'coord_min': 0.0, 'coord_max': 1.0, 'E': E, 'A': A}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    load_mag = 1000.0
    neumann_bc_list = [{'x_location': 1.0, 'load_mag': load_mag}]
    n_gauss = 2
    result = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
    node_coords = result['node_coords']
    analytical_displacements = load_mag * node_coords / (E * A)
    assert np.allclose(result['displacements'], analytical_displacements, rtol=1e-10)
    assert np.allclose(result['reactions'], [-load_mag], rtol=1e-10)
    assert np.isclose(result['displacements'][0], 0.0)
    n_nodes = num_elements + 1
    assert result['displacements'].shape == (n_nodes,)
    assert result['reactions'].shape == (1,)
    assert result['node_coords'].shape == (n_nodes,)
    assert result['reaction_nodes'].shape == (1,)