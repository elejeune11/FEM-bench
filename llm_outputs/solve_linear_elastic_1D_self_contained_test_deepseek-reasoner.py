def test_no_load_self_contained(fcn):
    """Test zero displacement and zero reaction in a fixed-free bar with no external load.
    A 2-element bar with uniform material, zero body force, and one fixed end should return:
    """
    import numpy as np
    from pytest import approx
    x_min = 0.0
    x_max = 1.0
    num_elements = 2
    material_regions = [{'coord_min': 0.0, 'coord_max': 1.0, 'E': 1.0, 'A': 1.0}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    neumann_bc_list = None
    n_gauss = 1
    result = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
    assert result['displacements'] == approx(np.zeros(3))
    assert result['reactions'] == approx(np.zeros(1))
    assert result['node_coords'].shape == (3,)
    assert result['reaction_nodes'].shape == (1,)
    assert result['node_coords'][0] == approx(0.0)
    assert result['node_coords'][-1] == approx(1.0)

def test_uniform_extension_analytical_self_contained(fcn):
    """Test displacement field against a known analytical solution."""
    import numpy as np
    from pytest import approx
    x_min = 0.0
    x_max = 2.0
    num_elements = 4
    E = 100.0
    A = 2.0
    material_regions = [{'coord_min': 0.0, 'coord_max': 2.0, 'E': E, 'A': A}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    neumann_bc_list = [{'x_location': 2.0, 'load_mag': 50.0}]
    n_gauss = 2
    result = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
    node_coords = result['node_coords']
    expected_displacements = 50.0 * node_coords / (E * A)
    assert result['displacements'] == approx(expected_displacements, rel=1e-05)
    assert abs(result['reactions'][0] + 50.0) < 1e-10