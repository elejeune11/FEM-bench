def test_no_load_self_contained(fcn):
    """Test zero displacement and zero reaction in a fixed-free bar with no external load."""
    import numpy as np
    result = fcn(x_min=0.0, x_max=1.0, num_elements=2, material_regions=[{'coord_min': 0.0, 'coord_max': 1.0, 'E': 1.0, 'A': 1.0}], body_force_fn=lambda x: 0.0, dirichlet_bc_list=[{'x_location': 0.0, 'u_prescribed': 0.0}], neumann_bc_list=None, n_gauss=2)
    assert np.allclose(result['displacements'], 0.0)
    assert np.allclose(result['reactions'], 0.0)
    assert result['displacements'].shape == (3,)
    assert result['reactions'].shape == (1,)
    assert result['node_coords'].shape == (3,)
    assert result['reaction_nodes'].shape == (1,)
    assert result['displacements'][0] == 0.0
    assert np.array_equal(result['node_coords'], np.array([0.0, 0.5, 1.0]))
    assert result['reaction_nodes'][0] == 0

def test_uniform_extension_analytical_self_contained(fcn):
    """Test displacement field against a known analytical solution."""
    import numpy as np
    E = 100.0
    A = 2.0
    P = 50.0
    L = 1.0
    result = fcn(x_min=0.0, x_max=L, num_elements=4, material_regions=[{'coord_min': 0.0, 'coord_max': L, 'E': E, 'A': A}], body_force_fn=lambda x: 0.0, dirichlet_bc_list=[{'x_location': 0.0, 'u_prescribed': 0.0}], neumann_bc_list=[{'x_location': L, 'load_mag': P}], n_gauss=2)
    analytical_u = P * result['node_coords'] / (E * A)
    assert np.allclose(result['displacements'], analytical_u, rtol=1e-10)
    assert np.allclose(result['reactions'][0], -P, rtol=1e-10)
    assert result['displacements'].shape == (5,)
    assert result['reactions'].shape == (1,)
    assert result['node_coords'].shape == (5,)
    assert result['reaction_nodes'].shape == (1,)