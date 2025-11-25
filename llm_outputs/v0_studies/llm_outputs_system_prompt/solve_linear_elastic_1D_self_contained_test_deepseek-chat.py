def test_no_load_self_contained(fcn):
    """Test zero displacement and zero reaction in a fixed-free bar with no external load.
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
    n_nodes = num_elements + 1
    n_dirichlet = len(dirichlet_bc_list)
    assert result['displacements'].shape == (n_nodes,)
    assert np.allclose(result['displacements'], 0.0)
    assert result['reactions'].shape == (n_dirichlet,)
    assert np.allclose(result['reactions'], 0.0)
    assert result['node_coords'].shape == (n_nodes,)
    assert result['reaction_nodes'].shape == (n_dirichlet,)
    assert result['reaction_nodes'][0] == 0

def test_uniform_extension_analytical_self_contained(fcn):
    """Test displacement field against a known analytical solution."""
    x_min = 0.0
    x_max = 1.0
    num_elements = 10
    material_regions = [{'coord_min': 0.0, 'coord_max': 1.0, 'E': 200000000000.0, 'A': 0.01}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    neumann_bc_list = [{'x_location': 1.0, 'load_mag': 1000000000.0}]
    n_gauss = 2
    result = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
    E = material_regions[0]['E']
    A = material_regions[0]['A']
    L = x_max - x_min
    P = neumann_bc_list[0]['load_mag']
    analytical_displacement = P * result['node_coords'] / (E * A)
    analytical_reaction = -P
    assert np.allclose(result['displacements'], analytical_displacement, rtol=1e-05)
    assert np.allclose(result['reactions'][0], analytical_reaction, rtol=1e-05)