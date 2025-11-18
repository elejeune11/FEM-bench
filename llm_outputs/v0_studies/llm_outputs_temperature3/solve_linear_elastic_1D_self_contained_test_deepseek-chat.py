def test_no_load_self_contained(fcn):
    """Test zero displacement and zero reaction in a fixed-free bar with no external load."""
    x_min = 0.0
    x_max = 2.0
    num_elements = 2
    material_regions = [{'coord_min': 0.0, 'coord_max': 2.0, 'E': 200000000000.0, 'A': 0.01}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    neumann_bc_list = None
    n_gauss = 2
    result = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
    assert np.allclose(result['displacements'], 0.0)
    assert np.allclose(result['reactions'], 0.0)
    assert result['node_coords'].shape == (3,)
    assert result['displacements'].shape == (3,)
    assert result['reaction_nodes'].shape == (1,)
    assert result['reaction_nodes'][0] == 0

def test_uniform_extension_analytical_self_contained(fcn):
    """Test displacement field against a known analytical solution."""
    x_min = 0.0
    x_max = 1.0
    num_elements = 3
    E = 200000000000.0
    A = 0.01
    material_regions = [{'coord_min': 0.0, 'coord_max': 1.0, 'E': E, 'A': A}]
    body_force_mag = 1000.0
    body_force_fn = lambda x: body_force_mag
    dirichlet_bc_list = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    neumann_bc_list = None
    n_gauss = 2

    def analytical_solution(x):
        L = x_max - x_min
        return body_force_mag / (2 * E * A) * (2 * L * x - x ** 2)
    result = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
    node_coords = result['node_coords']
    displacements = result['displacements']
    for (i, x) in enumerate(node_coords):
        analytical_u = analytical_solution(x)
        assert abs(displacements[i] - analytical_u) < 1e-06
    assert abs(displacements[0]) < 1e-10