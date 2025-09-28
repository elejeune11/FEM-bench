def test_no_load_self_contained(fcn):
    """Test zero displacement and zero reaction in a fixed-free bar with no external load.
    A 2-element bar with uniform material, zero body force, and one fixed end should return:
    """
    (x_min, x_max) = (0.0, 1.0)
    num_elements = 2
    material_regions = [{'coord_min': 0.0, 'coord_max': 1.0, 'E': 100.0, 'A': 1.0}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    neumann_bc_list = []
    n_gauss = 1
    result = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
    assert np.allclose(result['displacements'], 0.0)
    assert np.allclose(result['reactions'], 0.0)
    assert len(result['displacements']) == num_elements + 1
    assert len(result['node_coords']) == num_elements + 1
    assert len(result['reaction_nodes']) == len(dirichlet_bc_list)
    assert result['node_coords'][0] == 0.0
    assert result['node_coords'][-1] == 1.0

def test_uniform_extension_analytical_self_contained(fcn):
    """Test displacement field against a known analytical solution."""
    L = 1.0
    E = 100.0
    A = 0.1
    F = 10.0
    (x_min, x_max) = (0.0, L)
    num_elements = 4
    material_regions = [{'coord_min': 0.0, 'coord_max': L, 'E': E, 'A': A}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    neumann_bc_list = [{'x_location': L, 'load_mag': F}]
    n_gauss = 2

    def analytical_solution(x):
        return F * x / (E * A)
    result = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
    for (i, x) in enumerate(result['node_coords']):
        expected_u = analytical_solution(x)
        assert abs(result['displacements'][i] - expected_u) < 1e-10
    assert abs(result['reactions'][0] + F) < 1e-10