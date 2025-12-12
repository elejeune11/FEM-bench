def test_no_load_self_contained(fcn):
    """
    Test zero displacement and zero reaction in a fixed-free bar with no external load.
    A 2-element bar with uniform material, zero body force, and one fixed end should return:
    """
    (x_min, x_max) = (0.0, 1.0)
    num_elements = 2
    material_regions = [{'coord_min': 0.0, 'coord_max': 1.0, 'E': 1.0, 'A': 1.0}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    neumann_bc_list = None
    n_gauss = 2
    result = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
    assert np.allclose(result['displacements'], 0.0)
    assert np.allclose(result['reactions'], 0.0)
    assert result['displacements'].shape == (num_elements + 1,)
    assert result['node_coords'].shape == (num_elements + 1,)
    assert result['reaction_nodes'].shape == (1,)

def test_analytical_solution(fcn):
    """
    Test a non-zero displacement field against a known analytical solution.
    """
    (x_min, x_max) = (0.0, 1.0)
    num_elements = 10
    material_regions = [{'coord_min': 0.0, 'coord_max': 1.0, 'E': 1.0, 'A': 1.0}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': 0.0, 'u_prescribed': 0.0}, {'x_location': 1.0, 'u_prescribed': 1.0}]
    neumann_bc_list = None
    n_gauss = 2
    result = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
    analytical_solution = result['node_coords']
    assert np.allclose(result['displacements'], analytical_solution)