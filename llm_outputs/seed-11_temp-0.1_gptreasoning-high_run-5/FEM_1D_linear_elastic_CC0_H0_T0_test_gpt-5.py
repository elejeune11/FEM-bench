def test_no_load_self_contained(fcn):
    """
    Test zero displacement and zero reaction in a fixed-free bar with no external load.
    A 2-element bar with uniform material, zero body force, and one fixed end should return:
    """
    x_min = 0.0
    x_max = 2.0
    num_elements = 2
    E = 100.0
    A = 1.0
    material_regions = [{'coord_min': x_min, 'coord_max': x_max, 'E': E, 'A': A}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': x_min, 'u_prescribed': 0.0}]
    neumann_bc_list = None
    n_gauss = 2
    result = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
    u = result['displacements']
    R = result['reactions']
    x = result['node_coords']
    rnodes = result['reaction_nodes']
    expected_coords = np.linspace(x_min, x_max, num_elements + 1)
    assert u.shape == (num_elements + 1,)
    assert x.shape == (num_elements + 1,)
    assert R.shape == (len(dirichlet_bc_list),)
    assert rnodes.shape == (len(dirichlet_bc_list),)
    assert np.allclose(x, expected_coords, atol=1e-14)
    assert np.allclose(u, 0.0, atol=1e-12)
    assert np.allclose(R, 0.0, atol=1e-12)
    bc_node_idx = int(rnodes[0])
    assert np.isclose(x[bc_node_idx], dirichlet_bc_list[0]['x_location'], atol=1e-14)
    assert np.isclose(u[bc_node_idx], dirichlet_bc_list[0]['u_prescribed'], atol=1e-12)

def test_analytical_solution(fcn):
    """
    Test a non-zero displacement field against a known analytical solution.
    For a bar with constant E and A, fixed at x=0 and subjected to a point Neumann
    load P at x=L, with zero body force, the analytical solution is:
        u(x) = (P / (E*A)) * x
    The FE solution with linear elements should match this exactly at the nodes.
    """
    x_min = 0.0
    x_max = 3.0
    num_elements = 8
    E = 70.0
    A = 2.5
    P = 10.0
    material_regions = [{'coord_min': x_min, 'coord_max': x_max, 'E': E, 'A': A}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': x_min, 'u_prescribed': 0.0}]
    neumann_bc_list = [{'x_location': x_max, 'load_mag': P}]
    n_gauss = 2
    result = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
    u = result['displacements']
    x = result['node_coords']
    expected_u = P / (E * A) * x
    assert u.shape == (num_elements + 1,)
    assert x.shape == (num_elements + 1,)
    assert np.allclose(u, expected_u, rtol=1e-12, atol=1e-12)
    assert np.isclose(u[0], 0.0, atol=1e-12)