def test_no_load_self_contained(fcn):
    """
    Test zero displacement and zero reaction in a fixed-free bar with no external load.
    A 2-element bar with uniform material, zero body force, and one fixed end should return:
    """
    x_min, x_max = (0.0, 1.0)
    num_elements = 2
    material_regions = [{'coord_min': x_min, 'coord_max': x_max, 'E': 100.0, 'A': 1.0}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': x_min, 'u_prescribed': 0.0}]
    neumann_bc_list = None
    n_gauss = 2
    result = fcn(x_min=x_min, x_max=x_max, num_elements=num_elements, material_regions=material_regions, body_force_fn=body_force_fn, dirichlet_bc_list=dirichlet_bc_list, neumann_bc_list=neumann_bc_list, n_gauss=n_gauss)
    u = result['displacements']
    r = result['reactions']
    x = result['node_coords']
    r_nodes = result['reaction_nodes']
    assert isinstance(u, np.ndarray) and isinstance(r, np.ndarray)
    assert isinstance(x, np.ndarray) and isinstance(r_nodes, np.ndarray)
    n_nodes = num_elements + 1
    assert u.shape == (n_nodes,)
    assert x.shape == (n_nodes,)
    assert r.shape == (len(dirichlet_bc_list),)
    assert r_nodes.shape == (len(dirichlet_bc_list),)
    assert np.allclose(u, 0.0)
    assert np.allclose(r, 0.0)
    idx_fixed = np.where(np.isclose(x, x_min))[0]
    assert idx_fixed.size == 1
    assert r_nodes[0] == idx_fixed[0]
    assert np.isclose(u[idx_fixed[0]], dirichlet_bc_list[0]['u_prescribed'])

def test_analytical_solution(fcn):
    """
    Test a non-zero displacement field against a known analytical solution.
    """
    x_min, x_max = (0.0, 1.0)
    L = x_max - x_min
    num_elements = 4
    E, A = (200.0, 5.0)
    P = 400.0
    material_regions = [{'coord_min': x_min, 'coord_max': x_max, 'E': E, 'A': A}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': x_min, 'u_prescribed': 0.0}]
    neumann_bc_list = [{'x_location': x_max, 'load_mag': P}]
    n_gauss = 2
    result = fcn(x_min=x_min, x_max=x_max, num_elements=num_elements, material_regions=material_regions, body_force_fn=body_force_fn, dirichlet_bc_list=dirichlet_bc_list, neumann_bc_list=neumann_bc_list, n_gauss=n_gauss)
    u = result['displacements']
    x = result['node_coords']
    expected_u = P / (E * A) * x
    assert u.shape == expected_u.shape
    assert np.allclose(u, expected_u, rtol=1e-12, atol=1e-14)