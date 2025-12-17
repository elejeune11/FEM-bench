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

    def body_force_fn(x):
        return 0.0
    dirichlet_bc_list = [{'x_location': x_min, 'u_prescribed': 0.0}]
    neumann_bc_list = None
    n_gauss = 2
    result = fcn(x_min=x_min, x_max=x_max, num_elements=num_elements, material_regions=material_regions, body_force_fn=body_force_fn, dirichlet_bc_list=dirichlet_bc_list, neumann_bc_list=neumann_bc_list, n_gauss=n_gauss)
    assert isinstance(result, dict)
    assert 'displacements' in result and 'reactions' in result
    assert 'node_coords' in result and 'reaction_nodes' in result
    n_nodes = num_elements + 1
    u = result['displacements']
    R = result['reactions']
    x = result['node_coords']
    r_nodes = result['reaction_nodes']
    assert isinstance(u, np.ndarray) and u.shape == (n_nodes,)
    assert isinstance(R, np.ndarray) and R.shape == (1,)
    assert isinstance(x, np.ndarray) and x.shape == (n_nodes,)
    assert isinstance(r_nodes, np.ndarray) and r_nodes.shape == (1,)
    assert np.allclose(u, 0.0, atol=1e-12, rtol=0.0)
    assert np.isclose(R[0], 0.0, atol=1e-12, rtol=0.0)
    fixed_node_idx = r_nodes[0]
    assert 0 <= fixed_node_idx < n_nodes
    assert np.isclose(x[fixed_node_idx], x_min, atol=1e-12, rtol=0.0)
    assert np.isclose(u[fixed_node_idx], 0.0, atol=1e-12, rtol=0.0)

def test_analytical_solution(fcn):
    """
    Test a non-zero displacement field against a known analytical solution.
    Uniform bar with E, A, length L; left end fixed (u=0) and a point load P at the right end.
    No body force. The exact displacement is u(x) = (P / (E*A)) * x, which is linear.
    """
    x_min = 0.0
    L = 3.0
    x_max = L
    num_elements = 6
    E = 100.0
    A = 2.0
    P = 50.0
    material_regions = [{'coord_min': x_min, 'coord_max': x_max, 'E': E, 'A': A}]

    def body_force_fn(x):
        return 0.0
    dirichlet_bc_list = [{'x_location': x_min, 'u_prescribed': 0.0}]
    neumann_bc_list = [{'x_location': x_max, 'load_mag': P}]
    n_gauss = 2
    result = fcn(x_min=x_min, x_max=x_max, num_elements=num_elements, material_regions=material_regions, body_force_fn=body_force_fn, dirichlet_bc_list=dirichlet_bc_list, neumann_bc_list=neumann_bc_list, n_gauss=n_gauss)
    x = result['node_coords']
    u = result['displacements']
    u_exact = P / (E * A) * x
    assert np.allclose(u, u_exact, rtol=1e-12, atol=1e-12)
    R = result['reactions']
    r_nodes = result['reaction_nodes']
    assert R.shape == (1,)
    assert r_nodes.shape == (1,)
    assert np.isclose(R[0], -P, rtol=1e-12, atol=1e-12)
    fixed_node_idx = r_nodes[0]
    assert np.isclose(x[fixed_node_idx], x_min, atol=1e-12, rtol=0.0)