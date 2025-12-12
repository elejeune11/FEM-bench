def test_no_load_self_contained(fcn):
    """Test zero displacement and zero reaction in a fixed-free bar with no external load.
    A 2-element bar with uniform material, zero body force, and one fixed end should return:
    """
    x_min = 0.0
    x_max = 1.0
    num_elements = 2
    material_regions = [{'coord_min': x_min, 'coord_max': x_max, 'E': 200000000000.0, 'A': 1.0}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    neumann_bc_list = None
    n_gauss = 2
    result = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
    assert isinstance(result, dict)
    assert 'displacements' in result and 'reactions' in result and ('node_coords' in result) and ('reaction_nodes' in result)
    disp = result['displacements']
    reactions = result['reactions']
    nodes = result['node_coords']
    reaction_nodes = result['reaction_nodes']
    assert isinstance(disp, np.ndarray) and isinstance(reactions, np.ndarray)
    assert isinstance(nodes, np.ndarray) and isinstance(reaction_nodes, np.ndarray)
    assert disp.shape == (num_elements + 1,)
    assert nodes.shape == (num_elements + 1,)
    assert reactions.shape == (1,)
    assert reaction_nodes.shape == (1,)
    assert np.allclose(disp, 0.0, atol=1e-12)
    assert np.allclose(reactions, 0.0, atol=1e-12)
    assert reaction_nodes[0] == 0
    assert np.isclose(nodes[0], x_min) and np.isclose(nodes[-1], x_max)

def test_analytical_solution(fcn):
    """Test a non-zero displacement field against a known analytical solution.
    For a bar of length L with constant EA and constant body force b, fixed at x=0 and
    free at x=L, the analytic solution is u(x) = b/(EA) * (L*x - x^2/2). The reaction at
    the fixed node equals -b*L.
    """
    L = 1.0
    x_min = 0.0
    x_max = L
    num_elements = 8
    E = 200.0
    A = 1.0
    EA = E * A
    b = 1.0
    material_regions = [{'coord_min': x_min, 'coord_max': x_max, 'E': E, 'A': A}]
    body_force_fn = lambda x: float(b)
    dirichlet_bc_list = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    neumann_bc_list = None
    n_gauss = 2
    result = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
    disp = result['displacements']
    reactions = result['reactions']
    nodes = result['node_coords']
    reaction_nodes = result['reaction_nodes']
    assert disp.shape == (num_elements + 1,)
    assert nodes.shape == (num_elements + 1,)
    assert reactions.shape == (1,)
    assert reaction_nodes.shape == (1,)
    assert reaction_nodes[0] == 0
    x = nodes
    u_exact = b / EA * (L * x - 0.5 * x ** 2)
    assert np.allclose(disp, u_exact, atol=1e-08, rtol=1e-08)
    expected_reaction = -b * L
    assert np.allclose(reactions[0], expected_reaction, atol=1e-08, rtol=1e-08)