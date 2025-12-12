def test_no_load_self_contained(fcn):
    """
    Test zero displacement and zero reaction in a fixed-free bar with no external load.
    A 2-element bar with uniform material, zero body force, and one fixed end should return:
    """
    x_min = 0.0
    x_max = 2.0
    num_elements = 2
    material_regions = [{'coord_min': x_min, 'coord_max': x_max, 'E': 200.0, 'A': 1.0}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': x_min, 'u_prescribed': 0.0}]
    neumann_bc_list = None
    n_gauss = 2
    res = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
    displacements = res['displacements']
    reactions = res['reactions']
    node_coords = res['node_coords']
    reaction_nodes = res['reaction_nodes']
    assert isinstance(displacements, np.ndarray)
    assert isinstance(reactions, np.ndarray)
    assert isinstance(node_coords, np.ndarray)
    assert isinstance(reaction_nodes, np.ndarray)
    n_nodes = num_elements + 1
    assert displacements.shape == (n_nodes,)
    assert node_coords.shape == (n_nodes,)
    assert reactions.shape == (1,)
    assert reaction_nodes.shape == (1,)
    expected_coords = np.linspace(x_min, x_max, n_nodes)
    assert np.allclose(node_coords, expected_coords, atol=1e-12)
    assert np.allclose(displacements, 0.0, atol=1e-12)
    rn = reaction_nodes[0]
    assert node_coords[rn] == pytest.approx(x_min, abs=1e-12)
    assert displacements[rn] == pytest.approx(0.0, abs=1e-12)
    assert reactions[0] == pytest.approx(0.0, abs=1e-10)

def test_analytical_solution(fcn):
    """
    Test a non-zero displacement field against a known analytical solution.
    For a uniform bar with no body force and prescribed displacements u(0)=0 and u(L)=U0,
    the analytical solution is u(x) = U0 * x / L, which should be matched exactly at nodes.
    """
    x_min = 0.0
    x_max = 1.0
    L = x_max - x_min
    num_elements = 5
    E = 100.0
    A = 2.0
    U0 = 0.01
    material_regions = [{'coord_min': x_min, 'coord_max': x_max, 'E': E, 'A': A}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': x_min, 'u_prescribed': 0.0}, {'x_location': x_max, 'u_prescribed': U0}]
    neumann_bc_list = None
    n_gauss = 2
    res = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
    displacements = res['displacements']
    node_coords = res['node_coords']
    expected = U0 * (node_coords - x_min) / L
    assert np.allclose(displacements, expected, atol=1e-10)