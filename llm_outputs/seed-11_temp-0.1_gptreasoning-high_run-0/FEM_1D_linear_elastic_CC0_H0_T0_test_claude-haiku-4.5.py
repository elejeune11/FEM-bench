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
    neumann_bc_list = []
    n_gauss = 2
    result = fcn(x_min=x_min, x_max=x_max, num_elements=num_elements, material_regions=material_regions, body_force_fn=body_force_fn, dirichlet_bc_list=dirichlet_bc_list, neumann_bc_list=neumann_bc_list, n_gauss=n_gauss)
    assert isinstance(result, dict)
    assert 'displacements' in result
    assert 'reactions' in result
    assert 'node_coords' in result
    assert 'reaction_nodes' in result
    displacements = result['displacements']
    reactions = result['reactions']
    node_coords = result['node_coords']
    reaction_nodes = result['reaction_nodes']
    assert displacements.shape == (num_elements + 1,)
    assert reactions.shape == (len(dirichlet_bc_list),)
    assert node_coords.shape == (num_elements + 1,)
    assert reaction_nodes.shape == (len(dirichlet_bc_list),)
    assert np.allclose(displacements, 0.0, atol=1e-10)
    assert np.allclose(reactions, 0.0, atol=1e-10)
    expected_coords = np.linspace(x_min, x_max, num_elements + 1)
    assert np.allclose(node_coords, expected_coords)
    assert reaction_nodes[0] == 0

def test_analytical_solution(fcn):
    """Test a non-zero displacement field against a known analytical solution.
    For a bar with uniform properties, fixed at x=0, and a constant body force,
    the analytical solution is u(x) = (f*L*x - f*x^2/2) / (E*A) where f is body force,
    L is length, E is Young's modulus, A is cross-sectional area.
    """
    x_min = 0.0
    x_max = 1.0
    num_elements = 4
    E = 1.0
    A = 1.0
    f = 1.0
    material_regions = [{'coord_min': 0.0, 'coord_max': 1.0, 'E': E, 'A': A}]
    body_force_fn = lambda x: f
    dirichlet_bc_list = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    neumann_bc_list = []
    n_gauss = 2
    result = fcn(x_min=x_min, x_max=x_max, num_elements=num_elements, material_regions=material_regions, body_force_fn=body_force_fn, dirichlet_bc_list=dirichlet_bc_list, neumann_bc_list=neumann_bc_list, n_gauss=n_gauss)
    displacements = result['displacements']
    node_coords = result['node_coords']
    L = x_max - x_min
    analytical_displacements = (f * L * node_coords - f * node_coords ** 2 / 2) / (E * A)
    assert np.allclose(displacements, analytical_displacements, atol=1e-06)
    assert np.isclose(displacements[0], 0.0, atol=1e-10)