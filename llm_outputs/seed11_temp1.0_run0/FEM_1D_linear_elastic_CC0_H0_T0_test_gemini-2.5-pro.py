def test_no_load_self_contained(fcn: Callable):
    """Test zero displacement and zero reaction in a fixed-free bar with no external load.
    A 2-element bar with uniform material, zero body force, and one fixed end should return:
    """
    x_min = 0.0
    x_max = 1.0
    num_elements = 2
    num_nodes = num_elements + 1
    material_regions = [{'coord_min': 0.0, 'coord_max': 1.0, 'E': 10.0, 'A': 2.0}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    neumann_bc_list = None
    n_gauss = 2
    result = fcn(x_min=x_min, x_max=x_max, num_elements=num_elements, material_regions=material_regions, body_force_fn=body_force_fn, dirichlet_bc_list=dirichlet_bc_list, neumann_bc_list=neumann_bc_list, n_gauss=n_gauss)
    assert result['displacements'].shape == (num_nodes,)
    assert result['reactions'].shape == (len(dirichlet_bc_list),)
    assert result['node_coords'].shape == (num_nodes,)
    assert result['reaction_nodes'].shape == (len(dirichlet_bc_list),)
    expected_displacements = np.zeros(num_nodes)
    expected_reactions = np.zeros(len(dirichlet_bc_list))
    expected_node_coords = np.linspace(x_min, x_max, num_nodes)
    expected_reaction_nodes = np.array([0])
    assert np.allclose(result['displacements'], expected_displacements, atol=1e-09)
    assert np.allclose(result['reactions'], expected_reactions, atol=1e-09)
    assert np.allclose(result['node_coords'], expected_node_coords)
    assert np.array_equal(result['reaction_nodes'], expected_reaction_nodes)
    assert result['displacements'][0] == dirichlet_bc_list[0]['u_prescribed']

def test_analytical_solution(fcn: Callable):
    """Test a non-zero displacement field against a known analytical solution."""
    x_min = 0.0
    L = 2.0
    x_max = L
    num_elements = 4
    num_nodes = num_elements + 1
    E = 1.0
    A = 1.0
    f = 1.0
    material_regions = [{'coord_min': x_min, 'coord_max': x_max, 'E': E, 'A': A}]
    body_force_fn = lambda x: f
    dirichlet_bc_list = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    neumann_bc_list = None
    n_gauss = 2
    result = fcn(x_min=x_min, x_max=x_max, num_elements=num_elements, material_regions=material_regions, body_force_fn=body_force_fn, dirichlet_bc_list=dirichlet_bc_list, neumann_bc_list=neumann_bc_list, n_gauss=n_gauss)
    node_coords = result['node_coords']
    expected_node_coords = np.linspace(x_min, x_max, num_nodes)
    assert np.allclose(node_coords, expected_node_coords)
    u_analytical = f / (E * A) * (L * node_coords - node_coords ** 2 / 2)
    assert np.allclose(result['displacements'], u_analytical)
    expected_reaction = -f * L
    assert np.allclose(result['reactions'], [expected_reaction])
    expected_reaction_nodes = np.array([0])
    assert np.array_equal(result['reaction_nodes'], expected_reaction_nodes)