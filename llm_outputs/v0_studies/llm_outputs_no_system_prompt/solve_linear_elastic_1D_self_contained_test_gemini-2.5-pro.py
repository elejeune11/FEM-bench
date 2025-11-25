def test_no_load_self_contained(fcn: Callable):
    """
    Test zero displacement and zero reaction in a fixed-free bar with no external load.
    A 2-element bar with uniform material, zero body force, and one fixed end should return:
    """
    x_min = 0.0
    x_max = 2.0
    num_elements = 2
    num_nodes = num_elements + 1
    material_regions = [{'coord_min': 0.0, 'coord_max': 2.0, 'E': 1.0, 'A': 1.0}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    neumann_bc_list = []
    n_gauss = 2
    result = fcn(x_min=x_min, x_max=x_max, num_elements=num_elements, material_regions=material_regions, body_force_fn=body_force_fn, dirichlet_bc_list=dirichlet_bc_list, neumann_bc_list=neumann_bc_list, n_gauss=n_gauss)
    displacements = result['displacements']
    reactions = result['reactions']
    node_coords = result['node_coords']
    reaction_nodes = result['reaction_nodes']
    assert displacements.shape == (num_nodes,)
    assert reactions.shape == (len(dirichlet_bc_list),)
    assert node_coords.shape == (num_nodes,)
    assert reaction_nodes.shape == (len(dirichlet_bc_list),)
    expected_displacements = np.zeros(num_nodes)
    expected_reactions = np.zeros(len(dirichlet_bc_list))
    expected_node_coords = np.linspace(x_min, x_max, num_nodes)
    expected_reaction_nodes = np.array([0])
    assert np.allclose(displacements, expected_displacements)
    assert np.allclose(reactions, expected_reactions)
    assert np.allclose(node_coords, expected_node_coords)
    assert np.array_equal(reaction_nodes, expected_reaction_nodes)
    assert np.isclose(displacements[reaction_nodes[0]], dirichlet_bc_list[0]['u_prescribed'])

def test_uniform_extension_analytical_self_contained(fcn: Callable):
    """Test displacement field against a known analytical solution."""
    L = 10.0
    E = 200.0
    A = 0.5
    P = 10.0
    num_elements = 4
    num_nodes = num_elements + 1

    def analytical_u(x: np.ndarray) -> np.ndarray:
        return P * x / (E * A)
    result = fcn(x_min=0.0, x_max=L, num_elements=num_elements, material_regions=[{'coord_min': 0.0, 'coord_max': L, 'E': E, 'A': A}], body_force_fn=lambda x: 0.0, dirichlet_bc_list=[{'x_location': 0.0, 'u_prescribed': 0.0}], neumann_bc_list=[{'x_location': L, 'load_mag': P}], n_gauss=2)
    displacements = result['displacements']
    reactions = result['reactions']
    node_coords = result['node_coords']
    reaction_nodes = result['reaction_nodes']
    assert displacements.shape == (num_nodes,)
    assert reactions.shape == (1,)
    assert node_coords.shape == (num_nodes,)
    assert reaction_nodes.shape == (1,)
    expected_node_coords = np.linspace(0.0, L, num_nodes)
    expected_displacements = analytical_u(expected_node_coords)
    expected_reaction = np.array([-P])
    expected_reaction_nodes = np.array([0])
    assert np.allclose(node_coords, expected_node_coords)
    assert np.allclose(displacements, expected_displacements)
    assert np.allclose(reactions, expected_reaction)
    assert np.array_equal(reaction_nodes, expected_reaction_nodes)