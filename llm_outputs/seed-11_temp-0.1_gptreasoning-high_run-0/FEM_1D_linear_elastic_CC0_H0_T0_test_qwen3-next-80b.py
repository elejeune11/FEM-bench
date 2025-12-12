def test_no_load_self_contained(fcn):
    (x_min, x_max) = (0.0, 2.0)
    num_elements = 2
    material_regions = [{'coord_min': 0.0, 'coord_max': 2.0, 'E': 1.0, 'A': 1.0}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    neumann_bc_list = None
    n_gauss = 2
    result = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
    displacements = result['displacements']
    reactions = result['reactions']
    node_coords = result['node_coords']
    reaction_nodes = result['reaction_nodes']
    assert len(displacements) == num_elements + 1
    assert len(reactions) == 1
    assert len(node_coords) == num_elements + 1
    assert len(reaction_nodes) == 1
    assert np.allclose(displacements, 0.0)
    assert np.allclose(reactions, 0.0)
    assert np.isclose(node_coords[0], 0.0)
    assert reaction_nodes[0] == 0

def test_analytical_solution(fcn):
    (x_min, x_max) = (0.0, 1.0)
    num_elements = 4
    material_regions = [{'coord_min': 0.0, 'coord_max': 1.0, 'E': 1.0, 'A': 1.0}]
    body_force_fn = lambda x: 1.0
    dirichlet_bc_list = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    neumann_bc_list = [{'x_location': 1.0, 'load_mag': 0.0}]
    n_gauss = 2
    result = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
    displacements = result['displacements']
    node_coords = result['node_coords']
    n_nodes = num_elements + 1
    exact_displacements = np.array([0.5 * x * (1 - x) for x in node_coords])
    assert len(displacements) == n_nodes
    assert np.allclose(displacements, exact_displacements, atol=1e-06)