def test_no_load_self_contained(fcn):
    """
    Test zero displacement and zero reaction in a fixed-free bar with no external load.
    A 2-element bar with uniform material, zero body force, and one fixed end should return:
    """
    x_min = 0.0
    x_max = 1.0
    num_elements = 2
    E = 210.0
    A = 1.0
    material_regions = [{'coord_min': x_min, 'coord_max': x_max, 'E': E, 'A': A}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': x_min, 'u_prescribed': 0.0}]
    neumann_bc_list = None
    n_gauss = 2
    result = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
    u = result['displacements']
    reactions = result['reactions']
    x_nodes = result['node_coords']
    reaction_nodes = result['reaction_nodes']
    assert isinstance(u, np.ndarray)
    assert isinstance(reactions, np.ndarray)
    assert isinstance(x_nodes, np.ndarray)
    assert isinstance(reaction_nodes, np.ndarray)
    assert u.shape == (num_elements + 1,)
    assert x_nodes.shape == (num_elements + 1,)
    assert reactions.shape == (len(dirichlet_bc_list),)
    assert reaction_nodes.shape == (len(dirichlet_bc_list),)
    assert np.issubdtype(reaction_nodes.dtype, np.integer)
    expected_coords = np.linspace(x_min, x_max, num_elements + 1)
    assert np.allclose(x_nodes, expected_coords)
    assert reaction_nodes[0] == 0
    assert np.isclose(u[0], dirichlet_bc_list[0]['u_prescribed'])
    assert np.allclose(u, 0.0, atol=1e-12)
    assert np.allclose(reactions, 0.0, atol=1e-12)

def test_analytical_solution(fcn):
    """
    Test a non-zero displacement field against a known analytical solution.
    For a bar with constant EA, fixed at x=0, and a point load P at x=L, the exact solution is:
    u(x) = (P / (E*A)) * x. Reactions must balance applied loads.
    """
    x_min = 0.0
    x_max = 1.0
    E = 200.0
    A = 5.0
    P = 50.0
    material_regions_template = [{'coord_min': x_min, 'coord_max': x_max, 'E': E, 'A': A}]
    body_force_fn = lambda x: 0.0
    for num_elements in [1, 2, 5, 10]:
        for n_gauss in [1, 2, 3]:
            dirichlet_bc_list = [{'x_location': x_min, 'u_prescribed': 0.0}]
            neumann_bc_list = [{'x_location': x_max, 'load_mag': P}]
            result = fcn(x_min, x_max, num_elements, material_regions_template, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
            u = result['displacements']
            reactions = result['reactions']
            x_nodes = result['node_coords']
            reaction_nodes = result['reaction_nodes']
            expected_u = P / (E * A) * x_nodes
            assert np.allclose(u, expected_u, rtol=1e-12, atol=1e-12)
            assert reaction_nodes.shape == (1,)
            assert reaction_nodes[0] == 0
            assert np.isclose(u[0], 0.0, atol=1e-12)
            total_reaction = np.sum(reactions)
            total_point_loads = sum((item['load_mag'] for item in neumann_bc_list))
            total_body_force = 0.0
            assert np.isclose(total_reaction + total_point_loads + total_body_force, 0.0, atol=1e-10)