def test_no_load_self_contained(fcn):
    """
    Test zero displacement and zero reaction in a fixed-free bar with no external load.
    A 2-element bar with uniform material, zero body force, and one fixed end should return:
    """
    x_min = 0.0
    x_max = 1.0
    num_elements = 2
    E = 100.0
    A = 1.0
    material_regions = [{'coord_min': x_min, 'coord_max': x_max, 'E': E, 'A': A}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': x_min, 'u_prescribed': 0.0}]
    neumann_bc_list = None
    n_gauss = 2
    result = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
    displacements = result['displacements']
    reactions = result['reactions']
    node_coords = result['node_coords']
    reaction_nodes = result['reaction_nodes']
    n_nodes = num_elements + 1
    assert displacements.shape == (n_nodes,)
    assert node_coords.shape == (n_nodes,)
    assert reactions.shape == (len(dirichlet_bc_list),)
    assert reaction_nodes.shape == (len(dirichlet_bc_list),)
    assert np.isclose(node_coords[0], x_min)
    assert np.isclose(node_coords[-1], x_max)
    assert np.all(np.diff(node_coords) > 0)
    assert np.allclose(displacements, 0.0, atol=1e-12)
    assert np.allclose(reactions, 0.0, atol=1e-12)
    idx_fixed = np.argmin(np.abs(node_coords - x_min))
    assert reaction_nodes[0] == idx_fixed
    assert np.isclose(displacements[idx_fixed], 0.0, atol=1e-12)

def test_analytical_solution(fcn):
    """
    Test a non-zero displacement field against a known analytical solution.
    For a uniform bar (constant E and A), fixed at x_min and with a point load P applied at x_max,
    and zero body force, the analytical solution is u(x) = (P / (E*A)) * (x - x_min).
    The linear FE solution should match this exactly at the nodes.
    """
    x_min = 0.0
    x_max = 2.0
    num_elements = 5
    E = 2.0
    A = 3.0
    P = 6.0
    material_regions = [{'coord_min': x_min, 'coord_max': x_max, 'E': E, 'A': A}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': x_min, 'u_prescribed': 0.0}]
    neumann_bc_list = [{'x_location': x_max, 'load_mag': P}]
    n_gauss = 2
    result = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss)
    displacements = result['displacements']
    node_coords = result['node_coords']
    factor = P / (E * A)
    expected = factor * (node_coords - x_min)
    assert displacements.shape == node_coords.shape
    assert np.allclose(displacements, expected, rtol=1e-12, atol=1e-12)