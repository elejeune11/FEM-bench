def test_no_load_self_contained(fcn):
    """
    Test zero displacement and zero reaction in a fixed-free bar with no external load.
    A 2-element bar with uniform material, zero body force, and one fixed end should return:
    """
    (x_min, x_max) = (0.0, 1.0)
    num_elements = 2
    material_regions = [{'coord_min': x_min, 'coord_max': x_max, 'E': 100.0, 'A': 1.0}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': x_min, 'u_prescribed': 0.0}]
    neumann_bc_list = []
    results = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss=2)
    u = results['displacements']
    reactions = results['reactions']
    coords = results['node_coords']
    r_nodes = results['reaction_nodes']
    num_nodes = num_elements + 1
    assert u.shape == (num_nodes,)
    assert coords.shape == (num_nodes,)
    assert reactions.shape == (1,)
    assert r_nodes.shape == (1,)
    assert np.isclose(coords[0], x_min)
    assert np.isclose(coords[-1], x_max)
    assert np.all(np.diff(coords) > 0)
    np.testing.assert_allclose(u, 0.0, atol=1e-15, err_msg='Displacements should be zero for no load.')
    np.testing.assert_allclose(reactions, 0.0, atol=1e-15, err_msg='Reaction should be zero for no load.')

def test_analytical_solution(fcn):
    """Test a non-zero displacement field against a known analytical solution."""
    (x_min, x_max) = (0.0, 2.0)
    L = x_max - x_min
    num_elements = 10
    E = 200.0
    A = 0.5
    f_val = 10.0
    EA = E * A
    material_regions = [{'coord_min': x_min, 'coord_max': x_max, 'E': E, 'A': A}]
    body_force_fn = lambda x: f_val
    dirichlet_bc_list = [{'x_location': x_min, 'u_prescribed': 0.0}]
    neumann_bc_list = []
    results = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, dirichlet_bc_list, neumann_bc_list, n_gauss=2)
    u_fem = results['displacements']
    x_nodes = results['node_coords']
    reactions = results['reactions']
    u_exact = f_val / EA * (L * x_nodes - 0.5 * x_nodes ** 2)
    np.testing.assert_allclose(u_fem, u_exact, rtol=1e-05, atol=1e-08, err_msg='FEM displacements do not match analytical solution.')
    expected_reaction = -(f_val * L)
    np.testing.assert_allclose(reactions, [expected_reaction], rtol=1e-05, atol=1e-08, err_msg='Reaction force does not match equilibrium requirement.')