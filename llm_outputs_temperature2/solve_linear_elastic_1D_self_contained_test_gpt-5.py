def test_no_load_self_contained(fcn):
    """Test zero displacement and zero reaction in a fixed-free bar with no external load.
    A 2-element bar with uniform material, zero body force, and one fixed end should return:
    """
    x_min = 0.0
    x_max = 1.0
    num_elements = 2
    material_regions = [{'coord_min': x_min, 'coord_max': x_max, 'E': 100.0, 'A': 1.0}]

    def body_force_fn(x):
        return 0.0
    dirichlet_bc_list = [{'x_location': x_min, 'u_prescribed': 0.0}]
    neumann_bc_list = None
    n_gauss = 2
    res = fcn(x_min=x_min, x_max=x_max, num_elements=num_elements, material_regions=material_regions, body_force_fn=body_force_fn, dirichlet_bc_list=dirichlet_bc_list, neumann_bc_list=neumann_bc_list, n_gauss=n_gauss)
    displacements = res['displacements']
    reactions = res['reactions']
    node_coords = res['node_coords']
    reaction_nodes = res['reaction_nodes']
    assert displacements.shape == (num_elements + 1,)
    assert node_coords.shape == (num_elements + 1,)
    assert reactions.shape == (1,)
    assert reaction_nodes.shape == (1,)
    assert np.allclose(displacements, 0.0, atol=1e-12)
    assert np.allclose(reactions, 0.0, atol=1e-12)
    idx_fixed = int(np.where(np.isclose(node_coords, x_min, atol=1e-12))[0][0])
    assert reaction_nodes[0] == idx_fixed
    assert np.isclose(displacements[idx_fixed], 0.0, atol=1e-12)

def test_uniform_extension_analytical_self_contained(fcn):
    """Test displacement field against a known analytical solution.
    Bar fixed at x=0, subjected to a point load P at x=L (Neumann BC), with zero body force.
    Analytical solution for constant E and A: u(x) = P*x / (E*A).
    """
    x_min = 0.0
    x_max = 1.0
    L = x_max - x_min
    num_elements = 10
    E = 1000.0
    A = 5.0
    EA = E * A
    P = 40.0
    material_regions = [{'coord_min': x_min, 'coord_max': x_max, 'E': E, 'A': A}]

    def body_force_fn(x):
        return 0.0
    dirichlet_bc_list = [{'x_location': x_min, 'u_prescribed': 0.0}]
    neumann_bc_list = [{'x_location': x_max, 'load_mag': P}]
    n_gauss = 2
    res = fcn(x_min=x_min, x_max=x_max, num_elements=num_elements, material_regions=material_regions, body_force_fn=body_force_fn, dirichlet_bc_list=dirichlet_bc_list, neumann_bc_list=neumann_bc_list, n_gauss=n_gauss)
    displacements = res['displacements']
    reactions = res['reactions']
    node_coords = res['node_coords']
    reaction_nodes = res['reaction_nodes']
    u_exact = P / EA * node_coords
    assert displacements.shape == node_coords.shape
    assert np.allclose(displacements, u_exact, atol=1e-10, rtol=1e-10)
    idx_fixed = int(np.where(np.isclose(node_coords, x_min, atol=1e-12))[0][0])
    assert reaction_nodes.shape == (1,)
    assert reaction_nodes[0] == idx_fixed
    assert np.isclose(displacements[idx_fixed], 0.0, atol=1e-12)
    assert reactions.shape == (1,)
    assert np.isclose(abs(reactions[0]), abs(P), rtol=1e-10, atol=1e-10)