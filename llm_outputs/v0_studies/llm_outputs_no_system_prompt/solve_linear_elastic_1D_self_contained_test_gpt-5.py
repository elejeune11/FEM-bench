def test_no_load_self_contained(fcn):
    """Test zero displacement and zero reaction in a fixed-free bar with no external load.
    A 2-element bar with uniform material, zero body force, and one fixed end should return:
    """
    (x_min, x_max) = (0.0, 1.0)
    num_elements = 2
    material_regions = [{'coord_min': x_min, 'coord_max': x_max, 'E': 100.0, 'A': 1.0}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': x_min, 'u_prescribed': 0.0}]
    neumann_bc_list = None
    n_gauss = 2
    result = fcn(x_min=x_min, x_max=x_max, num_elements=num_elements, material_regions=material_regions, body_force_fn=body_force_fn, dirichlet_bc_list=dirichlet_bc_list, neumann_bc_list=neumann_bc_list, n_gauss=n_gauss)
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
    assert np.all(node_coords[1:] > node_coords[:-1])
    assert np.allclose(displacements, 0.0, atol=1e-12)
    assert np.allclose(reactions, 0.0, atol=1e-12)
    assert reaction_nodes.size == 1
    assert reaction_nodes[0] == 0
    assert np.isclose(displacements[reaction_nodes[0]], 0.0, atol=1e-12)

def test_uniform_extension_analytical_self_contained(fcn):
    """Test displacement field against a known analytical solution for a bar under end load.
    A uniform bar fixed at x=0 with a point load P at x=L has u(x) = P*x/(E*A).
    Verify FE nodal displacements match the analytical solution.
    """
    (x_min, x_max) = (0.0, 2.0)
    L = x_max - x_min
    num_elements = 8
    E = 210.0
    A = 3.0
    P = 7.5
    material_regions = [{'coord_min': x_min, 'coord_max': x_max, 'E': E, 'A': A}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': x_min, 'u_prescribed': 0.0}]
    neumann_bc_list = [{'x_location': x_max, 'load_mag': P}]
    n_gauss = 2
    result = fcn(x_min=x_min, x_max=x_max, num_elements=num_elements, material_regions=material_regions, body_force_fn=body_force_fn, dirichlet_bc_list=dirichlet_bc_list, neumann_bc_list=neumann_bc_list, n_gauss=n_gauss)
    displacements = result['displacements']
    node_coords = result['node_coords']
    assert displacements.shape == (num_elements + 1,)
    assert node_coords.shape == (num_elements + 1,)
    assert np.isclose(node_coords[0], x_min)
    assert np.isclose(node_coords[-1], x_max)
    u_exact = P / (E * A) * node_coords
    assert np.allclose(displacements, u_exact, rtol=1e-10, atol=1e-12)
    assert np.isclose(displacements[0], 0.0, atol=1e-12)