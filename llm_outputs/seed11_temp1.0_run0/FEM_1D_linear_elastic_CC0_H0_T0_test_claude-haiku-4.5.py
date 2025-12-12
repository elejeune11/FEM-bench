def test_no_load_self_contained(fcn):
    """Test zero displacement and zero reaction in a fixed-free bar with no external load.
    A 2-element bar with uniform material, zero body force, and one fixed end should return:
    """
    x_min = 0.0
    x_max = 1.0
    num_elements = 2
    material_regions = [{'coord_min': 0.0, 'coord_max': 1.0, 'E': 1.0, 'A': 1.0}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    neumann_bc_list = None
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
    assert displacements.shape == (3,)
    assert np.allclose(displacements, 0.0)
    assert np.allclose(reactions, 0.0)
    assert node_coords.shape == (3,)
    assert np.allclose(node_coords, [0.0, 0.5, 1.0])
    assert len(reaction_nodes) == len(dirichlet_bc_list)

def test_analytical_solution(fcn):
    """Test a non-zero displacement field against a known analytical solution.
    For a bar with unit length, uniform properties (E=1, A=1), fixed at x=0,
    and constant body force f(x)=1, the analytical solution for displacement is:
    u(x) = x^2/2 (maximum at x=1 is 0.5).
    """
    x_min = 0.0
    x_max = 1.0
    num_elements = 10
    material_regions = [{'coord_min': 0.0, 'coord_max': 1.0, 'E': 1.0, 'A': 1.0}]
    body_force_fn = lambda x: 1.0
    dirichlet_bc_list = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    neumann_bc_list = None
    n_gauss = 2
    result = fcn(x_min=x_min, x_max=x_max, num_elements=num_elements, material_regions=material_regions, body_force_fn=body_force_fn, dirichlet_bc_list=dirichlet_bc_list, neumann_bc_list=neumann_bc_list, n_gauss=n_gauss)
    displacements = result['displacements']
    node_coords = result['node_coords']
    analytical = 0.5 * node_coords ** 2
    assert np.allclose(displacements, analytical, atol=0.01)
    assert np.isclose(displacements[0], 0.0)
    assert np.isclose(displacements[-1], 0.5, atol=0.01)