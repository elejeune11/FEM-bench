def test_no_load_self_contained(fcn):
    """Test zero displacement and zero reaction in a fixed-free bar with no external load.
    A 2-element bar with uniform material, zero body force, and one fixed end should return:
    """
    x_min = 0.0
    x_max = 2.0
    num_elements = 2
    material_regions = [{'coord_min': 0.0, 'coord_max': 2.0, 'E': 200.0, 'A': 1.0}]

    def zero_body_force(x: float) -> float:
        return 0.0
    dirichlet_bc_list = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    neumann_bc_list = []
    n_gauss = 2
    result = fcn(x_min=x_min, x_max=x_max, num_elements=num_elements, material_regions=material_regions, body_force_fn=zero_body_force, dirichlet_bc_list=dirichlet_bc_list, neumann_bc_list=neumann_bc_list, n_gauss=n_gauss)
    assert 'displacements' in result
    assert 'reactions' in result
    assert 'node_coords' in result
    assert 'reaction_nodes' in result
    n_nodes = num_elements + 1
    assert result['displacements'].shape == (n_nodes,)
    assert result['reactions'].shape == (1,)
    assert result['node_coords'].shape == (n_nodes,)
    assert result['reaction_nodes'].shape == (1,)
    assert np.allclose(result['displacements'], 0.0, atol=1e-12)
    assert np.allclose(result['reactions'], 0.0, atol=1e-12)
    expected_coords = np.array([0.0, 1.0, 2.0])
    assert np.allclose(result['node_coords'], expected_coords, atol=1e-12)
    assert result['reaction_nodes'][0] == 0
    assert np.isclose(result['displacements'][0], 0.0, atol=1e-12)

def test_analytical_solution(fcn):
    """Test a non-zero displacement field against a known analytical solution."""
    x_min = 0.0
    x_max = 1.0
    L = x_max - x_min
    num_elements = 4
    E = 100.0
    A = 2.0
    P = 10.0
    material_regions = [{'coord_min': 0.0, 'coord_max': 1.0, 'E': E, 'A': A}]

    def zero_body_force(x: float) -> float:
        return 0.0
    dirichlet_bc_list = [{'x_location': 0.0, 'u_prescribed': 0.0}]
    neumann_bc_list = [{'x_location': x_max, 'load_mag': P}]
    n_gauss = 2
    result = fcn(x_min=x_min, x_max=x_max, num_elements=num_elements, material_regions=material_regions, body_force_fn=zero_body_force, dirichlet_bc_list=dirichlet_bc_list, neumann_bc_list=neumann_bc_list, n_gauss=n_gauss)
    node_coords = result['node_coords']
    analytical_displacements = P * node_coords / (E * A)
    assert np.allclose(result['displacements'], analytical_displacements, rtol=1e-10, atol=1e-12)
    expected_reaction = -P
    assert np.allclose(result['reactions'][0], expected_reaction, rtol=1e-10, atol=1e-12)
    u_tip_analytical = P * L / (E * A)
    assert np.isclose(result['displacements'][-1], u_tip_analytical, rtol=1e-10, atol=1e-12)
    assert np.isclose(result['displacements'][0], 0.0, atol=1e-12)