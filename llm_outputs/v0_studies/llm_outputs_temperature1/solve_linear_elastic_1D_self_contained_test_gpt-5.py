def test_no_load_self_contained(fcn):
    """
    Test zero displacement and zero reaction in a fixed-free bar with no external load.
    A 2-element bar with uniform material, zero body force, and one fixed end should return:
    """
    import numpy as np
    x_min = 0.0
    x_max = 1.0
    num_elements = 2
    material_regions = [{'coord_min': x_min, 'coord_max': x_max, 'E': 1.0, 'A': 1.0}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': x_min, 'u_prescribed': 0.0}]
    neumann_bc_list = None
    n_gauss = 2
    res = fcn(x_min=x_min, x_max=x_max, num_elements=num_elements, material_regions=material_regions, body_force_fn=body_force_fn, dirichlet_bc_list=dirichlet_bc_list, neumann_bc_list=neumann_bc_list, n_gauss=n_gauss)
    n_nodes = num_elements + 1
    assert 'displacements' in res and 'reactions' in res and ('node_coords' in res) and ('reaction_nodes' in res)
    assert isinstance(res['displacements'], np.ndarray)
    assert isinstance(res['reactions'], np.ndarray)
    assert isinstance(res['node_coords'], np.ndarray)
    assert isinstance(res['reaction_nodes'], np.ndarray)
    assert res['displacements'].shape == (n_nodes,)
    assert res['node_coords'].shape == (n_nodes,)
    assert res['reaction_nodes'].shape == (1,)
    assert res['reactions'].shape == (1,)
    assert np.allclose(res['displacements'], 0.0, atol=1e-12, rtol=0.0)
    assert np.allclose(res['reactions'], 0.0, atol=1e-12, rtol=0.0)
    rxn_node_idx = int(res['reaction_nodes'][0])
    assert 0 <= rxn_node_idx < n_nodes
    assert np.isclose(res['node_coords'][rxn_node_idx], x_min, atol=1e-12, rtol=0.0)
    assert np.isclose(res['displacements'][rxn_node_idx], 0.0, atol=1e-12, rtol=0.0)

def test_uniform_extension_analytical_self_contained(fcn):
    """
    Test displacement field against a known analytical solution.
    For a uniform bar (E, A constant) on [0, L], with u(0)=0 and a point load P at x=L,
    the analytical solution is u(x) = P x / (E A). This should be captured exactly by linear elements.
    """
    import numpy as np
    x_min = 0.0
    x_max = 1.0
    L = x_max - x_min
    num_elements = 8
    E = 1000.0
    A = 2.0
    P = 5.0
    material_regions = [{'coord_min': x_min, 'coord_max': x_max, 'E': E, 'A': A}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': x_min, 'u_prescribed': 0.0}]
    neumann_bc_list = [{'x_location': x_max, 'load_mag': P}]
    n_gauss = 2
    res = fcn(x_min=x_min, x_max=x_max, num_elements=num_elements, material_regions=material_regions, body_force_fn=body_force_fn, dirichlet_bc_list=dirichlet_bc_list, neumann_bc_list=neumann_bc_list, n_gauss=n_gauss)
    node_coords = res['node_coords']
    displacements = res['displacements']
    u_expected = node_coords * (P / (E * A))
    assert np.allclose(displacements, u_expected, rtol=1e-09, atol=1e-12)
    total_reaction = res['reactions'].sum()
    assert np.isclose(total_reaction + P, 0.0, rtol=0.0, atol=1e-09)