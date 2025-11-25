def test_no_load_self_contained(fcn):
    """
    Test zero displacement and zero reaction in a fixed-free bar with no external load.
    A 2-element bar with uniform material, zero body force, and one fixed end should return:
    """
    x_min = 0.0
    x_max = 2.0
    num_elements = 2
    E = 100.0
    A = 1.0
    material_regions = [{'coord_min': x_min, 'coord_max': x_max, 'E': E, 'A': A}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': x_min, 'u_prescribed': 0.0}]
    neumann_bc_list = None
    n_gauss = 2
    out = fcn(x_min=x_min, x_max=x_max, num_elements=num_elements, material_regions=material_regions, body_force_fn=body_force_fn, dirichlet_bc_list=dirichlet_bc_list, neumann_bc_list=neumann_bc_list, n_gauss=n_gauss)
    assert isinstance(out, dict)
    assert 'displacements' in out and 'reactions' in out and ('node_coords' in out) and ('reaction_nodes' in out)
    u = out['displacements']
    r = out['reactions']
    x = out['node_coords']
    rc = out['reaction_nodes']
    n_nodes = num_elements + 1
    assert isinstance(u, np.ndarray) and u.shape == (n_nodes,)
    assert isinstance(x, np.ndarray) and x.shape == (n_nodes,)
    assert isinstance(rc, np.ndarray) and rc.shape == (len(dirichlet_bc_list),)
    assert isinstance(r, np.ndarray) and r.shape == (len(dirichlet_bc_list),)
    assert np.isclose(x[0], x_min)
    assert np.isclose(x[-1], x_max)
    assert np.isclose(x[rc[0]], x_min)
    assert np.isclose(u[rc[0]], dirichlet_bc_list[0]['u_prescribed'])
    assert np.allclose(u, 0.0, atol=1e-12, rtol=0.0)
    assert np.allclose(r, 0.0, atol=1e-12, rtol=0.0)

def test_uniform_extension_analytical_self_contained(fcn):
    """
    Test displacement field against a known analytical solution.
    For a uniform bar fixed at the left end with a tensile point load P at the right end,
    the analytical displacement is u(x) = P*x/(E*A). Verify FE solution matches.
    """
    x_min = 0.0
    x_max = 3.0
    L = x_max - x_min
    num_elements = 6
    E = 200.0
    A = 2.0
    P = 5.0
    material_regions = [{'coord_min': x_min, 'coord_max': x_max, 'E': E, 'A': A}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': x_min, 'u_prescribed': 0.0}]
    neumann_bc_list = [{'x_location': x_max, 'load_mag': P}]
    n_gauss = 3
    out = fcn(x_min=x_min, x_max=x_max, num_elements=num_elements, material_regions=material_regions, body_force_fn=body_force_fn, dirichlet_bc_list=dirichlet_bc_list, neumann_bc_list=neumann_bc_list, n_gauss=n_gauss)
    u = out['displacements']
    r = out['reactions']
    x = out['node_coords']
    rc = out['reaction_nodes']
    n_nodes = num_elements + 1
    assert isinstance(u, np.ndarray) and u.shape == (n_nodes,)
    assert isinstance(x, np.ndarray) and x.shape == (n_nodes,)
    assert isinstance(rc, np.ndarray) and rc.shape == (len(dirichlet_bc_list),)
    assert isinstance(r, np.ndarray) and r.shape == (len(dirichlet_bc_list),)
    assert np.isclose(x[rc[0]], x_min)
    assert np.isclose(u[rc[0]], dirichlet_bc_list[0]['u_prescribed'], atol=1e-12)
    u_exact = P * x / (E * A)
    assert np.allclose(u, u_exact, rtol=1e-09, atol=1e-12)
    assert np.all(np.diff(u) >= -1e-12)
    assert np.isclose(u[-1], P * L / (E * A), rtol=1e-09, atol=1e-12)
    assert np.isclose(abs(r[0]), abs(P), rtol=1e-09, atol=1e-12)