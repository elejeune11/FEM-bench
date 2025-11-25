def test_no_load_self_contained(fcn):
    """Test zero displacement and zero reaction in a fixed-free bar with no external load.
    A 2-element bar with uniform material, zero body force, and one fixed end should return:
    """
    x_min = 0.0
    x_max = 2.0
    num_elements = 2
    material_regions = [{'coord_min': x_min, 'coord_max': x_max, 'E': 100.0, 'A': 1.0}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': x_min, 'u_prescribed': 0.0}]
    neumann_bc_list = None
    n_gauss = 2
    res = fcn(x_min=x_min, x_max=x_max, num_elements=num_elements, material_regions=material_regions, body_force_fn=body_force_fn, dirichlet_bc_list=dirichlet_bc_list, neumann_bc_list=neumann_bc_list, n_gauss=n_gauss)
    u = res['displacements']
    R = res['reactions']
    x = res['node_coords']
    rnodes = res['reaction_nodes']
    tol = 1e-12
    assert u.shape == (num_elements + 1,)
    assert x.shape == (num_elements + 1,)
    assert R.shape == (len(dirichlet_bc_list),)
    assert rnodes.shape == (len(dirichlet_bc_list),)
    for ui in u:
        assert abs(ui - 0.0) <= tol
    assert abs(R[0] - 0.0) <= tol
    fixed_index = None
    for (i, xi) in enumerate(x):
        if abs(xi - x_min) < tol:
            fixed_index = i
            break
    assert fixed_index is not None
    assert rnodes[0] == fixed_index
    assert abs(u[fixed_index] - dirichlet_bc_list[0]['u_prescribed']) <= tol

def test_uniform_extension_analytical_self_contained(fcn):
    """Test displacement field against a known analytical solution for a fixed-free bar under end load."""
    x_min = 0.0
    L = 5.0
    x_max = L
    num_elements = 10
    E = 1000.0
    A = 2.0
    P = 10.0
    material_regions = [{'coord_min': x_min, 'coord_max': x_max, 'E': E, 'A': A}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{'x_location': x_min, 'u_prescribed': 0.0}]
    neumann_bc_list = [{'x_location': x_max, 'load_mag': P}]
    n_gauss = 2
    res = fcn(x_min=x_min, x_max=x_max, num_elements=num_elements, material_regions=material_regions, body_force_fn=body_force_fn, dirichlet_bc_list=dirichlet_bc_list, neumann_bc_list=neumann_bc_list, n_gauss=n_gauss)
    u = res['displacements']
    R = res['reactions']
    x = res['node_coords']
    rnodes = res['reaction_nodes']
    tol = 1e-12
    assert u.shape == (num_elements + 1,)
    assert x.shape == (num_elements + 1,)
    assert R.shape == (len(dirichlet_bc_list),)
    assert rnodes.shape == (len(dirichlet_bc_list),)
    for (i, xi) in enumerate(x):
        u_exact = P * xi / (E * A)
        assert abs(u[i] - u_exact) <= tol
    assert abs(R[0] + P) <= tol
    fixed_index = None
    for (i, xi) in enumerate(x):
        if abs(xi - x_min) < tol:
            fixed_index = i
            break
    assert fixed_index is not None
    assert rnodes[0] == fixed_index
    assert abs(u[fixed_index] - dirichlet_bc_list[0]['u_prescribed']) <= tol