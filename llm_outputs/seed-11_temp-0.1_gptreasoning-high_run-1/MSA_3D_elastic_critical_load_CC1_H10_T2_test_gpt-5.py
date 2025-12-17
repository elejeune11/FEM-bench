def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever value analytical solution.
    Use 10 elements with a relative tolerance suitable for discretization with linear beam elements.
    """
    E = 210000000000.0
    nu = 0.3
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    n_elems = 10
    tol = 0.02
    for r in radii:
        A = np.pi * r ** 2
        I = 0.25 * np.pi * r ** 4
        J = 0.5 * np.pi * r ** 4
        for L in lengths:
            n_nodes = n_elems + 1
            z = np.linspace(0.0, L, n_nodes)
            node_coords = np.column_stack((np.zeros_like(z), np.zeros_like(z), z))
            elements = []
            for i in range(n_elems):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([0.0, 1.0, 0.0])})
            boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
            nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
            lam, _ = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            P_euler = np.pi ** 2 * E * I / (4.0 * L ** 2)
            rel_err = abs(lam - P_euler) / P_euler
            assert rel_err < tol

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    Solve the cantilever model in its base orientation and again after applying a rigid-body
    rotation R to geometry, element axes, and applied load. The critical load factor λ should
    match. The rotated buckling mode should equal the base mode transformed by T built from R.
    """
    E = 210000000000.0
    nu = 0.3
    b = 0.08
    h = 0.12
    A = b * h
    I_y = b * h ** 3 / 12.0
    I_z = h * b ** 3 / 12.0
    J = I_y + I_z
    L = 12.5
    n_elems = 12
    n_nodes = n_elems + 1
    z = np.linspace(0.0, L, n_nodes)
    node_coords = np.column_stack((np.zeros_like(z), np.zeros_like(z), z))
    elements = []
    for i in range(n_elems):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 1.0, 0.0])})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
    lam_base, mode_base = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    alpha = 0.37
    c, s = (np.cos(alpha), np.sin(alpha))
    R = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    node_coords_rot = node_coords @ R.T
    local_z_base = np.array([0.0, 1.0, 0.0])
    local_z_rot = R @ local_z_base
    elements_rot = []
    for i in range(n_elems):
        elements_rot.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_rot})
    F_base = np.array([0.0, 0.0, -1.0])
    F_rot = R @ F_base
    nodal_loads_rot = {n_nodes - 1: [F_rot[0], F_rot[1], F_rot[2], 0.0, 0.0, 0.0]}
    lam_rot, mode_rot = fcn(node_coords_rot, elements_rot, boundary_conditions, nodal_loads_rot)
    rel_diff_lam = abs(lam_rot - lam_base) / abs(lam_base)
    assert rel_diff_lam < 5e-09
    T = np.zeros((6 * n_nodes, 6 * n_nodes))
    for i in range(n_nodes):
        idx = 6 * i
        T[idx:idx + 3, idx:idx + 3] = R
        T[idx + 3:idx + 6, idx + 3:idx + 6] = R
    mapped_mode = T @ mode_base
    denom = np.dot(mapped_mode, mapped_mode)
    if denom > 0:
        s_opt = np.dot(mode_rot, mapped_mode) / denom
        diff = mode_rot - s_opt * mapped_mode
        rel_mode_err = np.linalg.norm(diff) / max(np.linalg.norm(mode_rot), 1e-16)
        assert rel_mode_err < 2e-06
    else:
        assert np.linalg.norm(mode_rot) < 1e-12

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    Refine the discretization and check that the numerical critical load approaches the
    analytical value with decreasing relative error, and the finest mesh is highly accurate.
    """
    E = 210000000000.0
    nu = 0.3
    r = 0.4
    A = np.pi * r ** 2
    I = 0.25 * np.pi * r ** 4
    J = 0.5 * np.pi * r ** 4
    L = 10.0
    meshes = [4, 8, 16, 32]
    errors = []
    P_euler = np.pi ** 2 * E * I / (4.0 * L ** 2)
    for n_elems in meshes:
        n_nodes = n_elems + 1
        z = np.linspace(0.0, L, n_nodes)
        node_coords = np.column_stack((np.zeros_like(z), np.zeros_like(z), z))
        elements = []
        for i in range(n_elems):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([0.0, 1.0, 0.0])})
        boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
        nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
        lam, _ = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        rel_err = abs(lam - P_euler) / P_euler
        errors.append(rel_err)
    assert errors[-1] <= min(errors[:-1])
    assert errors[-1] < 0.002