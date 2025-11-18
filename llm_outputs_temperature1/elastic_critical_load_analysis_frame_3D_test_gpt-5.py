def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever analytical solution.
    Use 10 elements, with tight tolerances appropriate for high-accuracy discretization.
    """
    E = 210000000000.0
    nu = 0.3
    r_values = [0.5, 0.75, 1.0]
    L_values = [10.0, 20.0, 40.0]
    n_elems = 10
    tol_rtol = 0.0001
    for r in r_values:
        A = np.pi * r ** 2
        I = np.pi * r ** 4 / 4.0
        J = np.pi * r ** 4 / 2.0
        I_rho = J
        for L in L_values:
            n_nodes = n_elems + 1
            z = np.linspace(0.0, L, n_nodes)
            node_coords = np.zeros((n_nodes, 3), dtype=float)
            node_coords[:, 2] = z
            elements = []
            for e in range(n_elems):
                ele = {'node_i': e, 'node_j': e + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'I_rho': I_rho, 'local_z': [1.0, 0.0, 0.0]}
                elements.append(ele)
            boundary_conditions = {0: [0, 1, 2, 3, 4, 5]}
            nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
            (lam, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            assert lam > 0.0
            assert mode.shape == (6 * n_nodes,)
            P_euler = np.pi ** 2 * E * I / (4.0 * L ** 2)
            rel_err = abs(lam - P_euler) / P_euler
            assert rel_err < tol_rtol

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance with a rectangular section (Iy ≠ Iz).
    The cantilever model is solved in its original orientation and again after applying
    a rigid-body rotation R to the geometry, element axes, and applied load. The critical
    load factor λ should be identical in both cases. The buckling mode from the rotated
    model should equal the base mode transformed by R (up to arbitrary scale and sign).
    """
    E = 200000000000.0
    nu = 0.3
    b = 0.05
    h = 0.2
    A = b * h
    I_y = b * h ** 3 / 12.0
    I_z = h * b ** 3 / 12.0
    J = 1e-06
    I_rho = 1e-06
    L = 12.0
    n_elems = 10
    n_nodes = n_elems + 1
    z = np.linspace(0.0, L, n_nodes)
    node_coords_base = np.zeros((n_nodes, 3), dtype=float)
    node_coords_base[:, 2] = z
    elements_base = []
    for e in range(n_elems):
        ele = {'node_i': e, 'node_j': e + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'I_rho': I_rho, 'local_z': [1.0, 0.0, 0.0]}
        elements_base.append(ele)
    boundary_conditions = {0: [0, 1, 2, 3, 4, 5]}
    nodal_loads_base = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
    (lam_base, mode_base) = fcn(node_coords_base, elements_base, boundary_conditions, nodal_loads_base)
    assert lam_base > 0.0
    assert mode_base.shape == (6 * n_nodes,)
    phi = 0.5
    c = np.cos(phi)
    s = np.sin(phi)
    R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)
    node_coords_rot = (R @ node_coords_base.T).T
    elements_rot = []
    base_local_z = np.array([1.0, 0.0, 0.0], dtype=float)
    rot_local_z = (R @ base_local_z.reshape(3, 1)).ravel()
    for e in range(n_elems):
        ele = {'node_i': e, 'node_j': e + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'I_rho': I_rho, 'local_z': rot_local_z.tolist()}
        elements_rot.append(ele)
    F_base = np.array([0.0, 0.0, -1.0], dtype=float)
    F_rot = R @ F_base
    nodal_loads_rot = {n_nodes - 1: [F_rot[0], F_rot[1], F_rot[2], 0.0, 0.0, 0.0]}
    (lam_rot, mode_rot) = fcn(node_coords_rot, elements_rot, boundary_conditions, nodal_loads_rot)
    assert lam_rot > 0.0
    assert mode_rot.shape == (6 * n_nodes,)
    assert abs(lam_rot - lam_base) / lam_base < 1e-06
    T = np.zeros((6 * n_nodes, 6 * n_nodes), dtype=float)
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    a = T @ mode_base
    denom = np.linalg.norm(a) * np.linalg.norm(mode_rot)
    if denom == 0.0:
        raise AssertionError('Mode vectors are zero; invalid test state.')
    corr = float(np.dot(mode_rot, a) / denom)
    assert abs(corr) > 0.999

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    Refine the beam discretization and check that the numerical critical load
    approaches the analytical Euler value with decreasing relative error.
    The finest mesh should achieve very high accuracy.
    """
    E = 210000000000.0
    nu = 0.3
    r = 0.3
    L = 20.0
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4.0
    J = np.pi * r ** 4 / 2.0
    I_rho = J
    meshes = [2, 4, 8, 16, 32]
    errors = []
    for n_elems in meshes:
        n_nodes = n_elems + 1
        z = np.linspace(0.0, L, n_nodes)
        node_coords = np.zeros((n_nodes, 3), dtype=float)
        node_coords[:, 2] = z
        elements = []
        for e in range(n_elems):
            ele = {'node_i': e, 'node_j': e + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'I_rho': I_rho, 'local_z': [1.0, 0.0, 0.0]}
            elements.append(ele)
        boundary_conditions = {0: [0, 1, 2, 3, 4, 5]}
        nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
        (lam, _) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        P_euler = np.pi ** 2 * E * I / (4.0 * L ** 2)
        rel_err = abs(lam - P_euler) / P_euler
        errors.append(rel_err)
    for i in range(len(errors) - 1):
        assert errors[i + 1] <= errors[i] + 1e-06
    assert errors[-1] < 1e-06