def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever
    analytical solution. Use 10 elements and a relative tolerance suitable for
    expected discretization accuracy.
    """
    E = 210000000000.0
    nu = 0.3
    r_list = [0.5, 0.75, 1.0]
    L_list = [10.0, 20.0, 40.0]
    n_elem = 10
    tol = 0.002
    for r in r_list:
        A = math.pi * r * r
        I = math.pi * r ** 4 / 4.0
        J = 2.0 * I
        for L in L_list:
            n_nodes = n_elem + 1
            z = np.linspace(0.0, L, n_nodes)
            node_coords = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
            elements = []
            for i in range(n_elem):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J})
            boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
            nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
            lam, mode = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            P_euler = math.pi ** 2 * E * I / (4.0 * L * L)
            rel_err = abs(lam - P_euler) / P_euler
            assert lam > 0.0
            assert mode.shape == (6 * n_nodes,)
            assert rel_err < tol

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    The cantilever model is solved in its original orientation and again after applying
    a rigid-body rotation R to the geometry, element axes, and applied load. The critical
    load factor λ should be identical in both cases.
    The buckling mode from the rotated model should equal the base mode transformed by R:
      [ux, uy, uz] and rotational [θx, θy, θz] DOFs at each node.
    """
    E = 200000000000.0
    nu = 0.3
    L = 20.0
    n_elem = 10
    n_nodes = n_elem + 1
    b = 2.0
    h = 1.0
    A = b * h
    I_y = b * h ** 3 / 12.0
    I_z = h * b ** 3 / 12.0
    J = b * h ** 3 / 3.0
    z = np.linspace(0.0, L, n_nodes)
    node_coords = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    local_z = np.array([1.0, 0.0, 0.0])
    elements_base = []
    for i in range(n_elem):
        elements_base.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    F_base = np.array([0.0, 0.0, -1.0])
    nodal_loads_base = {n_nodes - 1: [F_base[0], F_base[1], F_base[2], 0.0, 0.0, 0.0]}
    lam_base, mode_base = fcn(node_coords, elements_base, boundary_conditions, nodal_loads_base)

    def Rx(a):
        ca, sa = (np.cos(a), np.sin(a))
        return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])

    def Ry(a):
        ca, sa = (np.cos(a), np.sin(a))
        return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])

    def Rz(a):
        ca, sa = (np.cos(a), np.sin(a))
        return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    ang_x = np.deg2rad(15.0)
    ang_y = np.deg2rad(20.0)
    ang_z = np.deg2rad(30.0)
    R = Rz(ang_z) @ Ry(ang_y) @ Rx(ang_x)
    node_coords_rot = (R @ node_coords.T).T
    elements_rot = []
    for i in range(n_elem):
        elements_rot.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': R @ local_z})
    F_rot = R @ F_base
    nodal_loads_rot = {n_nodes - 1: [F_rot[0], F_rot[1], F_rot[2], 0.0, 0.0, 0.0]}
    lam_rot, mode_rot = fcn(node_coords_rot, elements_rot, boundary_conditions, nodal_loads_rot)
    rel_diff_lam = abs(lam_rot - lam_base) / abs(lam_base)
    assert rel_diff_lam < 1e-08
    R6 = np.block([[R, np.zeros((3, 3))], [np.zeros((3, 3)), R]])
    T = np.kron(np.eye(n_nodes), R6)
    mode_pred = T @ mode_base
    denom = np.dot(mode_pred, mode_pred)
    assert denom > 0.0
    alpha = float(np.dot(mode_rot, mode_pred) / denom)
    rel_mode_error = np.linalg.norm(mode_rot - alpha * mode_pred) / np.linalg.norm(mode_pred)
    assert rel_mode_error < 5e-06

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The discretization is refined and the numerical critical load approaches the
    analytical Euler value with decreasing relative error, and the finest mesh
    achieves high accuracy.
    """
    E = 210000000000.0
    nu = 0.3
    L = 20.0
    r = 0.75
    A = math.pi * r ** 2
    I = math.pi * r ** 4 / 4.0
    J = 2.0 * I
    P_euler = math.pi ** 2 * E * I / (4.0 * L * L)
    elems_list = [2, 4, 8, 16, 32]
    errors = []
    for n_elem in elems_list:
        n_nodes = n_elem + 1
        z = np.linspace(0.0, L, n_nodes)
        node_coords = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
        elements = []
        for i in range(n_elem):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J})
        boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
        nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
        lam, _ = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        rel_err = abs(lam - P_euler) / P_euler
        errors.append(rel_err)
    assert errors[0] > errors[-1]
    for i in range(len(errors) - 1):
        assert errors[i + 1] <= errors[i] + 1e-12
    assert errors[-1] < 0.001