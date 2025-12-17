def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever value analytical solution.
    Use 10 elements, with tolerances appropriate for the discretization error.
    """
    E = 210000000000.0
    nu = 0.3
    nelem = 10
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    tol = 0.005
    for r in radii:
        A = np.pi * r ** 2
        I = 0.25 * np.pi * r ** 4
        J = 0.5 * np.pi * r ** 4
        for L in lengths:
            n_nodes = nelem + 1
            z = np.linspace(0.0, L, n_nodes)
            node_coords = np.column_stack([np.zeros(n_nodes), np.zeros(n_nodes), z])
            elements = []
            for i in range(nelem):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': [0.0, 1.0, 0.0]})
            boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
            nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
            lam, mode = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            assert np.isfinite(lam) and lam > 0.0
            Pcr_euler = np.pi ** 2 / 4.0 * E * I / L ** 2
            rel_err = abs(lam - Pcr_euler) / Pcr_euler
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

    def rotation_matrix(axis, theta):
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        a = np.cos(theta)
        b = np.sin(theta)
        ax, ay, az = axis
        K = np.array([[0, -az, ay], [az, 0, -ax], [-ay, ax, 0]], dtype=float)
        R = a * np.eye(3) + (1 - a) * np.outer(axis, axis) + b * K
        return R
    E = 210000000000.0
    nu = 0.3
    L = 25.0
    nelem = 12
    n_nodes = nelem + 1
    b = 0.3
    h = 0.6
    A = b * h
    I_y = b * h ** 3 / 12.0
    I_z = h * b ** 3 / 12.0
    J = I_y + I_z
    z = np.linspace(0.0, L, n_nodes)
    node_coords = np.column_stack([np.zeros(n_nodes), np.zeros(n_nodes), z])
    elements = []
    base_local_z = np.array([0.0, 1.0, 0.0])
    for i in range(nelem):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': base_local_z})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
    lam_base, mode_base = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    mode_base = np.asarray(mode_base, dtype=float).reshape(-1)
    R = rotation_matrix([1.0, 2.0, 3.0], 0.67)
    node_coords_rot = (R @ node_coords.T).T
    local_z_rot = (R @ base_local_z.reshape(3, 1)).reshape(3)
    elements_rot = []
    for i in range(nelem):
        elements_rot.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_rot})
    F_base = np.array([0.0, 0.0, -1.0])
    F_rot = R @ F_base
    nodal_loads_rot = {n_nodes - 1: [F_rot[0], F_rot[1], F_rot[2], 0.0, 0.0, 0.0]}
    lam_rot, mode_rot = fcn(node_coords_rot, elements_rot, boundary_conditions, nodal_loads_rot)
    mode_rot = np.asarray(mode_rot, dtype=float).reshape(-1)
    rel_diff_lam = abs(lam_rot - lam_base) / lam_base
    assert rel_diff_lam < 1e-07
    T = np.zeros((6 * n_nodes, 6 * n_nodes), dtype=float)
    for i in range(n_nodes):
        blk = np.zeros((6, 6), dtype=float)
        blk[:3, :3] = R
        blk[3:, 3:] = R
        T[i * 6:(i + 1) * 6, i * 6:(i + 1) * 6] = blk
    mode_rot_pred = T @ mode_base
    denom = float(np.dot(mode_rot_pred, mode_rot_pred))
    if denom == 0.0:
        scale = 1.0
    else:
        scale = float(np.dot(mode_rot, mode_rot_pred)) / denom
    diff = mode_rot - scale * mode_rot_pred
    rel_mode_err = np.linalg.norm(diff) / (np.linalg.norm(mode_rot) + 1e-20)
    assert rel_mode_err < 1e-05

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The beam discretization is refined and the numerical critical load is checked to
    approach the analytical Euler value with decreasing relative error. The finest mesh
    must achieve very high accuracy.
    """
    E = 210000000000.0
    nu = 0.3
    r = 0.5
    A = np.pi * r ** 2
    I = 0.25 * np.pi * r ** 4
    J = 0.5 * np.pi * r ** 4
    L = 15.0

    def solve_with_nelems(nelem):
        n_nodes = nelem + 1
        z = np.linspace(0.0, L, n_nodes)
        node_coords = np.column_stack([np.zeros(n_nodes), np.zeros(n_nodes), z])
        elements = []
        for i in range(nelem):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': [0.0, 1.0, 0.0]})
        boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
        nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
        lam, _ = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        return lam
    nelems_list = [2, 4, 8, 16, 32]
    lambdas = [solve_with_nelems(ne) for ne in nelems_list]
    Pcr_euler = np.pi ** 2 / 4.0 * E * I / L ** 2
    errors = [abs(lam - Pcr_euler) / Pcr_euler for lam in lambdas]
    assert errors[-1] < errors[0]
    assert errors[-1] < 0.0001
    assert errors[-3] > errors[-2] > errors[-1]