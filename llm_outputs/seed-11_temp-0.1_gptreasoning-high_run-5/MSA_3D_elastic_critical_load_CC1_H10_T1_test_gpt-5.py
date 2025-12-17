def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever value analytical solution.
    Use 10 elements, set tolerances to be appropriate for the anticipated discretization error at 10-5.
    """
    E = 210000000000.0
    nu = 0.3
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    n_elems = 10
    tol = 0.0005
    for L in lengths:
        for r in radii:
            n_nodes = n_elems + 1
            node_coords = np.zeros((n_nodes, 3), dtype=float)
            node_coords[:, 2] = np.linspace(0.0, L, n_nodes)
            A = np.pi * r ** 2
            I = 0.25 * np.pi * r ** 4
            J = 2.0 * I
            elements = []
            for i in range(n_nodes - 1):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([0.0, 1.0, 0.0], dtype=float)})
            boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
            nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
            lam, mode = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            P_euler = np.pi ** 2 * E * I / (4.0 * L ** 2)
            rel_err = abs(lam - P_euler) / P_euler
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

    def rot_from_axis_angle(axis, theta):
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        ax = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]], dtype=float)
        I3 = np.eye(3)
        R = I3 * np.cos(theta) + (1.0 - np.cos(theta)) * np.outer(axis, axis) + np.sin(theta) * ax
        return R
    E = 210000000000.0
    nu = 0.3
    L = 15.0
    b = 0.3
    h = 0.5
    A = b * h
    I_y = b * h ** 3 / 12.0
    I_z = h * b ** 3 / 12.0
    J = I_y + I_z
    n_elems = 12
    n_nodes = n_elems + 1
    node_coords = np.zeros((n_nodes, 3), dtype=float)
    node_coords[:, 2] = np.linspace(0.0, L, n_nodes)
    local_z_dir = np.array([0.0, 1.0, 0.0], dtype=float)
    elements = []
    for i in range(n_nodes - 1):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_dir.copy()})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    load_base = np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0], dtype=float)
    nodal_loads = {n_nodes - 1: load_base.tolist()}
    lam_base, mode_base = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    axis = np.array([0.7, 0.2, 1.0], dtype=float)
    theta = 0.73
    R = rot_from_axis_angle(axis, theta)
    node_coords_rot = (R @ node_coords.T).T
    local_z_rot = R @ local_z_dir
    elements_rot = []
    for i in range(n_nodes - 1):
        elements_rot.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_rot.copy()})
    load_rot_vec = np.zeros(6, dtype=float)
    load_rot_vec[:3] = R @ np.array([0.0, 0.0, -1.0], dtype=float)
    nodal_loads_rot = {n_nodes - 1: load_rot_vec.tolist()}
    lam_rot, mode_rot = fcn(node_coords_rot, elements_rot, boundary_conditions, nodal_loads_rot)
    assert abs(lam_base - lam_rot) / lam_base < 1e-08
    T = np.zeros((6 * n_nodes, 6 * n_nodes), dtype=float)
    for i in range(n_nodes):
        idx = slice(6 * i, 6 * i + 3)
        idr = slice(6 * i + 3, 6 * i + 6)
        T[idx, idx] = R
        T[idr, idr] = R
    mode_pred = T @ mode_base
    denom = float(mode_pred @ mode_pred)
    alpha = float(mode_pred @ mode_rot / denom) if denom > 0 else 1.0
    resid = mode_rot - alpha * mode_pred
    rel_resid = np.linalg.norm(resid) / max(np.linalg.norm(mode_rot), 1e-16)
    assert rel_resid < 1e-06

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves very high accuracy.
    """
    E = 210000000000.0
    nu = 0.3
    L = 20.0
    r = 0.75
    A = np.pi * r ** 2
    I = 0.25 * np.pi * r ** 4
    J = 2.0 * I
    P_euler = np.pi ** 2 * E * I / (4.0 * L ** 2)
    elems_list = [2, 4, 8, 16, 32]
    errors = []
    for n_elems in elems_list:
        n_nodes = n_elems + 1
        node_coords = np.zeros((n_nodes, 3), dtype=float)
        node_coords[:, 2] = np.linspace(0.0, L, n_nodes)
        elements = []
        for i in range(n_nodes - 1):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([0.0, 1.0, 0.0], dtype=float)})
        boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
        nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
        lam, mode = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        err = abs(lam - P_euler) / P_euler
        errors.append(err)
    assert errors[0] > errors[-1]
    assert errors[-1] < 0.0001