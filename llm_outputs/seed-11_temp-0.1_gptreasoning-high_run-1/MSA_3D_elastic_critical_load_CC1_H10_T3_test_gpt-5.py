def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever value analytical solution.
    Use 10 elements and assert relative error is small for this discretization.
    """
    E = 210000000000.0
    nu = 0.3
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    n_elems = 10
    n_per_axis = n_elems + 1
    P_ref = 1.0
    tol_rel = 0.02
    for r in radii:
        A = np.pi * r ** 2
        I = np.pi * r ** 4 / 4.0
        J = np.pi * r ** 4 / 2.0
        for L in lengths:
            z = np.linspace(0.0, L, n_per_axis)
            node_coords = np.zeros((n_per_axis, 3), dtype=float)
            node_coords[:, 2] = z
            elements = []
            for i in range(n_elems):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([0.0, 1.0, 0.0], dtype=float)})
            boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
            nodal_loads = {n_per_axis - 1: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
            lam, mode = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            assert lam > 0.0
            P_cr_num = lam * P_ref
            P_cr_euler = np.pi ** 2 * E * I / (4.0 * L ** 2)
            rel_err = abs(P_cr_num - P_cr_euler) / P_cr_euler
            assert rel_err < tol_rel

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    The cantilever model is solved in its original orientation and again after applying a rigid-body rotation R to the geometry,
    element local axes, and applied load. The critical load factor λ should be identical.
    The buckling mode from the rotated model should equal the base mode transformed by R:
    mode_rot ≈ T @ mode_base for T built by applying R to both translational and rotational DOFs at each node.
    """

    def Rx(a):
        ca, sa = (np.cos(a), np.sin(a))
        return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=float)

    def Ry(b):
        cb, sb = (np.cos(b), np.sin(b))
        return np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]], dtype=float)

    def Rz(g):
        cg, sg = (np.cos(g), np.sin(g))
        return np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]], dtype=float)
    E = 210000000000.0
    nu = 0.3
    L = 12.0
    n_elems = 16
    n_nodes = n_elems + 1
    P_ref = 1.0
    A = 0.02
    I_y = 1e-06
    I_z = 4e-06
    J = I_y + I_z
    z = np.linspace(0.0, L, n_nodes)
    node_coords = np.zeros((n_nodes, 3), dtype=float)
    node_coords[:, 2] = z
    local_z_base = np.array([0.0, 1.0, 0.0], dtype=float)
    elements_base = []
    for i in range(n_elems):
        elements_base.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_base.copy()})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads_base = {n_nodes - 1: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
    lam_base, mode_base = fcn(node_coords, elements_base, boundary_conditions, nodal_loads_base)
    assert lam_base > 0.0
    a, b, g = (0.4, -0.35, 0.25)
    R = Rz(g) @ Ry(b) @ Rx(a)
    node_coords_rot = (R @ node_coords.T).T
    local_z_rot = (R @ local_z_base.reshape(3, 1)).flatten()
    elements_rot = []
    for i in range(n_elems):
        elements_rot.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_rot.copy()})
    fvec_base = np.array([0.0, 0.0, -P_ref], dtype=float)
    fvec_rot = R @ fvec_base
    nodal_loads_rot = {n_nodes - 1: [fvec_rot[0], fvec_rot[1], fvec_rot[2], 0.0, 0.0, 0.0]}
    lam_rot, mode_rot = fcn(node_coords_rot, elements_rot, boundary_conditions, nodal_loads_rot)
    assert lam_rot > 0.0
    rel_diff = abs(lam_rot - lam_base) / lam_base
    assert rel_diff < 1e-08
    T = np.zeros((6 * n_nodes, 6 * n_nodes), dtype=float)
    for i in range(n_nodes):
        idx = 6 * i
        T[idx:idx + 3, idx:idx + 3] = R
        T[idx + 3:idx + 6, idx + 3:idx + 6] = R
    mode_pred = T @ mode_base
    denom = np.dot(mode_pred, mode_pred)
    if denom == 0.0:
        scale = 0.0
    else:
        scale = np.dot(mode_pred, mode_rot) / denom
    mismatch = np.linalg.norm(mode_rot - scale * mode_pred) / max(np.linalg.norm(mode_rot), 1e-16)
    assert mismatch < 1e-05

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    Refine the beam discretization and check that the numerical critical load approaches
    the analytical Euler value with decreasing relative error, and that the finest mesh
    achieves very high accuracy.
    """
    E = 210000000000.0
    nu = 0.3
    r = 0.75
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4.0
    J = np.pi * r ** 4 / 2.0
    L = 20.0
    P_ref = 1.0
    P_cr_euler = np.pi ** 2 * E * I / (4.0 * L ** 2)
    n_elems_list = [2, 4, 8, 16, 32]
    errors = []
    for n_elems in n_elems_list:
        n_nodes = n_elems + 1
        z = np.linspace(0.0, L, n_nodes)
        node_coords = np.zeros((n_nodes, 3), dtype=float)
        node_coords[:, 2] = z
        elements = []
        for i in range(n_elems):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([0.0, 1.0, 0.0], dtype=float)})
        boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
        nodal_loads = {n_nodes - 1: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
        lam, mode = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        assert lam > 0.0
        P_cr_num = lam * P_ref
        rel_err = abs(P_cr_num - P_cr_euler) / P_cr_euler
        errors.append(rel_err)
    for i in range(1, len(errors)):
        assert errors[i] <= errors[i - 1] + 1e-12
    assert errors[-1] < 0.001