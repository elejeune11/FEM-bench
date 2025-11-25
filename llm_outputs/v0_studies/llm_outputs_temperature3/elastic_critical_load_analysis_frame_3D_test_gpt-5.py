def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever value analytical solution.
    Use 10 elements, and a tolerance suitable for discretization error.
    """
    E = 210000000000.0
    nu = 0.3
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    n_elems = 10
    local_z = [1.0, 0.0, 0.0]
    tol_rel = 0.01
    for r in radii:
        A = np.pi * r ** 2
        Iy = Iz = 0.25 * np.pi * r ** 4
        J = 0.5 * np.pi * r ** 4
        I_rho = Iy + Iz
        for L in lengths:
            n_nodes = n_elems + 1
            z = np.linspace(0.0, L, n_nodes)
            node_coords = np.column_stack([np.zeros(n_nodes), np.zeros(n_nodes), z])
            elements = []
            for i in range(n_elems):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': local_z})
            boundary_conditions = {0: [0, 1, 2, 3, 4, 5]}
            nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
            (lam, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            p_cr_numeric = lam
            K_eff = 2.0
            p_cr_analytical = np.pi ** 2 * E * Iy / (K_eff * L) ** 2
            rel_err = abs(p_cr_numeric - p_cr_analytical) / p_cr_analytical
            assert rel_err < tol_rel

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    The cantilever is solved in its original orientation and again after applying a rigid-body rotation
    R to the geometry, element axes, and applied load. The critical load factor λ should be identical.
    The buckling mode from the rotated model should equal the base mode transformed by R via a block-diagonal T
    acting on translational and rotational DOFs at each node (up to arbitrary scale/sign).
    """

    def rot_x(a):
        (ca, sa) = (np.cos(a), np.sin(a))
        return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])

    def rot_y(b):
        (cb, sb) = (np.cos(b), np.sin(b))
        return np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])

    def rot_z(c):
        (cc, sc) = (np.cos(c), np.sin(c))
        return np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    R = rot_z(0.3) @ rot_y(-0.25) @ rot_x(0.4)
    E = 210000000000.0
    nu = 0.3
    b_z = 0.06
    h_y = 0.1
    A = b_z * h_y
    Iy = b_z ** 3 * h_y / 12.0
    Iz = h_y ** 3 * b_z / 12.0
    J = Iy + Iz
    I_rho = Iy + Iz
    local_z_base = np.array([1.0, 0.0, 0.0])
    L = 12.0
    n_elems = 12
    n_nodes = n_elems + 1
    z = np.linspace(0.0, L, n_nodes)
    node_coords_base = np.column_stack([np.zeros(n_nodes), np.zeros(n_nodes), z])
    elements_base = []
    for i in range(n_elems):
        elements_base.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': local_z_base.tolist()})
    bc_base = {0: [0, 1, 2, 3, 4, 5]}
    F_base = np.array([0.0, 0.0, -1.0])
    nodal_loads_base = {n_nodes - 1: [F_base[0], F_base[1], F_base[2], 0.0, 0.0, 0.0]}
    (lam_base, mode_base) = fcn(node_coords_base, elements_base, bc_base, nodal_loads_base)
    node_coords_rot = node_coords_base.dot(R.T)
    local_z_rot = (R @ local_z_base).tolist()
    elements_rot = []
    for i in range(n_elems):
        elements_rot.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': local_z_rot})
    bc_rot = {0: [0, 1, 2, 3, 4, 5]}
    F_rot = (R @ F_base).tolist()
    nodal_loads_rot = {n_nodes - 1: [F_rot[0], F_rot[1], F_rot[2], 0.0, 0.0, 0.0]}
    (lam_rot, mode_rot) = fcn(node_coords_rot, elements_rot, bc_rot, nodal_loads_rot)
    assert np.isclose(lam_rot, lam_base, rtol=1e-08, atol=0.0)
    T = np.zeros((6 * n_nodes, 6 * n_nodes))
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    w = T @ mode_base
    denom = float(w @ w)
    assert denom > 0.0
    alpha = float(mode_rot @ w) / denom
    rel_mismatch = np.linalg.norm(mode_rot - alpha * w) / np.linalg.norm(w)
    assert rel_mismatch < 1e-05

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    Refine the beam discretization and check that the numerical critical load approaches
    the analytical Euler value with decreasing relative error, and that the finest mesh
    achieves very high accuracy.
    """
    E = 210000000000.0
    nu = 0.3
    r = 0.5
    L = 20.0
    A = np.pi * r ** 2
    Iy = Iz = 0.25 * np.pi * r ** 4
    J = 0.5 * np.pi * r ** 4
    I_rho = Iy + Iz
    local_z = [1.0, 0.0, 0.0]
    meshes = [2, 4, 8, 16, 32]
    K_eff = 2.0
    p_cr_analytical = np.pi ** 2 * E * Iy / (K_eff * L) ** 2
    errors = []
    for n_elems in meshes:
        n_nodes = n_elems + 1
        z = np.linspace(0.0, L, n_nodes)
        node_coords = np.column_stack([np.zeros(n_nodes), np.zeros(n_nodes), z])
        elements = []
        for i in range(n_elems):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': local_z})
        boundary_conditions = {0: [0, 1, 2, 3, 4, 5]}
        nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
        (lam, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        p_cr_numeric = lam
        rel_err = abs(p_cr_numeric - p_cr_analytical) / p_cr_analytical
        errors.append(rel_err)
    for i in range(len(errors) - 1):
        assert errors[i + 1] <= errors[i] * 1.05
    assert errors[-1] < 0.001