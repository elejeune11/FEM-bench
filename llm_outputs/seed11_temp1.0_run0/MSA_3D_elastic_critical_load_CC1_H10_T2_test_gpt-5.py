def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """Cantilever (fixed-free) circular column aligned with +z. Sweep radii and lengths and compare λ·P_ref to Euler cantilever analytical value. Uses 10 elements with tolerances suitable for discretization error."""
    E = 210000000000.0
    nu = 0.3
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    nel = 10
    for r in radii:
        A = math.pi * r ** 2
        I = math.pi * r ** 4 / 4.0
        J = 2.0 * I
        for L in lengths:
            n_nodes = nel + 1
            z = np.linspace(0.0, L, n_nodes)
            node_coords = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
            elements = []
            for i in range(nel):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([0.0, 1.0, 0.0])})
            boundary_conditions = {0: (1, 1, 1, 1, 1, 1)}
            nodal_loads = {n_nodes - 1: (0.0, 0.0, -1.0, 0.0, 0.0, 0.0)}
            (lam, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            Pcr_euler = math.pi ** 2 * E * I / (4.0 * L ** 2)
            assert np.isfinite(lam)
            rel_err = abs(lam - Pcr_euler) / Pcr_euler
            assert rel_err < 0.01

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """Orientation invariance test with a rectangular section (Iy ≠ Iz). Solve base and rotated models; λ must match, and rotated mode equals T @ base mode up to a scalar."""
    E = 210000000000.0
    nu = 0.3
    L = 20.0
    nel = 10
    b = 1.0
    h = 0.5
    A = b * h
    I_y = b * h ** 3 / 12.0
    I_z = h * b ** 3 / 12.0
    J = I_y + I_z
    n_nodes = nel + 1
    z = np.linspace(0.0, L, n_nodes)
    node_coords = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    elements = []
    base_local_z = np.array([0.0, 1.0, 0.0])
    for i in range(nel):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': base_local_z})
    boundary_conditions = {0: (1, 1, 1, 1, 1, 1)}
    nodal_loads = {n_nodes - 1: (0.0, 0.0, -1.0, 0.0, 0.0, 0.0)}
    (lam_base, mode_base) = fcn(node_coords, elements, boundary_conditions, nodal_loads)

    def Rz(g):
        (cg, sg) = (math.cos(g), math.sin(g))
        return np.array([[cg, -sg, 0.0], [sg, cg, 0.0], [0.0, 0.0, 1.0]])

    def Ry(bet):
        (cb, sb) = (math.cos(bet), math.sin(bet))
        return np.array([[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]])

    def Rx(a):
        (ca, sa) = (math.cos(a), math.sin(a))
        return np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]])
    R = Rz(0.5) @ Ry(-0.2) @ Rx(0.3)
    node_coords_rot = (R @ node_coords.T).T
    local_z_rot = R @ base_local_z
    elements_rot = []
    for i in range(nel):
        elements_rot.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_rot})
    F_base = np.array([0.0, 0.0, -1.0])
    F_rot = R @ F_base
    nodal_loads_rot = {n_nodes - 1: (F_rot[0], F_rot[1], F_rot[2], 0.0, 0.0, 0.0)}
    (lam_rot, mode_rot) = fcn(node_coords_rot, elements_rot, boundary_conditions, nodal_loads_rot)
    assert abs(lam_rot - lam_base) / lam_base < 1e-07
    T = np.zeros((6 * n_nodes, 6 * n_nodes))
    for k in range(n_nodes):
        i0 = 6 * k
        T[i0:i0 + 3, i0:i0 + 3] = R
        T[i0 + 3:i0 + 6, i0 + 3:i0 + 6] = R
    w = T @ mode_base
    denom = float(w @ w)
    assert denom > 0.0
    alpha = float(w @ mode_rot) / denom
    rel_res = np.linalg.norm(mode_rot - alpha * w) / np.linalg.norm(mode_rot)
    assert rel_res < 0.0001

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    Refinement should decrease relative error toward the analytical Euler value and the finest mesh must be highly accurate."""
    E = 210000000000.0
    nu = 0.3
    r = 1.0
    A = math.pi * r ** 2
    I = math.pi * r ** 4 / 4.0
    J = 2.0 * I
    L = 30.0
    mesh_sizes = [4, 8, 16, 32]
    Pcr_euler = math.pi ** 2 * E * I / (4.0 * L ** 2)
    errors = []
    for nel in mesh_sizes:
        n_nodes = nel + 1
        z = np.linspace(0.0, L, n_nodes)
        node_coords = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
        elements = []
        for i in range(nel):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([0.0, 1.0, 0.0])})
        boundary_conditions = {0: (1, 1, 1, 1, 1, 1)}
        nodal_loads = {n_nodes - 1: (0.0, 0.0, -1.0, 0.0, 0.0, 0.0)}
        (lam, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        rel_err = abs(lam - Pcr_euler) / Pcr_euler
        errors.append(rel_err)
    for i in range(len(errors) - 1):
        assert errors[i + 1] <= errors[i] * 1.05
    assert errors[-1] < 0.005