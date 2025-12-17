def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever analytical value.
    Use 10 elements; set relative tolerance suitable for discretization error.
    """
    E = 210000000000.0
    nu = 0.3
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    n_elems = 10
    P_ref = 1.0
    rel_tol = 0.001
    for r in radii:
        A = np.pi * r ** 2
        I = np.pi * r ** 4 / 4.0
        Iy = I
        Iz = I
        J = 2.0 * I
        for L in lengths:
            n_nodes = n_elems + 1
            z_coords = np.linspace(0.0, L, n_nodes)
            node_coords = np.column_stack([np.zeros(n_nodes), np.zeros(n_nodes), z_coords])
            elements = []
            local_z = np.array([0.0, 1.0, 0.0])
            for i in range(n_elems):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': local_z})
            boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
            nodal_loads = {n_nodes - 1: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
            lam, mode = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            assert lam > 0.0
            assert mode.shape == (6 * n_nodes,)
            Pcr_analytical = np.pi ** 2 * E * I / (4.0 * L ** 2)
            Pcr_numerical = lam * P_ref
            rel_err = abs(Pcr_numerical - Pcr_analytical) / Pcr_analytical
            assert rel_err < rel_tol

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    The cantilever is analyzed in base orientation and again after a rigid rotation R is applied
    to geometry, element local_z, and applied load. Critical load factor λ should be identical.
    The rotated mode should equal the base mode transformed by T = blockdiag(R, R) per node,
    up to an arbitrary scale/sign.
    """
    E = 200000000000.0
    nu = 0.3
    b = 0.1
    h = 0.2
    A = b * h
    Iy = b * h ** 3 / 12.0
    Iz = h * b ** 3 / 12.0
    J = Iy + Iz
    L = 15.0
    n_elems = 10
    n_nodes = n_elems + 1
    z_coords = np.linspace(0.0, L, n_nodes)
    node_coords = np.column_stack([np.zeros(n_nodes), np.zeros(n_nodes), z_coords])
    elements_base = []
    local_z_base = np.array([0.0, 1.0, 0.0])
    for i in range(n_elems):
        elements_base.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': local_z_base})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    P_ref = 1.0
    nodal_loads_base = {n_nodes - 1: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
    lam_base, mode_base = fcn(node_coords, elements_base, boundary_conditions, nodal_loads_base)
    assert lam_base > 0.0
    assert mode_base.shape == (6 * n_nodes,)
    alpha = 0.3
    beta = -0.25
    gamma = 0.5
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(alpha), -np.sin(alpha)], [0.0, np.sin(alpha), np.cos(alpha)]])
    Ry = np.array([[np.cos(beta), 0.0, np.sin(beta)], [0.0, 1.0, 0.0], [-np.sin(beta), 0.0, np.cos(beta)]])
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0.0], [np.sin(gamma), np.cos(gamma), 0.0], [0.0, 0.0, 1.0]])
    R = Rz @ Ry @ Rx
    node_coords_rot = node_coords @ R.T
    local_z_rot = R @ local_z_base
    elements_rot = []
    for i in range(n_elems):
        elements_rot.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': local_z_rot})
    F_base = np.array([0.0, 0.0, -P_ref])
    F_rot = R @ F_base
    nodal_loads_rot = {n_nodes - 1: [F_rot[0], F_rot[1], F_rot[2], 0.0, 0.0, 0.0]}
    lam_rot, mode_rot = fcn(node_coords_rot, elements_rot, boundary_conditions, nodal_loads_rot)
    assert lam_rot > 0.0
    assert mode_rot.shape == (6 * n_nodes,)
    assert abs(lam_rot - lam_base) / lam_base < 1e-08
    T = np.zeros((6 * n_nodes, 6 * n_nodes))
    for i in range(n_nodes):
        T[i * 6:i * 6 + 3, i * 6:i * 6 + 3] = R
        T[i * 6 + 3:i * 6 + 6, i * 6 + 3:i * 6 + 6] = R
    mode_base_rot = T @ mode_base
    denom = np.dot(mode_base_rot, mode_base_rot)
    if denom == 0.0:
        scale = 1.0
    else:
        scale = np.dot(mode_rot, mode_base_rot) / denom
    diff = mode_rot - scale * mode_base_rot
    rel_mode_err = np.linalg.norm(diff) / max(np.linalg.norm(mode_rot), 1e-16)
    assert rel_mode_err < 1e-05

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    Refine the beam discretization and check that the numerical critical load approaches
    the analytical Euler value with decreasing relative error, and that the finest mesh
    achieves very high accuracy.
    """
    E = 210000000000.0
    nu = 0.3
    r = 0.8
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4.0
    Iy = I
    Iz = I
    J = 2.0 * I
    L = 30.0
    P_ref = 1.0
    n_elems_list = [2, 4, 8, 16]
    errors = []
    Pcr_analytical = np.pi ** 2 * E * I / (4.0 * L ** 2)
    for n_elems in n_elems_list:
        n_nodes = n_elems + 1
        z_coords = np.linspace(0.0, L, n_nodes)
        node_coords = np.column_stack([np.zeros(n_nodes), np.zeros(n_nodes), z_coords])
        elements = []
        local_z = np.array([0.0, 1.0, 0.0])
        for i in range(n_elems):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': local_z})
        boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
        nodal_loads = {n_nodes - 1: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
        lam, mode = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        assert lam > 0.0
        assert mode.shape == (6 * n_nodes,)
        Pcr_numerical = lam * P_ref
        rel_err = abs(Pcr_numerical - Pcr_analytical) / Pcr_analytical
        errors.append(rel_err)
    assert errors[0] > errors[-1]
    decreases = sum((1 for i in range(1, len(errors)) if errors[i] < errors[i - 1]))
    assert decreases >= 2
    assert errors[-1] < 1e-05