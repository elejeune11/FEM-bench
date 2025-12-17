def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever analytical solution.
    Use 10 elements; tolerances account for discretization error.
    """
    E = 210000000000.0
    nu = 0.3
    P_ref = 1.0
    r_values = [0.5, 0.75, 1.0]
    L_values = [10.0, 20.0, 40.0]
    n_elems = 10
    rtol = 0.02
    for r in r_values:
        A = np.pi * r ** 2
        I = np.pi * r ** 4 / 4.0
        J = 2.0 * I
        for L in L_values:
            n_nodes = n_elems + 1
            z = np.linspace(0.0, L, n_nodes)
            node_coords = np.column_stack((np.zeros_like(z), np.zeros_like(z), z))
            elements = []
            for i in range(n_elems):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': [0.0, 1.0, 0.0]})
            boundary_conditions = {0: (1, 1, 1, 1, 1, 1)}
            nodal_loads = {n_nodes - 1: (0.0, 0.0, -P_ref, 0.0, 0.0, 0.0)}
            lam, _ = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            Pcr_num = lam * P_ref
            Pcr_euler = np.pi ** 2 * E * I / (2.0 * L) ** 2
            rel_err = abs(Pcr_num - Pcr_euler) / Pcr_euler
            assert rel_err <= rtol

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    The cantilever model is solved in its original orientation and again after applying a rigid-body rotation R
    to the geometry, element axes, and applied load. The critical load factor λ should be identical in both cases.
    The buckling mode from the rotated model should equal the base mode transformed by R via block-diagonal T
    that applies R to both translational and rotational DOFs at each node, up to arbitrary scale/sign.
    """
    E = 210000000000.0
    nu = 0.3
    L = 12.0
    n_elems = 10
    n_nodes = n_elems + 1
    P_ref = 1.0
    b = 0.05
    h = 0.15
    A = b * h
    I_y = b * h ** 3 / 12.0
    I_z = h * b ** 3 / 12.0
    J = 1.0 / 3.0 * b * h ** 3 if h >= b else 1.0 / 3.0 * h * b ** 3
    z = np.linspace(0.0, L, n_nodes)
    node_coords = np.column_stack((np.zeros_like(z), np.zeros_like(z), z))
    elements = []
    for i in range(n_elems):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': [0.0, 1.0, 0.0]})
    boundary_conditions = {0: (1, 1, 1, 1, 1, 1)}
    nodal_loads = {n_nodes - 1: (0.0, 0.0, -P_ref, 0.0, 0.0, 0.0)}
    lam_base, mode_base = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    ax = np.deg2rad(25.0)
    ay = np.deg2rad(15.0)
    az = np.deg2rad(20.0)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(ax), -np.sin(ax)], [0.0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0.0, np.sin(ay)], [0.0, 1.0, 0.0], [-np.sin(ay), 0.0, np.cos(ay)]])
    Rz = np.array([[np.cos(az), -np.sin(az), 0.0], [np.sin(az), np.cos(az), 0.0], [0.0, 0.0, 1.0]])
    R = Rz @ Ry @ Rx
    node_coords_rot = (R @ node_coords.T).T
    elements_rot = []
    for e in elements:
        local_z_rot = (R @ np.array(e['local_z']).reshape(3, 1)).ravel()
        elements_rot.append({'node_i': e['node_i'], 'node_j': e['node_j'], 'E': e['E'], 'nu': e['nu'], 'A': e['A'], 'I_y': e['I_y'], 'I_z': e['I_z'], 'J': e['J'], 'local_z': local_z_rot.tolist()})
    F_base = np.array([0.0, 0.0, -P_ref])
    F_rot = R @ F_base
    nodal_loads_rot = {n_nodes - 1: (F_rot[0], F_rot[1], F_rot[2], 0.0, 0.0, 0.0)}
    lam_rot, mode_rot = fcn(node_coords_rot, elements_rot, boundary_conditions, nodal_loads_rot)
    assert np.isclose(lam_rot, lam_base, rtol=1e-08, atol=0.0)
    T_blocks = []
    for _ in range(n_nodes):
        T_block = np.zeros((6, 6))
        T_block[:3, :3] = R
        T_block[3:, 3:] = R
        T_blocks.append(T_block)
    T = np.block([[T_blocks[i] if i == j else np.zeros((6, 6)) for j in range(n_nodes)] for i in range(n_nodes)])
    v = T @ mode_base
    w = mode_rot
    alpha = float(np.dot(v, w) / (np.dot(v, v) + 1e-30))
    diff = v * alpha - w
    rel = np.linalg.norm(diff) / (np.linalg.norm(w) + 1e-30)
    assert rel < 1e-05

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    Refine the beam discretization and check that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that
    the finest mesh achieves very high accuracy.
    """
    E = 210000000000.0
    nu = 0.3
    L = 20.0
    r = 0.5
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4.0
    J = 2.0 * I
    P_ref = 1.0
    ne_list = [5, 10, 20, 40]
    errors = []
    for n_elems in ne_list:
        n_nodes = n_elems + 1
        z = np.linspace(0.0, L, n_nodes)
        node_coords = np.column_stack((np.zeros_like(z), np.zeros_like(z), z))
        elements = []
        for i in range(n_elems):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': [0.0, 1.0, 0.0]})
        boundary_conditions = {0: (1, 1, 1, 1, 1, 1)}
        nodal_loads = {n_nodes - 1: (0.0, 0.0, -P_ref, 0.0, 0.0, 0.0)}
        lam, _ = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        Pcr_num = lam * P_ref
        Pcr_euler = np.pi ** 2 * E * I / (2.0 * L) ** 2
        rel_err = abs(Pcr_num - Pcr_euler) / Pcr_euler
        errors.append(rel_err)
    assert errors[-1] < errors[0]
    decreases = sum((1 for i in range(len(errors) - 1) if errors[i + 1] < errors[i]))
    assert decreases >= len(errors) - 2
    assert errors[-1] < 0.005