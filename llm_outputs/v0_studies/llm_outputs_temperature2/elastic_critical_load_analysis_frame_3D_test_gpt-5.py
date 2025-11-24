def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever circular column aligned with +z: parameter sweep over radii and lengths.
    For each combination of radius r in {0.5, 0.75, 1.0} and length L in {10, 20, 40},
    discretize the cantilever with 10 Euler-Bernoulli beam elements, apply a unit
    compressive tip load along the column axis, and perform elastic critical load
    analysis. Compare the predicted critical load factor λ (for P_ref = 1) to the
    analytical Euler cantilever load P_cr = π^2 E I / (4 L^2). Use a relative
    tolerance that reflects expected discretization error for 10 elements.
    """
    E = 210000000000.0
    nu = 0.3
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    n_elems = 10
    local_z_vec = [1.0, 0.0, 0.0]

    def euler_cantilever_load(E_, I_, L_):
        return math.pi ** 2 * E_ * I_ / (4.0 * L_ ** 2)
    for r in radii:
        A = math.pi * r * r
        I = math.pi * r ** 4 / 4.0
        J = math.pi * r ** 4 / 2.0
        I_rho = I + I
        for L in lengths:
            n_nodes = n_elems + 1
            z = np.linspace(0.0, L, n_nodes)
            node_coords = np.zeros((n_nodes, 3), dtype=float)
            node_coords[:, 2] = z
            elements = []
            for i in range(n_elems):
                ele = {'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': I_rho, 'local_z': local_z_vec}
                elements.append(ele)
            boundary_conditions = {0: [True, True, True, True, True, True]}
            nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
            (lam, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            assert lam > 0.0
            assert mode.shape == (6 * n_nodes,)
            P_cr_analytical = euler_cantilever_load(E, I, L)
            rel_err = abs(lam - P_cr_analytical) / P_cr_analytical
            assert rel_err < 0.0025

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance for cantilever with rectangular section (Iy ≠ Iz).
    Solve the buckling problem in a base configuration and again after a rigid-body
    rotation of the entire model (nodes, element local axes, loads). The critical
    load factor must be invariant. The rotated mode should match the base mode
    transformed by the block-diagonal T = diag(R, R, ..., R) on translational and
    rotational DOFs at each node, up to an arbitrary scalar factor (including sign).
    """
    E = 70000000000.0
    nu = 0.29
    L = 12.0
    n_elems = 12
    n_nodes = n_elems + 1
    b = 2.0
    h = 1.0
    A = b * h
    Iy = b * h ** 3 / 12.0
    Iz = h * b ** 3 / 12.0
    J = Iy + Iz
    I_rho = Iy + Iz
    z = np.linspace(0.0, L, n_nodes)
    node_coords = np.zeros((n_nodes, 3), dtype=float)
    node_coords[:, 2] = z
    local_z_vec_base = np.array([1.0, 0.0, 0.0], dtype=float)
    elements_base = []
    for i in range(n_elems):
        ele = {'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': local_z_vec_base.tolist()}
        elements_base.append(ele)
    boundary_conditions = {0: [True, True, True, True, True, True]}
    P_ref_base = np.zeros(6, dtype=float)
    P_ref_base[2] = -1.0
    nodal_loads_base = {n_nodes - 1: P_ref_base.tolist()}
    (lam_base, mode_base) = fcn(node_coords, elements_base, boundary_conditions, nodal_loads_base)
    assert lam_base > 0.0
    assert mode_base.shape == (6 * n_nodes,)
    axis = np.array([1.0, 1.0, 0.5], dtype=float)
    axis = axis / np.linalg.norm(axis)
    angle = 0.7
    K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]], dtype=float)
    I3 = np.eye(3)
    R = I3 * math.cos(angle) + (1 - math.cos(angle)) * np.outer(axis, axis) + math.sin(angle) * K
    node_coords_rot = (R @ node_coords.T).T
    local_z_vec_rot = R @ local_z_vec_base
    elements_rot = []
    for i in range(n_elems):
        ele = {'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': local_z_vec_rot.tolist()}
        elements_rot.append(ele)
    axis_rot = R @ np.array([0.0, 0.0, 1.0], dtype=float)
    P_ref_rot = np.zeros(6, dtype=float)
    P_ref_rot[:3] = -axis_rot
    nodal_loads_rot = {n_nodes - 1: P_ref_rot.tolist()}
    (lam_rot, mode_rot) = fcn(node_coords_rot, elements_rot, boundary_conditions, nodal_loads_rot)
    assert lam_rot > 0.0
    assert mode_rot.shape == (6 * n_nodes,)
    rel_diff_lam = abs(lam_rot - lam_base) / lam_base
    assert rel_diff_lam < 5e-07
    T = np.zeros((6 * n_nodes, 6 * n_nodes), dtype=float)
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    mode_base_transformed = T @ mode_base
    denom = float(np.dot(mode_base_transformed, mode_base_transformed))
    assert denom > 0.0
    alpha = float(np.dot(mode_base_transformed, mode_rot)) / denom
    resid = np.linalg.norm(mode_rot - alpha * mode_base_transformed)
    norm_rot = np.linalg.norm(mode_rot)
    if norm_rot == 0.0:
        assert False
    rel_resid = resid / norm_rot
    assert rel_resid < 5e-06

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Mesh convergence for Euler buckling of a fixed-free circular cantilever.
    Discretize the cantilever with progressively refined meshes and verify that
    the numerical critical load approaches the analytical Euler value. Check that
    errors decrease with refinement (allowing a small tolerance) and that the
    finest mesh achieves high accuracy.
    """
    E = 200000000000.0
    nu = 0.3
    L = 20.0
    r = 0.5
    A = math.pi * r * r
    I = math.pi * r ** 4 / 4.0
    J = math.pi * r ** 4 / 2.0
    I_rho = I + I
    local_z_vec = [1.0, 0.0, 0.0]

    def euler_cantilever_load(E_, I_, L_):
        return math.pi ** 2 * E_ * I_ / (4.0 * L_ ** 2)
    P_cr_analytical = euler_cantilever_load(E, I, L)
    n_elems_list = [2, 4, 8, 16, 32]
    errors = []
    for n_elems in n_elems_list:
        n_nodes = n_elems + 1
        z = np.linspace(0.0, L, n_nodes)
        node_coords = np.zeros((n_nodes, 3), dtype=float)
        node_coords[:, 2] = z
        elements = []
        for i in range(n_elems):
            ele = {'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': I_rho, 'local_z': local_z_vec}
            elements.append(ele)
        boundary_conditions = {0: [True, True, True, True, True, True]}
        nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
        (lam, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        assert lam > 0.0
        assert mode.shape == (6 * n_nodes,)
        rel_err = abs(lam - P_cr_analytical) / P_cr_analytical
        errors.append(rel_err)
    for k in range(1, len(errors)):
        assert errors[k] <= errors[k - 1] * 1.05
    assert errors[-1] < 0.001