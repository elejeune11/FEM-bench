def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever analytical solution.
    Uses 10 elements; tolerance accounts for discretization error.
    """
    E = 210000000000.0
    nu = 0.3
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    n_el = 10
    P_ref = 1.0
    tol_rel = 0.005
    for r in radii:
        A = np.pi * r ** 2
        I = np.pi * r ** 4 / 4.0
        Iy = I
        Iz = I
        J = 2.0 * I
        I_rho = Iy + Iz
        for L in lengths:
            n_nodes = n_el + 1
            z = np.linspace(0.0, L, n_nodes)
            node_coords = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
            elements = []
            local_z = [1.0, 0.0, 0.0]
            for e in range(n_el):
                elements.append({'node_i': e, 'node_j': e + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': local_z})
            boundary_conditions = {0: [True, True, True, True, True, True]}
            loads = np.zeros(6)
            loads[2] = -P_ref
            nodal_loads = {n_nodes - 1: loads.tolist()}
            (lam, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            assert lam > 0.0
            P_cr_analytical = np.pi ** 2 * E * I / (4.0 * L ** 2)
            rel_err = abs(lam * P_ref - P_cr_analytical) / P_cr_analytical
            assert rel_err < tol_rel

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    The cantilever model is solved in its original orientation and again after applying
    a rigid-body rotation R to the geometry, element axes, and applied load. The critical
    load factor λ should be identical in both cases.
    The buckling mode from the rotated model should equal the base mode transformed by R
    (up to arbitrary scale/sign) using a block-diagonal T built with R for both translational
    and rotational DOFs at each node.
    """
    E = 210000000000.0
    nu = 0.3
    n_el = 12
    L = 15.0
    b = 0.2
    h = 0.1
    A = b * h
    Iy = b * h ** 3 / 12.0
    Iz = h * b ** 3 / 12.0
    J = 1.0 / 3.0 * b * h ** 3
    I_rho = Iy + Iz
    n_nodes = n_el + 1
    z = np.linspace(0.0, L, n_nodes)
    node_coords = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    local_z_base = np.array([1.0, 0.0, 0.0])
    elements_base = []
    for e in range(n_el):
        elements_base.append({'node_i': e, 'node_j': e + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': local_z_base.tolist()})
    boundary_conditions = {0: [True, True, True, True, True, True]}
    P_ref = 1.0
    loads_base = np.zeros(6)
    loads_base[2] = -P_ref
    nodal_loads_base = {n_nodes - 1: loads_base.tolist()}
    (lam_base, mode_base) = fcn(node_coords, elements_base, boundary_conditions, nodal_loads_base)
    assert lam_base > 0.0
    assert mode_base.shape == (6 * n_nodes,)

    def rot_x(a):
        (ca, sa) = (np.cos(a), np.sin(a))
        return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])

    def rot_y(a):
        (ca, sa) = (np.cos(a), np.sin(a))
        return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])

    def rot_z(a):
        (ca, sa) = (np.cos(a), np.sin(a))
        return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    Rx = rot_x(np.deg2rad(-15.0))
    Ry = rot_y(np.deg2rad(10.0))
    Rz = rot_z(np.deg2rad(25.0))
    R = Rz @ Ry @ Rx
    node_coords_rot = (R @ node_coords.T).T
    elements_rot = []
    local_z_rot = (R @ local_z_base).tolist()
    for e in range(n_el):
        el = dict(elements_base[e])
        el['local_z'] = local_z_rot
        elements_rot.append(el)
    F_base = np.array([0.0, 0.0, -P_ref])
    M_base = np.zeros(3)
    F_rot = R @ F_base
    M_rot = R @ M_base
    load_vec_rot = np.concatenate([F_rot, M_rot])
    nodal_loads_rot = {n_nodes - 1: load_vec_rot.tolist()}
    (lam_rot, mode_rot) = fcn(node_coords_rot, elements_rot, boundary_conditions, nodal_loads_rot)
    assert lam_rot > 0.0
    assert mode_rot.shape == (6 * n_nodes,)
    rel_diff_lambda = abs(lam_rot - lam_base) / lam_base
    assert rel_diff_lambda < 1e-08
    T_block = np.block([[R, np.zeros((3, 3))], [np.zeros((3, 3)), R]])
    T = np.kron(np.eye(n_nodes), T_block)
    v = T @ mode_base
    w = mode_rot
    s = float(v @ w) / float(v @ v)
    rel_mode_residual = np.linalg.norm(s * v - w) / np.linalg.norm(w)
    assert rel_mode_residual < 1e-06

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The discretization is refined and the numerical critical load approaches the analytical
    Euler value with decreasing relative error; finest mesh achieves very high accuracy.
    """
    E = 210000000000.0
    nu = 0.3
    r = 0.75
    L = 20.0
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4.0
    Iy = I
    Iz = I
    J = 2.0 * I
    I_rho = Iy + Iz
    P_ref = 1.0
    n_elems_list = [2, 4, 8, 16, 32]
    errors = []
    for n_el in n_elems_list:
        n_nodes = n_el + 1
        z = np.linspace(0.0, L, n_nodes)
        node_coords = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
        elements = []
        local_z = [1.0, 0.0, 0.0]
        for e in range(n_el):
            elements.append({'node_i': e, 'node_j': e + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': local_z})
        boundary_conditions = {0: [True, True, True, True, True, True]}
        loads = np.zeros(6)
        loads[2] = -P_ref
        nodal_loads = {n_nodes - 1: loads.tolist()}
        (lam, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        assert lam > 0.0
        P_cr_analytical = np.pi ** 2 * E * I / (4.0 * L ** 2)
        rel_err = abs(lam * P_ref - P_cr_analytical) / P_cr_analytical
        errors.append(rel_err)
    for k in range(len(errors) - 1):
        assert errors[k + 1] <= errors[k] + 1e-12
    assert errors[-1] < 0.002