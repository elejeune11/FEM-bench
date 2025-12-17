def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever analytical solution.
    Uses 10 elements and checks accuracy within a tight tolerance consistent with discretization.
    """
    E = 210000000000.0
    nu = 0.3
    P_ref = 1.0
    n_el = 10
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    for r in radii:
        A = math.pi * r * r
        I = math.pi * r ** 4 / 4.0
        J = math.pi * r ** 4 / 2.0
        for L in lengths:
            node_coords = np.zeros((n_el + 1, 3))
            node_coords[:, 2] = np.linspace(0.0, L, n_el + 1)
            elements = []
            local_z_vec = np.array([0.0, 1.0, 0.0])
            for i in range(n_el):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': local_z_vec})
            boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
            nodal_loads = {n_el: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
            lam, mode = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            assert np.ndim(lam) == 0 and lam > 0.0
            assert mode.shape == (6 * (n_el + 1),)
            Pcr_num = lam * P_ref
            Pcr_euler = math.pi ** 2 * E * I / (4.0 * L ** 2)
            rel_err = abs(Pcr_num - Pcr_euler) / Pcr_euler
            assert rel_err < 0.005

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
    nu = 0.29
    L = 12.3
    n_el = 10
    n_nodes = n_el + 1
    b = 1.6
    h = 0.9
    A = b * h
    Iy = b * h ** 3 / 12.0
    Iz = h * b ** 3 / 12.0
    J = Iy + Iz
    P_ref = 1.0
    node_coords = np.zeros((n_nodes, 3))
    node_coords[:, 2] = np.linspace(0.0, L, n_nodes)
    elements = []
    base_local_z = np.array([1.0, 0.0, 0.0])
    for i in range(n_el):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': base_local_z})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {n_el: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
    lam_base, mode_base = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert lam_base > 0.0
    assert mode_base.shape == (6 * n_nodes,)

    def Rx(a):
        ca, sa = (math.cos(a), math.sin(a))
        return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=float)

    def Ry(bang):
        cb, sb = (math.cos(bang), math.sin(bang))
        return np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]], dtype=float)

    def Rz(g):
        cg, sg = (math.cos(g), math.sin(g))
        return np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]], dtype=float)
    a, b_ang, g = (0.25, -0.31, 0.42)
    R = Rz(g) @ Ry(b_ang) @ Rx(a)
    node_coords_rot = (R @ node_coords.T).T
    elements_rot = []
    for i in range(n_el):
        elements_rot.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': R @ base_local_z})
    base_force = np.array([0.0, 0.0, -P_ref])
    rotated_force = R @ base_force
    nodal_loads_rot = {n_el: rotated_force.tolist() + [0.0, 0.0, 0.0]}
    lam_rot, mode_rot = fcn(node_coords_rot, elements_rot, boundary_conditions, nodal_loads_rot)
    assert lam_rot > 0.0
    assert mode_rot.shape == (6 * n_nodes,)
    assert math.isclose(lam_base, lam_rot, rel_tol=1e-07, abs_tol=0.0)
    B = np.zeros((6, 6))
    B[:3, :3] = R
    B[3:, 3:] = R
    T = np.kron(np.eye(n_nodes), B)
    mode_base_rotated = T @ mode_base
    denom = float(mode_base_rotated @ mode_base_rotated)
    if denom == 0.0:
        scale = 1.0
    else:
        scale = float(mode_rot @ mode_base_rotated) / denom
    resid = np.linalg.norm(mode_rot - scale * mode_base_rotated) / max(np.linalg.norm(mode_rot), 1.0)
    assert resid < 1e-05

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves very high accuracy.
    """
    E = 210000000000.0
    nu = 0.3
    r = 1.0
    A = math.pi * r * r
    I = math.pi * r ** 4 / 4.0
    J = math.pi * r ** 4 / 2.0
    L = 20.0
    P_ref = 1.0
    Pcr_euler = math.pi ** 2 * E * I / (4.0 * L ** 2)
    mesh_list = [2, 4, 8, 16]
    errors = []
    for n_el in mesh_list:
        n_nodes = n_el + 1
        node_coords = np.zeros((n_nodes, 3))
        node_coords[:, 2] = np.linspace(0.0, L, n_nodes)
        elements = []
        local_z_vec = np.array([0.0, 1.0, 0.0])
        for i in range(n_el):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': local_z_vec})
        boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
        nodal_loads = {n_el: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
        lam, _ = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        Pcr_num = lam * P_ref
        rel_err = abs(Pcr_num - Pcr_euler) / Pcr_euler
        errors.append(rel_err)
    for i in range(len(errors) - 1):
        assert errors[i + 1] < errors[i]
    assert errors[-1] < 0.002