def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever value analytical solution.
    Use 10 elements, and assert relative error within a reasonable tolerance for this discretization.
    """

    def build_cantilever(L, r, n_elems, E=210000000000.0, nu=0.3):
        n_nodes = n_elems + 1
        z = np.linspace(0.0, L, n_nodes)
        node_coords = np.column_stack((np.zeros(n_nodes), np.zeros(n_nodes), z))
        A = np.pi * r ** 2
        I = np.pi * r ** 4 / 4.0
        Iy = I
        Iz = I
        J = Iy + Iz
        I_rho = Iy + Iz
        elements = []
        for i in range(n_elems):
            ele = {'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': [1.0, 0.0, 0.0]}
            elements.append(ele)
        boundary_conditions = {0: [True, True, True, True, True, True]}
        nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
        return (node_coords, elements, boundary_conditions, nodal_loads, E, Iy, Iz)
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    n_elems = 10
    tol_rel = 0.01
    for r in radii:
        for L in lengths:
            (node_coords, elements, bcs, loads, E, Iy, Iz) = build_cantilever(L, r, n_elems)
            (lam, mode) = fcn(node_coords, elements, bcs, loads)
            P_cr_num = lam
            I_eff = min(Iy, Iz)
            P_cr_euler = np.pi ** 2 * E * I_eff / (4.0 * L ** 2)
            rel_err = abs(P_cr_num - P_cr_euler) / P_cr_euler
            assert rel_err < tol_rel

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    The cantilever model is solved in its original orientation and again after applying
    a rigid-body rotation R to the geometry, element axes, and applied load. The critical
    load factor λ should be identical in both cases.
    The buckling mode from the rotated model should equal the base mode transformed by R:
      [ux, uy, uz] and rotational [θx, θy, θz] DOFs at each node.
    """

    def rodrigues_rotation(axis, theta):
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
        I = np.eye(3)
        R = I + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
        return R

    def build_rect_cantilever(L, n_elems, E=210000000000.0, nu=0.3, b=1.0, h=2.0, R=None):
        n_nodes = n_elems + 1
        z = np.linspace(0.0, L, n_nodes)
        node_coords = np.column_stack((np.zeros(n_nodes), np.zeros(n_nodes), z))
        if R is not None:
            node_coords = node_coords @ R.T
        A = b * h
        Iy = b * h ** 3 / 12.0
        Iz = h * b ** 3 / 12.0
        J = Iy + Iz
        I_rho = Iy + Iz
        local_z_base = np.array([1.0, 0.0, 0.0])
        local_z = R @ local_z_base if R is not None else local_z_base
        elements = []
        for i in range(n_elems):
            ele = {'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': local_z.tolist()}
            elements.append(ele)
        boundary_conditions = {0: [True, True, True, True, True, True]}
        e_z = np.array([0.0, 0.0, 1.0])
        F = -1.0 * (R @ e_z if R is not None else e_z)
        nodal_loads = {n_nodes - 1: [float(F[0]), float(F[1]), float(F[2]), 0.0, 0.0, 0.0]}
        return (node_coords, elements, boundary_conditions, nodal_loads)
    L = 12.3
    n_elems = 10
    (node_coords, elements, bcs, loads) = build_rect_cantilever(L, n_elems, R=None)
    (lam_base, mode_base) = fcn(node_coords, elements, bcs, loads)
    axis = [1.0, 1.0, 1.0]
    theta = 0.6
    R = rodrigues_rotation(axis, theta)
    (node_coords_r, elements_r, bcs_r, loads_r) = build_rect_cantilever(L, n_elems, R=R)
    (lam_rot, mode_rot) = fcn(node_coords_r, elements_r, bcs_r, loads_r)
    assert np.isclose(lam_base, lam_rot, rtol=1e-05, atol=0.0)
    n_nodes = node_coords.shape[0]
    T = np.zeros((6 * n_nodes, 6 * n_nodes), dtype=float)
    for i in range(n_nodes):
        T[i * 6:i * 6 + 3, i * 6:i * 6 + 3] = R
        T[i * 6 + 3:i * 6 + 6, i * 6 + 3:i * 6 + 6] = R
    psi = T @ mode_base
    denom = np.dot(psi, psi)
    if denom == 0.0:
        scale = 1.0
    else:
        scale = np.dot(mode_rot, psi) / denom
    diff = mode_rot - scale * psi
    rel_err = np.linalg.norm(diff) / max(np.linalg.norm(mode_rot), 1e-16)
    assert rel_err < 0.0002

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves very high accuracy.
    """

    def build_cantilever(L, r, n_elems, E=210000000000.0, nu=0.3):
        n_nodes = n_elems + 1
        z = np.linspace(0.0, L, n_nodes)
        node_coords = np.column_stack((np.zeros(n_nodes), np.zeros(n_nodes), z))
        A = np.pi * r ** 2
        I = np.pi * r ** 4 / 4.0
        Iy = I
        Iz = I
        J = Iy + Iz
        I_rho = Iy + Iz
        elements = []
        for i in range(n_elems):
            ele = {'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': [1.0, 0.0, 0.0]}
            elements.append(ele)
        boundary_conditions = {0: [True, True, True, True, True, True]}
        nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
        return (node_coords, elements, boundary_conditions, nodal_loads, E, Iy, Iz)
    L = 20.0
    r = 0.75
    mesh_sizes = [2, 4, 8, 16, 32]
    errors = []
    for ne in mesh_sizes:
        (node_coords, elements, bcs, loads, E, Iy, Iz) = build_cantilever(L, r, ne)
        (lam, mode) = fcn(node_coords, elements, bcs, loads)
        P_cr_num = lam
        I_eff = min(Iy, Iz)
        P_cr_euler = np.pi ** 2 * E * I_eff / (4.0 * L ** 2)
        rel_err = abs(P_cr_num - P_cr_euler) / P_cr_euler
        errors.append(rel_err)
    for i in range(len(errors) - 1):
        assert errors[i] > errors[i + 1]
    assert errors[-1] < 0.001