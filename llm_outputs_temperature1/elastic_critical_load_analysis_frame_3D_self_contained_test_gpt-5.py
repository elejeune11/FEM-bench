def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever value analytical solution.
    Use 10 elements and check relative error within a reasonable tolerance for this discretization.
    """
    E = 210000000000.0
    nu = 0.3
    ne = 10
    P0 = 1.0
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    tol = 0.02
    for r in radii:
        A = np.pi * r ** 2
        I = np.pi * r ** 4 / 4.0
        J = 2.0 * I
        I_rho = J
        for L in lengths:
            z = np.linspace(0.0, L, ne + 1)
            node_coords = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
            local_z = [1.0, 0.0, 0.0]
            elements = []
            for i in range(ne):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': I_rho, 'local_z': local_z})
            boundary_conditions = {0: [True, True, True, True, True, True]}
            nodal_loads = {ne: [0.0, 0.0, -P0, 0.0, 0.0, 0.0]}
            (lam, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            P_cr_num = lam * P0
            P_cr_euler = np.pi ** 2 * E * I / (4.0 * L ** 2)
            rel_err = abs(P_cr_num - P_cr_euler) / P_cr_euler
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
    E = 210000000000.0
    nu = 0.3
    ne = 10
    L = 12.0
    b = 1.0
    h = 2.0
    A = b * h
    Iy = b * h ** 3 / 12.0
    Iz = h * b ** 3 / 12.0
    J = Iy + Iz
    I_rho = Iy + Iz
    P0 = 1.0
    z = np.linspace(0.0, L, ne + 1)
    node_coords_base = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    local_z_base = [1.0, 0.0, 0.0]
    elements_base = []
    for i in range(ne):
        elements_base.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': local_z_base})
    boundary_conditions = {0: [True, True, True, True, True, True]}
    nodal_loads_base = {ne: [0.0, 0.0, -P0, 0.0, 0.0, 0.0]}
    (lam_base, mode_base) = fcn(node_coords_base, elements_base, boundary_conditions, nodal_loads_base)
    alpha = 0.7
    (ca, sa) = (np.cos(alpha), np.sin(alpha))
    R = np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])
    node_coords_rot = (R @ node_coords_base.T).T
    elements_rot = []
    local_z_rot = (R @ np.array(local_z_base)).tolist()
    for i in range(ne):
        elements_rot.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': local_z_rot})
    F_base = np.array([0.0, 0.0, -P0])
    F_rot = (R @ F_base).tolist()
    nodal_loads_rot = {ne: F_rot + [0.0, 0.0, 0.0]}
    (lam_rot, mode_rot) = fcn(node_coords_rot, elements_rot, boundary_conditions, nodal_loads_rot)
    rel_diff_lam = abs(lam_rot - lam_base) / abs(lam_base)
    assert rel_diff_lam < 1e-08
    n_nodes = node_coords_base.shape[0]
    T = np.zeros((6 * n_nodes, 6 * n_nodes))
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    v_base_T = T @ mode_base
    denom = float(np.dot(v_base_T, v_base_T))
    scale = float(np.dot(mode_rot, v_base_T)) / denom if denom > 0 else 1.0
    diff = mode_rot - scale * v_base_T
    rel_mode_err = np.linalg.norm(diff) / (np.linalg.norm(mode_rot) + 1e-16)
    assert rel_mode_err < 1e-05

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves very high accuracy.
    """
    E = 210000000000.0
    nu = 0.3
    L = 10.0
    r = 0.75
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4.0
    J = 2.0 * I
    I_rho = J
    P0 = 1.0
    meshes = [2, 4, 8, 16, 32]
    errors = []
    for ne in meshes:
        z = np.linspace(0.0, L, ne + 1)
        node_coords = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
        local_z = [1.0, 0.0, 0.0]
        elements = []
        for i in range(ne):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': I_rho, 'local_z': local_z})
        boundary_conditions = {0: [True, True, True, True, True, True]}
        nodal_loads = {ne: [0.0, 0.0, -P0, 0.0, 0.0, 0.0]}
        (lam, _) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        P_cr_num = lam * P0
        P_cr_euler = np.pi ** 2 * E * I / (4.0 * L ** 2)
        rel_err = abs(P_cr_num - P_cr_euler) / P_cr_euler
        errors.append(rel_err)
    for i in range(1, len(errors)):
        assert errors[i] <= errors[i - 1] + 1e-12
    assert errors[-1] < 0.001