def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever value analytical solution.
    Use 10 elements, set tolerances to be appropriate for the anticipated discretization error at 10-5.
    """
    E = 210000000000.0
    nu = 0.3
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    ne = 10
    tol = 0.03
    for r in radii:
        A = np.pi * r ** 2
        I = np.pi * r ** 4 / 4.0
        Iy = I
        Iz = I
        J = np.pi * r ** 4 / 2.0
        I_rho = Iy + Iz
        for L in lengths:
            n_nodes = ne + 1
            z = np.linspace(0.0, L, n_nodes)
            node_coords = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
            elements = []
            for i in range(ne):
                elements.append(dict(node_i=i, node_j=i + 1, E=E, nu=nu, A=A, I_y=Iy, I_z=Iz, J=J, I_rho=I_rho, local_z=[0.0, 1.0, 0.0]))
            boundary_conditions = {0: [True, True, True, True, True, True]}
            nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
            (lam, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            pcr_euler = np.pi ** 2 * E * I / (2.0 * L) ** 2
            rel_err = abs(lam - pcr_euler) / pcr_euler
            assert rel_err <= tol
            assert mode.shape == (6 * n_nodes,)
            base_dofs = mode[:6]
            mode_scale = np.max(np.abs(mode)) if np.max(np.abs(mode)) != 0.0 else 1.0
            assert np.max(np.abs(base_dofs)) <= 1e-09 * mode_scale + 1e-12

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    The cantilever model is solved in its original orientation and again after applying
    a rigid-body rotation R to the geometry, element axes, and applied load. The critical
    load factor λ should be identical in both cases.
    The buckling mode from the rotated model should equal the base mode transformed by R:
      [ux, uy, uz] and rotational [θx, θy, θz] DOFs at each node.
    """
    E = 70000000000.0
    nu = 0.29
    L = 12.5
    ne = 12
    A = 1.0
    Iy = 0.8
    Iz = 0.2
    J = Iy + Iz
    I_rho = Iy + Iz
    n_nodes = ne + 1
    z = np.linspace(0.0, L, n_nodes)
    node_coords_base = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    elements_base = []
    for i in range(ne):
        elements_base.append(dict(node_i=i, node_j=i + 1, E=E, nu=nu, A=A, I_y=Iy, I_z=Iz, J=J, I_rho=I_rho, local_z=[0.0, 1.0, 0.0]))
    boundary_conditions = {0: [True, True, True, True, True, True]}
    load_base_vec = np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0])
    nodal_loads_base = {n_nodes - 1: load_base_vec.tolist()}
    (lam_base, mode_base) = fcn(node_coords_base, elements_base, boundary_conditions, nodal_loads_base)

    def Rx(a):
        (ca, sa) = (np.cos(a), np.sin(a))
        return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])

    def Ry(a):
        (ca, sa) = (np.cos(a), np.sin(a))
        return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])

    def Rz(a):
        (ca, sa) = (np.cos(a), np.sin(a))
        return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    R = Rz(0.2) @ Ry(-0.3) @ Rx(0.5)
    node_coords_rot = (R @ node_coords_base.T).T
    elements_rot = []
    base_local_z = np.array([0.0, 1.0, 0.0])
    local_z_rot = (R @ base_local_z).tolist()
    for i in range(ne):
        elements_rot.append(dict(node_i=i, node_j=i + 1, E=E, nu=nu, A=A, I_y=Iy, I_z=Iz, J=J, I_rho=I_rho, local_z=local_z_rot))
    load_rot_vec = np.zeros(6)
    load_rot_vec[:3] = R @ load_base_vec[:3]
    nodal_loads_rot = {n_nodes - 1: load_rot_vec.tolist()}
    (lam_rot, mode_rot) = fcn(node_coords_rot, elements_rot, boundary_conditions, nodal_loads_rot)
    rel_diff_lam = abs(lam_rot - lam_base) / max(abs(lam_base), 1.0)
    assert rel_diff_lam <= 1e-06
    T = np.zeros((6 * n_nodes, 6 * n_nodes))
    for i in range(n_nodes):
        T_block = np.zeros((6, 6))
        T_block[:3, :3] = R
        T_block[3:, 3:] = R
        T[6 * i:6 * (i + 1), 6 * i:6 * (i + 1)] = T_block
    mode_base_rotated = T @ mode_base
    denom = np.dot(mode_base_rotated, mode_base_rotated)
    alpha = np.dot(mode_rot, mode_base_rotated) / denom if denom != 0.0 else 0.0
    diff = mode_rot - alpha * mode_base_rotated
    rel_mode_err = np.linalg.norm(diff) / (np.linalg.norm(mode_rot) + 1e-16)
    assert rel_mode_err <= 1e-05

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves very high accuracy.
    """
    E = 200000000000.0
    nu = 0.3
    L = 10.0
    r = 1.0
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4.0
    Iy = I
    Iz = I
    J = np.pi * r ** 4 / 2.0
    I_rho = Iy + Iz
    pcr_euler = np.pi ** 2 * E * I / (2.0 * L) ** 2
    meshes = [2, 4, 8, 16, 32]
    errors = []
    for ne in meshes:
        n_nodes = ne + 1
        z = np.linspace(0.0, L, n_nodes)
        node_coords = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
        elements = []
        for i in range(ne):
            elements.append(dict(node_i=i, node_j=i + 1, E=E, nu=nu, A=A, I_y=Iy, I_z=Iz, J=J, I_rho=I_rho, local_z=[0.0, 1.0, 0.0]))
        boundary_conditions = {0: [True, True, True, True, True, True]}
        nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
        (lam, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        rel_err = abs(lam - pcr_euler) / pcr_euler
        errors.append(rel_err)
    for i in range(1, len(errors)):
        assert errors[i] <= errors[i - 1] + 1e-12
    assert errors[-1] <= 0.001