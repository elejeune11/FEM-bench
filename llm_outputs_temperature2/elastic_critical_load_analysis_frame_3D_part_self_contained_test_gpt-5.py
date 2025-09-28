def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever analytical solution.
    Use 10 elements; tolerances selected for anticipated discretization error near 1e-5.
    """
    E = 210000000000.0
    nu = 0.3
    r_values = [0.5, 0.75, 1.0]
    L_values = [10.0, 20.0, 40.0]
    n_el = 10
    local_z = [0.0, 1.0, 0.0]
    for r in r_values:
        A = np.pi * r ** 2
        I = np.pi * r ** 4 / 4.0
        J = np.pi * r ** 4 / 2.0
        for L in L_values:
            n_nodes = n_el + 1
            z = np.linspace(0.0, L, n_nodes)
            node_coords = np.zeros((n_nodes, 3))
            node_coords[:, 2] = z
            elements = []
            for i in range(n_el):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': J, 'local_z': local_z})
            boundary_conditions = {0: [True, True, True, True, True, True]}
            nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
            (lam, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            assert mode.shape == (6 * n_nodes,)
            K_eff = 2.0
            Pcr_euler = np.pi ** 2 * E * I / (K_eff * L) ** 2
            rel_err = abs(lam - Pcr_euler) / Pcr_euler
            assert rel_err < 2e-05

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
    nu = 0.3
    b = 0.4
    h = 1.0
    A = b * h
    Iy = b * h ** 3 / 12.0
    Iz = h * b ** 3 / 12.0
    J = Iy + Iz
    I_rho = Iy + Iz
    L = 10.0
    n_el = 10
    n_nodes = n_el + 1
    z = np.linspace(0.0, L, n_nodes)
    node_coords_base = np.zeros((n_nodes, 3))
    node_coords_base[:, 2] = z
    local_z_base = np.array([0.0, 1.0, 0.0])
    elements_base = []
    for i in range(n_el):
        elements_base.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': local_z_base.tolist()})
    boundary_conditions_base = {0: [True, True, True, True, True, True]}
    nodal_loads_base = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
    (lam_base, mode_base) = fcn(node_coords_base, elements_base, boundary_conditions_base, nodal_loads_base)
    assert mode_base.shape == (6 * n_nodes,)

    def rot_x(theta):
        (c, s) = (np.cos(theta), np.sin(theta))
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def rot_y(theta):
        (c, s) = (np.cos(theta), np.sin(theta))
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    R = rot_y(-0.6) @ rot_x(0.35)
    node_coords_rot = (R @ node_coords_base.T).T
    local_z_rot = (R @ local_z_base).tolist()
    elements_rot = []
    for i in range(n_el):
        elements_rot.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': local_z_rot})
    boundary_conditions_rot = {0: [True, True, True, True, True, True]}
    base_force = np.array([0.0, 0.0, -1.0])
    force_rot = R @ base_force
    nodal_loads_rot = {n_nodes - 1: [force_rot[0], force_rot[1], force_rot[2], 0.0, 0.0, 0.0]}
    (lam_rot, mode_rot) = fcn(node_coords_rot, elements_rot, boundary_conditions_rot, nodal_loads_rot)
    assert mode_rot.shape == (6 * n_nodes,)
    assert np.isclose(lam_rot, lam_base, rtol=1e-06, atol=0.0)
    T = np.zeros((6 * n_nodes, 6 * n_nodes))
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    w = T @ mode_base
    denom = np.dot(w, w)
    assert denom > 0.0
    alpha = float(np.dot(mode_rot, w) / denom)
    residual = np.linalg.norm(mode_rot - alpha * w) / max(np.linalg.norm(mode_rot), 1e-30)
    assert residual < 1e-06

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves very high accuracy.
    """
    E = 210000000000.0
    nu = 0.3
    r = 0.8
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4.0
    J = np.pi * r ** 4 / 2.0
    I_rho = J
    L = 25.0
    meshes = [2, 4, 8, 16, 32]
    Pcr_euler = np.pi ** 2 * E * I / (2.0 * L) ** 2
    errors = []
    for n_el in meshes:
        n_nodes = n_el + 1
        z = np.linspace(0.0, L, n_nodes)
        node_coords = np.zeros((n_nodes, 3))
        node_coords[:, 2] = z
        elements = []
        for i in range(n_el):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': I_rho, 'local_z': [0.0, 1.0, 0.0]})
        boundary_conditions = {0: [True, True, True, True, True, True]}
        nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
        (lam, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        rel_err = abs(lam - Pcr_euler) / Pcr_euler
        errors.append(rel_err)
    for i in range(1, len(errors)):
        assert errors[i] <= errors[i - 1] + 1e-12
    assert errors[-1] < 1e-08