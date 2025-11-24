def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever analytical solution.
    Use 10 elements and a relative tolerance suitable for discretization error.
    """
    E = 1.0
    nu = 0.3
    P_ref = 1.0
    ne = 10
    r_vals = [0.5, 0.75, 1.0]
    L_vals = [10.0, 20.0, 40.0]
    rtol = 0.02
    for r in r_vals:
        A = np.pi * r ** 2
        Iy = np.pi / 4.0 * r ** 4
        Iz = Iy
        J = np.pi / 2.0 * r ** 4
        I_rho = Iy + Iz
        for L in L_vals:
            z = np.linspace(0.0, L, ne + 1)
            node_coords = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
            elements = []
            for i in range(ne):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': [1.0, 0.0, 0.0]})
            boundary_conditions = {0: [True, True, True, True, True, True]}
            load_vec = [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]
            nodal_loads = {ne: load_vec}
            (lam, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            P_cr_num = lam * P_ref
            P_cr_analytical = np.pi ** 2 * E * Iy / (4.0 * L ** 2)
            rel_err = abs(P_cr_num - P_cr_analytical) / P_cr_analytical
            assert rel_err <= rtol

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    The cantilever model is solved in its original orientation and again after applying
    a rigid-body rotation R to the geometry, element axes, and applied load. The critical
    load factor λ should be identical in both cases.
    The buckling mode from the rotated model should equal the base mode transformed by R:
      [ux, uy, uz] and rotational [θx, θy, θz] DOFs at each node.
    """
    E = 3.0
    nu = 0.3
    Iy = 2.0
    Iz = 1.0
    A = 10.0
    J = Iy + Iz
    I_rho = Iy + Iz
    L = 15.0
    ne = 12
    P_ref = 1.0
    z = np.linspace(0.0, L, ne + 1)
    node_coords = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    elements = []
    for i in range(ne):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': [1.0, 0.0, 0.0]})
    boundary_conditions = {0: [True, True, True, True, True, True]}
    nodal_loads = {ne: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
    (lam_base, mode_base) = fcn(node_coords, elements, boundary_conditions, nodal_loads)

    def Rx(a):
        (c, s) = (np.cos(a), np.sin(a))
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def Ry(a):
        (c, s) = (np.cos(a), np.sin(a))
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def Rz(a):
        (c, s) = (np.cos(a), np.sin(a))
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    R = Rz(np.deg2rad(30.0)) @ Ry(np.deg2rad(20.0)) @ Rx(np.deg2rad(10.0))
    node_coords_rot = node_coords @ R.T
    elements_rot = []
    local_z_base = np.array([1.0, 0.0, 0.0])
    local_z_rot = (R @ local_z_base).tolist()
    for i in range(ne):
        elements_rot.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': local_z_rot})
    F_base = np.array([0.0, 0.0, -P_ref])
    F_rot = (R @ F_base).tolist()
    nodal_loads_rot = {ne: F_rot + [0.0, 0.0, 0.0]}
    (lam_rot, mode_rot) = fcn(node_coords_rot, elements_rot, boundary_conditions, nodal_loads_rot)
    assert abs(lam_rot - lam_base) <= 1e-06 * max(1.0, abs(lam_base))
    n_nodes = node_coords.shape[0]
    T = np.zeros((6 * n_nodes, 6 * n_nodes))
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    mode_base_to_rot = T @ mode_base
    denom = float(np.dot(mode_base_to_rot, mode_base_to_rot))
    if denom == 0.0:
        scale = 1.0
    else:
        scale = float(np.dot(mode_rot, mode_base_to_rot)) / denom
    diff = mode_rot - scale * mode_base_to_rot
    rel = np.linalg.norm(diff) / max(1e-12, np.linalg.norm(mode_rot))
    assert rel <= 0.001

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves very high accuracy.
    """
    E = 1.0
    nu = 0.3
    r = 1.0
    A = np.pi * r ** 2
    Iy = np.pi / 4.0 * r ** 4
    Iz = Iy
    J = np.pi / 2.0 * r ** 4
    I_rho = Iy + Iz
    L = 20.0
    P_ref = 1.0
    P_cr_analytical = np.pi ** 2 * E * Iy / (4.0 * L ** 2)
    mesh_sizes = [2, 4, 8, 16, 32]
    errors = []
    for ne in mesh_sizes:
        z = np.linspace(0.0, L, ne + 1)
        node_coords = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
        elements = []
        for i in range(ne):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': [1.0, 0.0, 0.0]})
        boundary_conditions = {0: [True, True, True, True, True, True]}
        nodal_loads = {ne: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
        (lam, _) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        P_cr_num = lam * P_ref
        rel_err = abs(P_cr_num - P_cr_analytical) / P_cr_analytical
        errors.append(rel_err)
    for i in range(1, len(errors)):
        assert errors[i] <= errors[i - 1] * 1.05
    assert errors[-1] < 0.01