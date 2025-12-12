def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever value analytical solution.
    Use 10 elements, and assert relative error is within a tight tolerance suitable for high-accuracy beam elements.
    """
    E = 200000000000.0
    nu = 0.3
    P_ref = 1.0
    n_elem = 10
    tol = 0.0001
    for r in [0.5, 0.75, 1.0]:
        A = np.pi * r ** 2
        I = np.pi * r ** 4 / 4.0
        J = 2.0 * I
        for L in [10.0, 20.0, 40.0]:
            n_nodes = n_elem + 1
            z = np.linspace(0.0, L, n_nodes)
            node_coords = np.column_stack([np.zeros(n_nodes), np.zeros(n_nodes), z])
            elements = []
            for i in range(n_elem):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': [0.0, 1.0, 0.0]})
            boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
            nodal_loads = {n_nodes - 1: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
            (lam, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            assert lam > 0.0
            assert mode.shape == (6 * n_nodes,)
            P_cr_num = lam * P_ref
            P_cr_analytical = np.pi ** 2 * E * I / (4.0 * L ** 2)
            rel_err = abs(P_cr_num - P_cr_analytical) / P_cr_analytical
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
    nu = 0.29
    L = 12.0
    n_elem = 10
    P_ref = 1.0
    b = 0.04
    h = 0.08
    Iy = b * h ** 3 / 12.0
    Iz = h * b ** 3 / 12.0
    A = b * h
    J = b * h ** 3 / 3.0
    n_nodes = n_elem + 1
    z = np.linspace(0.0, L, n_nodes)
    node_coords_base = np.column_stack([np.zeros(n_nodes), np.zeros(n_nodes), z])
    elements_base = []
    for i in range(n_elem):
        elements_base.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': [0.0, 1.0, 0.0]})
    boundary_conditions_base = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads_base = {n_nodes - 1: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
    (lam_base, mode_base) = fcn(node_coords_base, elements_base, boundary_conditions_base, nodal_loads_base)
    assert lam_base > 0.0
    assert mode_base.shape == (6 * n_nodes,)
    ax = np.deg2rad(33.0)
    ay = np.deg2rad(-25.0)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(ax), -np.sin(ax)], [0.0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0.0, np.sin(ay)], [0.0, 1.0, 0.0], [-np.sin(ay), 0.0, np.cos(ay)]])
    R = Ry @ Rx
    node_coords_rot = (R @ node_coords_base.T).T
    elements_rot = []
    local_z_base = np.array([0.0, 1.0, 0.0])
    local_z_rot = (R @ local_z_base).tolist()
    for i in range(n_elem):
        elements_rot.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': local_z_rot})
    boundary_conditions_rot = {0: [1, 1, 1, 1, 1, 1]}
    F_base = np.array([0.0, 0.0, -P_ref])
    F_rot = (R @ F_base).tolist()
    nodal_loads_rot = {n_nodes - 1: [F_rot[0], F_rot[1], F_rot[2], 0.0, 0.0, 0.0]}
    (lam_rot, mode_rot) = fcn(node_coords_rot, elements_rot, boundary_conditions_rot, nodal_loads_rot)
    assert lam_rot > 0.0
    assert mode_rot.shape == (6 * n_nodes,)
    assert np.isclose(lam_base, lam_rot, rtol=1e-08, atol=0.0)
    T = np.zeros((6 * n_nodes, 6 * n_nodes))
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    x = T @ mode_base
    y = mode_rot
    denom = np.dot(x, x)
    assert denom > 0.0
    s = float(np.dot(x, y) / denom)
    rel_resid = np.linalg.norm(y - s * x) / np.linalg.norm(y)
    assert rel_resid < 1e-06

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves very high accuracy.
    """
    E = 200000000000.0
    nu = 0.3
    L = 20.0
    r = 0.6
    P_ref = 1.0
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4.0
    J = 2.0 * I
    n_elem_list = [2, 4, 8, 16, 32]
    errors = []
    for n_elem in n_elem_list:
        n_nodes = n_elem + 1
        z = np.linspace(0.0, L, n_nodes)
        node_coords = np.column_stack([np.zeros(n_nodes), np.zeros(n_nodes), z])
        elements = []
        for i in range(n_elem):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': [0.0, 1.0, 0.0]})
        boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
        nodal_loads = {n_nodes - 1: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
        (lam, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        assert lam > 0.0
        assert mode.shape == (6 * n_nodes,)
        P_cr_num = lam * P_ref
        P_cr_analytical = np.pi ** 2 * E * I / (4.0 * L ** 2)
        rel_err = abs(P_cr_num - P_cr_analytical) / P_cr_analytical
        errors.append(rel_err)
    for i in range(len(errors) - 1):
        assert errors[i + 1] <= errors[i] + 1e-12
    assert errors[-1] < 1e-06