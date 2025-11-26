def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever value analytical solution.
    Use 10 elements, set tolerances to be appropriate for the anticipated discretization error at 10-5.
    """
    E = 200000000000.0
    nu = 0.3
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    n_elems = 10
    for r in radii:
        for L in lengths:
            A = np.pi * r ** 2
            I = np.pi * r ** 4 / 4.0
            J = np.pi * r ** 4 / 2.0
            P_cr_analytic = np.pi ** 2 * E * I / (4.0 * L ** 2)
            z_coords = np.linspace(0, L, n_elems + 1)
            node_coords = np.zeros((n_elems + 1, 3))
            node_coords[:, 2] = z_coords
            elements = []
            for i in range(n_elems):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': None})
            boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
            P_ref = 1.0
            nodal_loads = {n_elems: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
            (crit_load_factor, _) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            P_cr_computed = crit_load_factor * P_ref
            assert np.isclose(P_cr_computed, P_cr_analytic, rtol=0.0001), f'Failed for r={r}, L={L}: Computed {P_cr_computed}, Expected {P_cr_analytic}'

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    The cantilever model is solved in its original orientation and again after applying
    a rigid-body rotation R to the geometry, element axes, and applied load. The critical
    load factor λ should be identical in both cases.
    The buckling mode from the rotated model should equal the base mode transformed by R:
      [ux, uy, uz] and rotational [θx, θy, θz] DOFs at each node.
    """
    L = 10.0
    n_elems = 5
    E = 100000000.0
    nu = 0.3
    (b, h) = (0.2, 0.5)
    A = b * h
    Iy = b * h ** 3 / 12.0
    Iz = h * b ** 3 / 12.0
    J = 0.01
    x_coords = np.linspace(0, L, n_elems + 1)
    nodes_base = np.zeros((n_elems + 1, 3))
    nodes_base[:, 0] = x_coords
    elements_base = []
    local_z_base = np.array([0.0, 0.0, 1.0])
    for i in range(n_elems):
        elements_base.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': local_z_base})
    bcs = {0: [1, 1, 1, 1, 1, 1]}
    P_ref = 100.0
    loads_base = {n_elems: [-P_ref, 0.0, 0.0, 0.0, 0.0, 0.0]}
    (lam_base, mode_base) = fcn(nodes_base, elements_base, bcs, loads_base)
    theta_z = np.radians(30)
    theta_y = np.radians(15)
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
    R = Rz @ Ry
    nodes_rot = nodes_base @ R.T
    elements_rot = []
    local_z_rot = R @ local_z_base
    for i in range(n_elems):
        el = elements_base[i].copy()
        el['local_z'] = local_z_rot
        elements_rot.append(el)
    loads_rot = {}
    for (node_idx, load_vec) in loads_base.items():
        F = np.array(load_vec[:3])
        M = np.array(load_vec[3:])
        F_rot = R @ F
        M_rot = R @ M
        loads_rot[node_idx] = list(F_rot) + list(M_rot)
    (lam_rot, mode_rot) = fcn(nodes_rot, elements_rot, bcs, loads_rot)
    assert np.isclose(lam_base, lam_rot, rtol=1e-05), f'Critical load changed under rotation. Base: {lam_base}, Rot: {lam_rot}'
    dofs = len(mode_base)
    num_nodes = dofs // 6
    T = np.zeros((dofs, dofs))
    for i in range(num_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    mode_base_projected = T @ mode_base

    def normalize_mode(v):
        idx = np.argmax(np.abs(v))
        if np.abs(v[idx]) < 1e-15:
            return v
        return v / v[idx]
    m1 = normalize_mode(mode_base_projected)
    m2 = normalize_mode(mode_rot)
    np.testing.assert_allclose(m1, m2, rtol=0.0001, atol=1e-05, err_msg='Rotated mode shape does not match transformed base mode')

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves very high accuracy.
    """
    L = 20.0
    r = 0.5
    E = 200000000000.0
    nu = 0.3
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4.0
    J = np.pi * r ** 4 / 2.0
    P_exact = np.pi ** 2 * E * I / (4 * L ** 2)
    P_ref = 1000.0
    element_counts = [2, 4, 8, 16]
    errors = []
    for n_elems in element_counts:
        z_coords = np.linspace(0, L, n_elems + 1)
        nodes = np.zeros((n_elems + 1, 3))
        nodes[:, 2] = z_coords
        elements = []
        for i in range(n_elems):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J})
        bcs = {0: [1] * 6}
        loads = {n_elems: [0, 0, -P_ref, 0, 0, 0]}
        (lam, _) = fcn(nodes, elements, bcs, loads)
        P_calc = lam * P_ref
        rel_err = abs(P_calc - P_exact) / P_exact
        errors.append(rel_err)
    for i in range(len(errors) - 1):
        assert errors[i + 1] < errors[i], f'Error did not decrease with refinement. n={element_counts[i]}->{element_counts[i + 1]}, err={errors[i]}->{errors[i + 1]}'
    assert errors[-1] < 0.0001, f'Finest mesh error {errors[-1]} is not sufficiently small.'