def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever
    analytical solution. Use 10 elements and tolerances appropriate for 1e-5.
    """
    E = 210000000000.0
    nu = 0.3
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    n_elems = 10
    tol_rel = 1e-05
    for r in radii:
        A = math.pi * r ** 2
        I = math.pi * r ** 4 / 4.0
        J = math.pi * r ** 4 / 2.0
        for L in lengths:
            node_coords = np.vstack([np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, L])])
            (coords, elements, bc) = _build_cantilever(node_coords, n_elems, E, nu, A, I, I, J, [0.0, 1.0, 0.0])
            F_ref = -1.0
            nodal_loads = {n_elems: [0.0, 0.0, F_ref, 0.0, 0.0, 0.0]}
            (lam, mode) = fcn(coords, elements, bc, nodal_loads)
            Pcr_expected = _analytical_euler_cantilever(E, I, L)
            Pcr_numerical = abs(lam * F_ref)
            rel_err = abs(Pcr_numerical - Pcr_expected) / Pcr_expected
            assert rel_err < tol_rel, f'r={r},L={L}: rel_err={rel_err} exceeds {tol_rel}'

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """Orientation invariance test with a rectangular section (Iy ≠ Iz).
    Solve cantilever in original orientation and after applying a rigid-body
    rotation R to geometry, element axes, and applied load. The critical load
    factor λ should be identical. The rotated mode should equal T @ mode_base
    up to scale and sign.
    """
    E = 210000000000.0
    nu = 0.3
    L = 10.0
    n_elems = 12
    b = 0.2
    h = 0.05
    I_y = b * h ** 3 / 12.0
    I_z = h * b ** 3 / 12.0
    A = b * h
    J = 2.0 * min(I_y, I_z)
    node_coords_base = np.vstack([np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, L])])
    (coords_base, elements_base, bc_base) = _build_cantilever(node_coords_base, n_elems, E, nu, A, I_y, I_z, J, [0.0, 1.0, 0.0])
    F_ref = -1.0
    nodal_loads_base = {n_elems: [0.0, 0.0, F_ref, 0.0, 0.0, 0.0]}
    (lam_base, mode_base) = fcn(coords_base, elements_base, bc_base, nodal_loads_base)
    theta = math.radians(37.0)
    R = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(theta), -math.sin(theta)], [0.0, math.sin(theta), math.cos(theta)]])
    n_nodes = n_elems + 1
    coords_full = np.zeros((n_nodes, 3))
    for i in range(n_nodes):
        coords_full[i] = np.array([0.0, 0.0, L * i / n_elems])
    coords_rot = (R @ coords_full.T).T
    elements_rot = []
    for el in elements_base:
        el_rot = dict(el)
        el_rot['local_z'] = (R @ np.asarray(el['local_z']).reshape(3)).reshape(3)
        elements_rot.append(el_rot)
    nodal_loads_rot = {}
    load_vec_trans = np.array([0.0, 0.0, F_ref])
    load_vec_rot = R @ load_vec_trans
    nodal_loads_rot[n_elems] = [float(load_vec_rot[0]), float(load_vec_rot[1]), float(load_vec_rot[2]), 0.0, 0.0, 0.0]
    (lam_rot, mode_rot) = fcn(coords_rot, elements_rot, bc_base, nodal_loads_rot)
    assert abs(lam_base - lam_rot) <= 1e-09 * max(1.0, abs(lam_base)), f'lambda mismatch: {lam_base} vs {lam_rot}'
    T = np.zeros((6 * n_nodes, 6 * n_nodes))
    for i in range(n_nodes):
        idx = 6 * i
        T[idx:idx + 3, idx:idx + 3] = R
        T[idx + 3:idx + 6, idx + 3:idx + 6] = R
    transformed_mode = T @ mode_base
    v = transformed_mode
    w = mode_rot
    denom = np.dot(v, v)
    assert denom > 0.0
    s = np.dot(w, v) / denom
    residual = w - s * v
    rel_residual = np.linalg.norm(residual) / max(1e-12, np.linalg.norm(w))
    assert rel_residual < 1e-06, f'mode transform residual too large: {rel_residual}'

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    Refine the beam discretization and check that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves high accuracy.
    """
    E = 210000000000.0
    nu = 0.3
    r = 1.0
    L = 20.0
    radii = [r]
    I = math.pi * r ** 4 / 4.0
    A = math.pi * r ** 2
    J = math.pi * r ** 4 / 2.0
    n_elems_list = [2, 4, 8, 16, 32]
    F_ref = -1.0
    errors = []
    for n_elems in n_elems_list:
        node_coords = np.vstack([np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, L])])
        (coords, elements, bc) = _build_cantilever(node_coords, n_elems, E, nu, A, I, I, J, [0.0, 1.0, 0.0])
        nodal_loads = {n_elems: [0.0, 0.0, F_ref, 0.0, 0.0, 0.0]}
        (lam, mode) = fcn(coords, elements, bc, nodal_loads)
        Pcr_num = abs(lam * F_ref)
        Pcr_exact = _analytical_euler_cantilever(E, I, L)
        rel_err = abs(Pcr_num - Pcr_exact) / Pcr_exact
        errors.append(rel_err)
    for i in range(len(errors) - 1):
        assert errors[i + 1] <= errors[i] + 1e-12, f'Error not decreasing at refinement step {i}: {errors[i]} -> {errors[i + 1]}'
    assert errors[-1] < 1e-05, f'Finest mesh relative error too large: {errors[-1]}'