def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler
    cantilever analytical solution. Use 10 elements and tolerances appropriate
    for the discretization.
    """
    E = 210000000000.0
    nu = 0.3
    n_elem = 10
    tol_rel = 0.001
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    for r in radii:
        A = np.pi * r ** 2
        I = np.pi * r ** 4 / 4.0
        J = np.pi * r ** 4 / 2.0
        for L in lengths:
            n_nodes = n_elem + 1
            zs = np.linspace(0.0, L, n_nodes)
            node_coords = np.vstack([np.zeros_like(zs), np.zeros_like(zs), zs]).T
            elements = []
            for i in range(n_elem):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': float(E), 'nu': float(nu), 'A': float(A), 'I_y': float(I), 'I_z': float(I), 'J': float(J), 'local_z': np.array([1.0, 0.0, 0.0])})
            boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
            nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
            (lam, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            assert lam > 0.0
            p_cr_analytic = np.pi ** 2 * E * I / (4.0 * L ** 2)
            rel_err = abs(lam - p_cr_analytic) / p_cr_analytic
            assert rel_err < tol_rel

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    Solve the cantilever in original orientation and after applying a rigid-body
    rotation R to geometry, element local_z, and applied loads. Critical load
    factor λ should be identical and the mode should transform as T @ mode_base.
    """
    E = 210000000000.0
    nu = 0.3
    n_elem = 10
    L = 20.0
    b = 0.1
    h = 0.2
    A = b * h
    I_y = b * h ** 3 / 12.0
    I_z = h * b ** 3 / 12.0
    J = 1e-06
    n_nodes = n_elem + 1
    zs = np.linspace(0.0, L, n_nodes)
    node_coords = np.vstack([np.zeros_like(zs), np.zeros_like(zs), zs]).T
    elements = []
    for i in range(n_elem):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': float(E), 'nu': float(nu), 'A': float(A), 'I_y': float(I_y), 'I_z': float(I_z), 'J': float(J), 'local_z': np.array([1.0, 0.0, 0.0])})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
    (lam_base, mode_base) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    angle = np.deg2rad(37.0)
    c = np.cos(angle)
    s = np.sin(angle)
    R = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    node_coords_rot = node_coords @ R.T
    elements_rot = []
    for el in elements:
        el_rot = el.copy()
        el_rot['local_z'] = (R @ np.asarray(el['local_z'])).astype(float)
        elements_rot.append(el_rot)
    nodal_loads_rot = {}
    for (n, load) in nodal_loads.items():
        f = np.asarray(load[:3], dtype=float)
        m = np.asarray(load[3:], dtype=float)
        f_rot = (R @ f).tolist()
        m_rot = (R @ m).tolist()
        nodal_loads_rot[n] = [f_rot[0], f_rot[1], f_rot[2], m_rot[0], m_rot[1], m_rot[2]]
    (lam_rot, mode_rot) = fcn(node_coords_rot, elements_rot, boundary_conditions, nodal_loads_rot)
    assert abs(lam_base - lam_rot) / max(1.0, abs(lam_base)) < 1e-08
    n_dof = 6 * n_nodes
    T = np.zeros((n_dof, n_dof))
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    transformed = T @ mode_base
    denom = np.dot(transformed, transformed)
    if denom == 0:
        assert np.linalg.norm(mode_rot) == 0.0
    else:
        s_fit = float(np.dot(mode_rot, transformed) / denom)
        res = mode_rot - s_fit * transformed
        rel_norm = np.linalg.norm(res) / max(1e-12, np.linalg.norm(mode_rot))
        assert rel_norm < 1e-06

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    Refine the beam discretization and check that numerical critical load approaches
    the analytical Euler value with decreasing relative error; the finest mesh
    should achieve very high accuracy.
    """
    E = 210000000000.0
    nu = 0.3
    r = 1.0
    L = 20.0
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4.0
    J = np.pi * r ** 4 / 2.0
    analytic = np.pi ** 2 * E * I / (4.0 * L ** 2)
    mesh_elems = [4, 8, 16, 32]
    prev_err = None
    errors = []
    for n_elem in mesh_elems:
        n_nodes = n_elem + 1
        zs = np.linspace(0.0, L, n_nodes)
        node_coords = np.vstack([np.zeros_like(zs), np.zeros_like(zs), zs]).T
        elements = []
        for i in range(n_elem):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': float(E), 'nu': float(nu), 'A': float(A), 'I_y': float(I), 'I_z': float(I), 'J': float(J), 'local_z': np.array([1.0, 0.0, 0.0])})
        boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
        nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
        (lam, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        rel_err = abs(lam - analytic) / analytic
        errors.append(rel_err)
        if prev_err is not None:
            assert rel_err <= prev_err + 1e-12
        prev_err = rel_err
    assert errors[-1] < 1e-05