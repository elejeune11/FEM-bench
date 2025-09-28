def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with resepct to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution.
    """
    n_el = 10
    n_nodes = n_el + 1
    L = 3.0
    axis_dir = np.array([1.0, 1.0, 1.0])
    axis_dir = axis_dir / np.linalg.norm(axis_dir)
    total_vec = axis_dir * L
    node_coords = np.array([i / n_el * total_vec for i in range(n_nodes)])
    E = 210000000000.0
    nu = 0.3
    A = 0.001
    I = 5e-06
    Iy = I
    Iz = I
    J = 2e-05
    z_ref = np.array([0.0, 0.0, 1.0])
    elements = []
    for i in range(n_el):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': z_ref})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    y_local = np.cross(z_ref, axis_dir)
    y_norm = np.linalg.norm(y_local)
    if y_norm < 1e-12:
        z_ref = np.array([0.0, 1.0, 0.0])
        y_local = np.cross(z_ref, axis_dir)
        y_norm = np.linalg.norm(y_local)
    y_local = y_local / y_norm
    P = 1000.0
    load_vector_tip = np.zeros(6)
    load_vector_tip[:3] = P * y_local
    nodal_loads = {n_nodes - 1: load_vector_tip}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    tip_idx = n_nodes - 1
    u_tip = u[6 * tip_idx:6 * tip_idx + 3]
    delta = P * L ** 3 / (3.0 * E * I)
    disp_along_load = np.dot(u_tip, y_local)
    assert np.isclose(disp_along_load, delta, rtol=0.02, atol=1e-08)

def test_complex_geometry_and_basic_loading(fcn):
    """
    Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Suggested Test stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 1.0, 1.2], [0.5, 1.5, 2.0]], dtype=float)
    E = 70000000000.0
    nu = 0.29
    A = 0.0015
    Iy = 3.2e-06
    Iz = 4.1e-06
    J = 5e-06

    def pick_local_z(pi, pj):
        axis = pj - pi
        axis_u = axis / np.linalg.norm(axis)
        z0 = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(axis_u, z0)) > 0.95:
            z0 = np.array([0.0, 1.0, 0.0])
            if abs(np.dot(axis_u, z0)) > 0.95:
                z0 = np.array([1.0, 0.0, 0.0])
        return z0
    conn = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 4)]
    elements = []
    for (ni, nj) in conn:
        z_ref = pick_local_z(node_coords[ni], node_coords[nj])
        elements.append({'node_i': ni, 'node_j': nj, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': z_ref})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {}
    (u0, r0) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u0, 0.0, atol=1e-12)
    assert np.allclose(r0, 0.0, atol=1e-12)
    loads1 = {2: np.array([+500.0, -300.0, +200.0, +100.0, -50.0, +70.0]), 4: np.array([-120.0, +400.0, -250.0, +40.0, +60.0, -30.0])}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, loads1)
    assert not np.allclose(u1, 0.0)
    fixed_nodes = [n for (n, mask) in boundary_conditions.items() if any(mask)]
    fixed_dofs = []
    for n in fixed_nodes:
        mask = boundary_conditions[n]
        for (k, m) in enumerate(mask):
            if m:
                fixed_dofs.append(6 * n + k)
    assert np.linalg.norm(r1[fixed_dofs]) > 0.0
    assert np.linalg.norm(r1) > 0.0
    sum_F_loads = np.zeros(3)
    sum_M_loads = np.zeros(3)
    sum_r_cross_F = np.zeros(3)
    for (n, load) in loads1.items():
        F = load[:3]
        M = load[3:]
        r_pos = node_coords[n]
        sum_F_loads += F
        sum_M_loads += M
        sum_r_cross_F += np.cross(r_pos, F)
    sum_R_forces = np.zeros(3)
    sum_R_moments = np.zeros(3)
    for n in fixed_nodes:
        Rf = r1[6 * n:6 * n + 3]
        Rm = r1[6 * n + 3:6 * n + 6]
        sum_R_forces += Rf
        sum_R_moments += Rm
    assert np.allclose(sum_R_forces + sum_F_loads, 0.0, rtol=1e-09, atol=1e-06)
    assert np.allclose(sum_R_moments + sum_M_loads + sum_r_cross_F, 0.0, rtol=1e-09, atol=1e-06)
    loads2 = {n: 2.0 * v for (n, v) in loads1.items()}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, loads2)
    assert np.allclose(u2, 2.0 * u1, rtol=1e-10, atol=1e-09)
    assert np.allclose(r2, 2.0 * r1, rtol=1e-10, atol=1e-09)
    loads3 = {n: -1.0 * v for (n, v) in loads1.items()}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, loads3)
    assert np.allclose(u3, -1.0 * u1, rtol=1e-10, atol=1e-09)
    assert np.allclose(r3, -1.0 * r1, rtol=1e-10, atol=1e-09)

def test_ill_conditioned_due_to_under_constrained_structure(fcn):
    """
    Test that solve_linear_elastic_frame_3d raises a ValueError
    when the structure is improperly constrained, leading to an
    ill-conditioned free-free stiffness matrix (K_ff).
    The solver should detect this (cond(K_ff) >= 1e16) and raise a ValueError.
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    E = 200000000000.0
    nu = 0.3
    A = 0.002
    Iy = 1e-06
    Iz = 1.5e-06
    J = 1e-05
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}]
    boundary_conditions = {}
    nodal_loads = {}
    with pytest.raises(ValueError):
        fcn(node_coords, elements, boundary_conditions, nodal_loads)