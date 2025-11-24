def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with resepct to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force
    perpendicular to the beam axis. Verify beam tip deflection with the appropriate analytical reference solution.
    """
    L = 3.0
    e = np.array([1.0, 1.0, 1.0])
    e = e / np.linalg.norm(e)
    n_el = 10
    n_nodes = n_el + 1
    node_coords = np.array([i * (L / n_el) * e for i in range(n_nodes)])
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I = 1e-05
    Iy = I
    Iz = I
    J = 2.0 * I
    local_z = np.array([0.0, 0.0, 1.0])
    elements = [dict(node_i=i, node_j=i + 1, E=E, nu=nu, A=A, I_y=Iy, I_z=Iz, J=J, local_z=local_z) for i in range(n_el)]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    v = np.array([0.0, 1.0, 0.0])
    F_perp = v - np.dot(v, e) * e
    F_perp /= np.linalg.norm(F_perp)
    P = 1000.0
    F = P * F_perp
    nodal_loads = {n_nodes - 1: [F[0], F[1], F[2], 0.0, 0.0, 0.0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    u_tip = u[6 * (n_nodes - 1):6 * (n_nodes - 1) + 3]
    delta_expected = P * L ** 3 / (3.0 * E * I)
    disp_along_load = float(np.dot(u_tip, F_perp))
    assert np.isclose(disp_along_load, delta_expected, rtol=0.0001, atol=1e-09)
    axial_disp = float(np.dot(u_tip, e))
    assert np.isclose(axial_disp, 0.0, atol=1e-08)
    r_forces = np.array([r[6 * i:6 * i + 3] for i in range(n_nodes)])
    r_moments = np.array([r[6 * i + 3:6 * i + 6] for i in range(n_nodes)])
    for i in range(1, n_nodes):
        assert np.allclose(r_forces[i], 0.0, atol=1e-09)
        assert np.allclose(r_moments[i], 0.0, atol=1e-09)
    total_reaction_force = r_forces.sum(axis=0)
    total_applied_force = np.zeros(3)
    total_applied_force += F
    assert np.allclose(total_reaction_force + total_applied_force, 0.0, atol=1e-08)
    coords = node_coords
    total_reaction_moment = r_moments.sum(axis=0)
    total_applied_moment = np.zeros(3)
    arm_forces = np.zeros(3)
    for i in range(n_nodes):
        arm_forces += np.cross(coords[i], r_forces[i])
    arm_forces += np.cross(coords[-1], F)
    total_moment_balance = total_reaction_moment + total_applied_moment + arm_forces
    assert np.allclose(total_moment_balance, 0.0, atol=1e-06)

def test_complex_geometry_and_basic_loading(fcn):
    """
    Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium.
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 0.5], [3.0, 2.0, 2.0], [1.0, 2.5, 3.5]])
    E = 70000000000.0
    nu = 0.33
    A = 0.008
    I = 5e-06
    Iy = I
    Iz = I
    J = 2.0 * I
    local_z = np.array([0.0, 0.0, 1.0])
    elements = [dict(node_i=0, node_j=1, E=E, nu=nu, A=A, I_y=Iy, I_z=Iz, J=J, local_z=local_z), dict(node_i=1, node_j=2, E=E, nu=nu, A=A, I_y=Iy, I_z=Iz, J=J, local_z=local_z), dict(node_i=2, node_j=3, E=E, nu=nu, A=A, I_y=Iy, I_z=Iz, J=J, local_z=local_z)]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    loads0 = {}
    (u0, r0) = fcn(node_coords, elements, boundary_conditions, loads0)
    assert np.allclose(u0, 0.0, atol=1e-12)
    assert np.allclose(r0, 0.0, atol=1e-12)
    loads1 = {1: [50.0, -120.0, 30.0, 10.0, 5.0, -8.0], 2: [-40.0, 80.0, -20.0, 0.0, -15.0, 12.0]}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, loads1)
    assert not np.allclose(u1, 0.0, atol=1e-12)
    r_forces_1 = np.array([r1[6 * i:6 * i + 3] for i in range(len(node_coords))])
    r_moments_1 = np.array([r1[6 * i + 3:6 * i + 6] for i in range(len(node_coords))])
    assert not np.allclose(r_forces_1[0], 0.0, atol=1e-12) or not np.allclose(r_moments_1[0], 0.0, atol=1e-12)
    for i in range(1, len(node_coords)):
        assert np.allclose(r_forces_1[i], 0.0, atol=1e-09)
        assert np.allclose(r_moments_1[i], 0.0, atol=1e-09)
    loads2 = {k: [2 * v for v in loads1[k]] for k in loads1}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, loads2)
    assert np.allclose(u2, 2.0 * u1, rtol=1e-08, atol=1e-10)
    assert np.allclose(r2, 2.0 * r1, rtol=1e-08, atol=1e-10)
    loads3 = {k: [-v for v in loads1[k]] for k in loads1}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, loads3)
    assert np.allclose(u3, -u1, rtol=1e-08, atol=1e-10)
    assert np.allclose(r3, -r1, rtol=1e-08, atol=1e-10)

    def check_global_equilibrium(node_coords, r_vec, loads_dict):
        n = len(node_coords)
        r_forces = np.array([r_vec[6 * i:6 * i + 3] for i in range(n)])
        r_moments = np.array([r_vec[6 * i + 3:6 * i + 6] for i in range(n)])
        sum_F_loads = np.zeros(3)
        sum_F_react = r_forces.sum(axis=0)
        for (k, vals) in loads_dict.items():
            sum_F_loads += np.array(vals[:3])
        assert np.allclose(sum_F_loads + sum_F_react, 0.0, atol=1e-07)
        sum_M_loads = np.zeros(3)
        sum_M_react = r_moments.sum(axis=0)
        arm_sum = np.zeros(3)
        for i in range(n):
            F_total_i = r_forces[i].copy()
            if i in loads_dict:
                F_total_i += np.array(loads_dict[i][:3])
                sum_M_loads += np.array(loads_dict[i][3:6])
            arm_sum += np.cross(node_coords[i], F_total_i)
        total_moment = sum_M_loads + sum_M_react + arm_sum
        assert np.allclose(total_moment, 0.0, atol=1e-06)
    check_global_equilibrium(node_coords, r1, loads1)
    check_global_equilibrium(node_coords, r2, loads2)
    check_global_equilibrium(node_coords, r3, loads3)

def test_ill_conditioned_due_to_under_constrained_structure(fcn):
    """
    Test that solve_linear_elastic_frame_3d raises a ValueError when the structure is improperly constrained,
    leading to an ill-conditioned free-free stiffness matrix (K_ff). The solver should detect this and raise ValueError.
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    I = 1e-05
    Iy = I
    Iz = I
    J = 2 * I
    local_z = np.array([0.0, 0.0, 1.0])
    elements = [dict(node_i=0, node_j=1, E=E, nu=nu, A=A, I_y=Iy, I_z=Iz, J=J, local_z=local_z)]
    boundary_conditions = {}
    nodal_loads = {}
    with pytest.raises(ValueError):
        fcn(node_coords, elements, boundary_conditions, nodal_loads)