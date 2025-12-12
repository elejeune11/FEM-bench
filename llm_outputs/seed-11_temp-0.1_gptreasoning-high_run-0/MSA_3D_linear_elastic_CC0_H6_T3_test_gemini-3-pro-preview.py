def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with resepct to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution.
    """
    end_coord = np.array([10.0, 10.0, 10.0])
    L_total = np.linalg.norm(end_coord)
    n_elem = 10
    n_nodes = n_elem + 1
    node_coords = np.linspace(np.zeros(3), end_coord, n_nodes)
    E_val = 200000000000.0
    nu_val = 0.3
    b = 0.1
    h = 0.1
    A_val = b * h
    I_val = b * h ** 3 / 12.0
    J_val = 0.1406 * b ** 4
    elements = []
    for i in range(n_elem):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E_val, 'nu': nu_val, 'A': A_val, 'I_y': I_val, 'I_z': I_val, 'J': J_val})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    F_mag = 1000.0
    force_dir = np.array([1.0, -1.0, 0.0])
    force_dir = force_dir / np.linalg.norm(force_dir)
    F_vec = F_mag * force_dir
    tip_load = np.zeros(6)
    tip_load[:3] = F_vec
    nodal_loads = {n_elem: tip_load}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    expected_deflection = F_mag * L_total ** 3 / (3.0 * E_val * I_val)
    tip_u_indices = slice(n_elem * 6, n_elem * 6 + 3)
    calc_disp_vec = u[tip_u_indices]
    calc_deflection = np.linalg.norm(calc_disp_vec)
    assert np.isclose(calc_deflection, expected_deflection, rtol=0.01), f'Computed deflection {calc_deflection:.6f} differs from analytical {expected_deflection:.6f}'
    if calc_deflection > 1e-09:
        calc_dir = calc_disp_vec / calc_deflection
        dot_product = np.dot(calc_dir, force_dir)
        assert np.isclose(dot_product, 1.0, atol=0.001), f'Displacement direction {calc_dir} not aligned with force direction {force_dir}'

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
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [2.0, 2.0, 2.0], [4.0, 0.0, 0.0]])
    props = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}
    elements = [{'node_i': 0, 'node_j': 1, **props}, {'node_i': 1, 'node_j': 2, **props}, {'node_i': 2, 'node_j': 3, **props}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1], 3: [1, 1, 1, 1, 1, 1]}
    nodal_loads_0 = {}
    (u0, r0) = fcn(coords, elements, boundary_conditions, nodal_loads_0)
    assert np.allclose(u0, 0.0), 'Stage 1: Displacements should be zero with no loads'
    assert np.allclose(r0, 0.0), 'Stage 1: Reactions should be zero with no loads'
    loads_1 = {1: np.array([1000.0, 0.0, 0.0, 0.0, 0.0, 500.0]), 2: np.array([0.0, -2000.0, 0.0, 0.0, 0.0, 0.0])}
    (u1, r1) = fcn(coords, elements, boundary_conditions, loads_1)
    assert np.linalg.norm(u1) > 1e-06, 'Stage 2: Structure should deform under load'
    assert np.linalg.norm(r1) > 1e-06, 'Stage 2: Reactions should develop'
    loads_2 = {node: val * 2.0 for (node, val) in loads_1.items()}
    (u2, r2) = fcn(coords, elements, boundary_conditions, loads_2)
    assert np.allclose(u2, 2.0 * u1), 'Stage 3: Displacements should double with double load'
    assert np.allclose(r2, 2.0 * r1), 'Stage 3: Reactions should double with double load'
    loads_neg = {node: val * -1.0 for (node, val) in loads_1.items()}
    (u_neg, r_neg) = fcn(coords, elements, boundary_conditions, loads_neg)
    assert np.allclose(u_neg, -u1), 'Stage 4: Displacements should flip sign'
    assert np.allclose(r_neg, -r1), 'Stage 4: Reactions should flip sign'
    total_force = np.zeros(3)
    total_moment = np.zeros(3)
    n_nodes = coords.shape[0]
    for i in range(n_nodes):
        F_app = loads_1.get(i, np.zeros(6))
        F_reac = r1[i * 6:i * 6 + 6]
        F_total = F_app + F_reac
        (fx, fy, fz) = F_total[:3]
        (mx, my, mz) = F_total[3:]
        force_vec = np.array([fx, fy, fz])
        moment_vec = np.array([mx, my, mz])
        pos_vec = coords[i]
        total_force += force_vec
        total_moment += moment_vec + np.cross(pos_vec, force_vec)
    assert np.allclose(total_force, 0.0, atol=0.0001), f'Stage 5: Force equilibrium violated. Residual: {total_force}'
    assert np.allclose(total_moment, 0.0, atol=0.0001), f'Stage 5: Moment equilibrium violated. Residual: {total_moment}'