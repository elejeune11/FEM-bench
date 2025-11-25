def test_simple_beam_discretized_axis_111(fcn):
    """Verification with respect to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution."""
    L_total = 10.0
    n_elements = 10
    L_element = L_total / n_elements
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 8.33e-06
    I_z = 8.33e-06
    J = 1.67e-05
    beam_dir = np.array([1.0, 1.0, 1.0])
    beam_dir = beam_dir / np.linalg.norm(beam_dir)
    node_coords = []
    for i in range(n_elements + 1):
        pos = i * L_element * beam_dir
        node_coords.append(pos)
    node_coords = np.array(node_coords)
    elements = []
    for i in range(n_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    tip_force_magnitude = 1000.0
    if abs(beam_dir[0]) > 1e-06:
        perp_vec = np.array([-beam_dir[1], beam_dir[0], 0])
    else:
        perp_vec = np.array([0, -beam_dir[2], beam_dir[1]])
    perp_vec = perp_vec / np.linalg.norm(perp_vec)
    tip_force = tip_force_magnitude * perp_vec
    nodal_loads = {n_elements: [tip_force[0], tip_force[1], tip_force[2], 0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    F_mag = tip_force_magnitude
    I_effective = I_z
    analytical_deflection = F_mag * L_total ** 3 / (3 * E * I_effective)
    tip_node = n_elements
    tip_disp_start = 6 * tip_node
    tip_disp = u[tip_disp_start:tip_disp_start + 3]
    computed_deflection = np.linalg.norm(tip_disp)
    assert abs(computed_deflection - analytical_deflection) / analytical_deflection < 0.05

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Suggested Test stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium"""
    node_coords = np.array([[0, 0, 0], [5, 0, 0], [5, 3, 0], [5, 3, 4]])
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 8.33e-06
    I_z = 8.33e-06
    J = 1.67e-05
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 2, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads_zero = {}
    (u_zero, r_zero) = fcn(node_coords, elements, boundary_conditions, nodal_loads_zero)
    assert np.allclose(u_zero, 0.0, atol=1e-12)
    assert np.allclose(r_zero, 0.0, atol=1e-12)
    nodal_loads_base = {3: [1000, 2000, -1500, 500, -300, 400]}
    (u_base, r_base) = fcn(node_coords, elements, boundary_conditions, nodal_loads_base)
    assert not np.allclose(u_base, 0.0, atol=1e-12)
    assert not np.allclose(r_base, 0.0, atol=1e-12)
    nodal_loads_double = {3: [2000, 4000, -3000, 1000, -600, 800]}
    (u_double, r_double) = fcn(node_coords, elements, boundary_conditions, nodal_loads_double)
    assert np.allclose(u_double, 2 * u_base, rtol=1e-10)
    assert np.allclose(r_double, 2 * r_base, rtol=1e-10)
    nodal_loads_neg = {3: [-1000, -2000, 1500, -500, 300, -400]}
    (u_neg, r_neg) = fcn(node_coords, elements, boundary_conditions, nodal_loads_neg)
    assert np.allclose(u_neg, -u_base, rtol=1e-10)
    assert np.allclose(r_neg, -r_base, rtol=1e-10)
    total_force_applied = np.zeros(3)
    total_moment_applied = np.zeros(3)
    for (node, loads) in nodal_loads_base.items():
        (Fx, Fy, Fz, Mx, My, Mz) = loads
        total_force_applied += np.array([Fx, Fy, Fz])
        node_pos = node_coords[node]
        force = np.array([Fx, Fy, Fz])
        moment_from_force = np.cross(node_pos, force)
        total_moment_applied += moment_from_force + np.array([Mx, My, Mz])
    total_force_reaction = np.zeros(3)
    total_moment_reaction = np.zeros(3)
    for (node, bc) in boundary_conditions.items():
        if any(bc):
            node_dofs = 6 * node
            (Fx, Fy, Fz, Mx, My, Mz) = r_base[node_dofs:node_dofs + 6]
            total_force_reaction += np.array([Fx, Fy, Fz])
            node_pos = node_coords[node]
            force = np.array([Fx, Fy, Fz])
            moment_from_force = np.cross(node_pos, force)
            total_moment_reaction += moment_from_force + np.array([Mx, My, Mz])
    assert np.allclose(total_force_applied + total_force_reaction, 0.0, atol=1e-10)
    assert np.allclose(total_moment_applied + total_moment_reaction, 0.0, atol=1e-10)

def test_ill_conditioned_due_to_under_constrained_structure(fcn):
    """Test that solve_linear_elastic_frame_3d raises a ValueError
    when the structure is improperly constrained, leading to an
    ill-conditioned free-free stiffness matrix (K_ff).
    The solver should detect this (cond(K_ff) >= 1e16) and raise a ValueError."""
    node_coords = np.array([[0, 0, 0], [5, 0, 0]])
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 8.33e-06
    I_z = 8.33e-06
    J = 1.67e-05
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}]
    boundary_conditions = {}
    nodal_loads = {0: [1000, 0, 0, 0, 0, 0], 1: [-1000, 0, 0, 0, 0, 0]}
    with pytest.raises(ValueError, match='ill-conditioned'):
        fcn(node_coords, elements, boundary_conditions, nodal_loads)