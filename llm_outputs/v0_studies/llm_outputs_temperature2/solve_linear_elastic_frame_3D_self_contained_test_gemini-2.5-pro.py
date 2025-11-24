def test_simple_beam_discretized_axis_111(fcn):
    """Verification with resepct to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution."""
    E = 210000000000.0
    nu = 0.3
    L = 10.0
    A = 0.01
    I_z = 1e-05
    I_y = 1e-05
    J = 2e-05
    P = 1000.0
    n_elem = 10
    n_nodes = n_elem + 1
    axis_vec = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    node_coords = np.array([i * (L / n_elem) * axis_vec for i in range(n_nodes)])
    local_z_dir = np.array([1.0, 1.0, -2.0])
    local_z_unit = local_z_dir / np.linalg.norm(local_z_dir)
    elements = []
    for i in range(n_elem):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_unit})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    force_dir = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
    force_vec = P * force_dir
    nodal_loads = {n_elem: [force_vec[0], force_vec[1], force_vec[2], 0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    delta_analytical = P * L ** 3 / (3 * E * I_z)
    u_tip = u[n_elem * 6:n_elem * 6 + 3]
    delta_numerical = np.linalg.norm(u_tip)
    deflection_dir_numerical = u_tip / delta_numerical
    assert np.isclose(delta_numerical, delta_analytical, rtol=1e-09)
    assert np.allclose(deflection_dir_numerical, force_dir, atol=1e-09)

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Suggested Test stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium"""
    node_coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, -1.0]])
    n_nodes = len(node_coords)
    elem_props = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0002, 'J': 0.0003}
    elements = [{'node_i': 0, 'node_j': 1, **elem_props, 'local_z': [0.0, 0.0, 1.0]}, {'node_i': 1, 'node_j': 2, **elem_props, 'local_z': [0.0, 0.0, 1.0]}, {'node_i': 2, 'node_j': 3, **elem_props, 'local_z': [0.0, 0.0, 1.0]}, {'node_i': 0, 'node_j': 4, **elem_props, 'local_z': [1.0, 0.0, 0.0]}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1], 4: [1, 1, 1, 1, 1, 1]}
    (u0, r0) = fcn(node_coords, elements, boundary_conditions, {})
    assert np.allclose(u0, 0.0, atol=1e-12)
    assert np.allclose(r0, 0.0, atol=1e-12)
    loads1 = {2: [1000.0, -2000.0, 500.0, 100.0, -300.0, 400.0]}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, loads1)
    assert not np.allclose(u1, 0.0)
    assert not np.allclose(r1, 0.0)
    loads2 = {node: [2 * val for val in forces] for (node, forces) in loads1.items()}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, loads2)
    assert np.allclose(u2, 2 * u1)
    assert np.allclose(r2, 2 * r1)
    loads3 = {node: [-1 * val for val in forces] for (node, forces) in loads1.items()}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, loads3)
    assert np.allclose(u3, -1 * u1)
    assert np.allclose(r3, -1 * r1)
    P_applied = np.zeros(n_nodes * 6)
    for (node_idx, load_vec) in loads1.items():
        P_applied[node_idx * 6:(node_idx + 1) * 6] = load_vec
    total_ext_loads = P_applied + r1
    total_ext_loads_reshaped = total_ext_loads.reshape((n_nodes, 6))
    sum_F = np.sum(total_ext_loads_reshaped[:, :3], axis=0)
    assert np.allclose(sum_F, [0, 0, 0], atol=1e-09)
    sum_M_applied_and_reactions = np.sum(total_ext_loads_reshaped[:, 3:], axis=0)
    sum_M_from_forces = np.sum([np.cross(node_coords[i], total_ext_loads_reshaped[i, :3]) for i in range(n_nodes)], axis=0)
    sum_M_total = sum_M_applied_and_reactions + sum_M_from_forces
    assert np.allclose(sum_M_total, [0, 0, 0], atol=1e-09)

def test_ill_conditioned_due_to_under_constrained_structure(fcn):
    """Test that solve_linear_elastic_frame_3d raises a ValueError
    when the structure is improperly constrained, leading to an
    ill-conditioned free-free stiffness matrix (K_ff).
    The solver should detect this (cond(K_ff) >= 1e16) and raise a ValueError."""
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': None}]
    boundary_conditions = {}
    nodal_loads = {}
    with pytest.raises(ValueError) as excinfo:
        fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert 'ill-conditioned' in str(excinfo.value).lower()