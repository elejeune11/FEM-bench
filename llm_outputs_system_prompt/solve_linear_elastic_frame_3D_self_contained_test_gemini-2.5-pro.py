def test_simple_beam_discretized_axis_111(fcn):
    """Verification with resepct to a known analytical solution.
Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
Verify beam tip deflection with the appropriate analytical reference solution."""
    L = 10.0
    E = 200000000000.0
    nu = 0.3
    I = 1e-05
    A = 0.01
    J = 2.0 * I
    P_mag = 1000.0
    num_elements = 10
    num_nodes = num_elements + 1
    delta_analytical = P_mag * L ** 3 / (3 * E * I)
    v_axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    node_coords = np.array([i * (L / num_elements) * v_axis for i in range(num_nodes)])
    v_ref = np.array([0.0, 0.0, 1.0])
    local_y = np.cross(v_ref, v_axis)
    local_y /= np.linalg.norm(local_y)
    local_z = np.cross(v_axis, local_y)
    local_z /= np.linalg.norm(local_z)
    elem_props = {'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': local_z}
    elements = [{'node_i': i, 'node_j': i + 1, **elem_props} for i in range(num_elements)]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    P_vec = P_mag * local_z
    nodal_loads = {num_nodes - 1: np.hstack([P_vec, [0, 0, 0]])}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    tip_dof_start = (num_nodes - 1) * 6
    tip_disp_vec = u[tip_dof_start:tip_dof_start + 3]
    tip_disp_mag = np.linalg.norm(tip_disp_vec)
    assert np.isclose(tip_disp_mag, delta_analytical, rtol=1e-09)
    disp_dir = tip_disp_vec / tip_disp_mag
    assert np.allclose(disp_dir, local_z, atol=1e-09)

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
Suggested Test stages:
1. Zero loads -> All displacements and reactions should be zero.
2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
3. Double the loads -> Displacements and reactions should double (linearity check).
4. Negate the original loads -> Displacements and reactions should flip sign.
5. Assure that reactions (forces and moments) satisfy global static equilibrium"""
    (L, H) = (2.0, 1.0)
    node_coords = np.array([[0, 0, 0], [L, 0, 0], [L, L, 0], [0, L, 0], [0, 0, H], [L, 0, H], [L, L, H], [0, L, H]])
    num_nodes = len(node_coords)
    leg_elements_conn = [{'node_i': i, 'node_j': i + 4} for i in range(4)]
    top_elements_conn = [{'node_i': 4, 'node_j': 5}, {'node_i': 5, 'node_j': 6}, {'node_i': 6, 'node_j': 7}, {'node_i': 7, 'node_j': 4}]
    props = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}
    elements = []
    for el in leg_elements_conn:
        el.update(props)
        el['local_z'] = np.array([1.0, 0.0, 0.0])
        elements.append(el)
    for el in top_elements_conn:
        el.update(props)
        el['local_z'] = np.array([0.0, 0.0, 1.0])
        elements.append(el)
    boundary_conditions = {i: [1, 1, 1, 1, 1, 1] for i in range(4)}
    (u0, r0) = fcn(node_coords, elements, boundary_conditions, {})
    assert np.allclose(u0, 0.0, atol=1e-12)
    assert np.allclose(r0, 0.0, atol=1e-12)
    loads1 = {7: [1000.0, 2000.0, -500.0, 100.0, -200.0, 50.0]}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, loads1)
    assert not np.allclose(u1, 0.0)
    assert not np.allclose(r1, 0.0)
    loads2 = {k: [2 * val for val in v] for (k, v) in loads1.items()}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, loads2)
    assert np.allclose(u2, 2 * u1)
    assert np.allclose(r2, 2 * r1)
    loads3 = {k: [-1 * val for val in v] for (k, v) in loads1.items()}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, loads3)
    assert np.allclose(u3, -1 * u1)
    assert np.allclose(r3, -1 * r1)
    applied_loads_vector = np.zeros(6 * num_nodes)
    for (node_idx, load) in loads1.items():
        applied_loads_vector[node_idx * 6:node_idx * 6 + 6] = load
    total_external_forces_vector = applied_loads_vector + r1
    sum_forces = np.zeros(3)
    sum_moments_origin = np.zeros(3)
    for i in range(num_nodes):
        F_i = total_external_forces_vector[i * 6:i * 6 + 3]
        M_i = total_external_forces_vector[i * 6 + 3:i * 6 + 6]
        pos_i = node_coords[i]
        sum_forces += F_i
        sum_moments_origin += M_i + np.cross(pos_i, F_i)
    assert np.allclose(sum_forces, 0.0, atol=1e-08)
    assert np.allclose(sum_moments_origin, 0.0, atol=1e-08)

def test_ill_conditioned_due_to_under_constrained_structure(fcn):
    """Test that solve_linear_elastic_frame_3d raises a ValueError
when the structure is improperly constrained, leading to an
ill-conditioned free-free stiffness matrix (K_ff).
The solver should detect this (cond(K_ff) >= 1e16) and raise a ValueError."""
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': np.array([0.0, 0.0, 1.0])}]
    boundary_conditions_none = {}
    nodal_loads = {1: [0.0, 1000.0, 0.0, 0.0, 0.0, 0.0]}
    with pytest.raises(ValueError, match='ill-conditioned'):
        fcn(node_coords, elements, boundary_conditions_none, nodal_loads)
    boundary_conditions_pinned = {0: [1, 1, 1, 0, 0, 0]}
    nodal_loads_torsion = {1: [0.0, 0.0, 0.0, 100.0, 0.0, 0.0]}
    with pytest.raises(ValueError, match='ill-conditioned'):
        fcn(node_coords, elements, boundary_conditions_pinned, nodal_loads_torsion)