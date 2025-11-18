def test_simple_beam_discretized_axis_111(fcn):
    """Verification with resepct to a known analytical solution.
Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
Verify beam tip deflection with the appropriate analytical reference solution."""
    L = 10.0
    E = 210000000000.0
    nu = 0.3
    G = E / (2 * (1 + nu))
    A = 0.01
    I_y = 1e-05
    I_z = 1e-05
    J = 2e-05
    P_val = 1000.0
    num_elements = 10
    num_nodes = num_elements + 1
    beam_axis_vec = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    node_coords = np.array([i * (L / num_elements) * beam_axis_vec for i in range(num_nodes)])
    force_dir = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
    force_vec = P_val * force_dir
    local_z_vec = force_dir
    elements = []
    for i in range(num_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_vec})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {num_nodes - 1: np.concatenate((force_vec, [0, 0, 0]))}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    delta_analytical = P_val * L ** 3 / (3 * E * I_y)
    tip_disp_vec = u[(num_nodes - 1) * 6:(num_nodes - 1) * 6 + 3]
    tip_disp_mag = np.linalg.norm(tip_disp_vec)
    assert np.isclose(tip_disp_mag, delta_analytical, rtol=1e-09)
    tip_disp_dir = tip_disp_vec / tip_disp_mag
    assert np.allclose(tip_disp_dir, force_dir)
    total_applied_loads = np.zeros_like(r)
    for (node_idx, loads) in nodal_loads.items():
        total_applied_loads[node_idx * 6:(node_idx + 1) * 6] = loads
    assert np.allclose(total_applied_loads + r, 0, atol=1e-09)

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
Suggested Test stages:
1. Zero loads -> All displacements and reactions should be zero.
2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
3. Double the loads -> Displacements and reactions should double (linearity check).
4. Negate the original loads -> Displacements and reactions should flip sign.
5. Assure that reactions (forces and moments) satisfy global static equilibrium"""
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    num_nodes = len(node_coords)
    elem_props = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': [0, 0, 1]}
    elements = [{'node_i': 0, 'node_j': 1, **elem_props}, {'node_i': 1, 'node_j': 2, **elem_props}, {'node_i': 2, 'node_j': 3, **elem_props}, {'node_i': 3, 'node_j': 0, **elem_props}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1], 1: [1, 1, 1, 0, 0, 0]}
    (u0, r0) = fcn(node_coords, elements, boundary_conditions, {})
    assert np.all(u0 == 0)
    assert np.all(r0 == 0)
    loads1 = {2: [1000.0, 2000.0, -1500.0, 500.0, -800.0, 1000.0], 3: [-500.0, 0, 1000.0, 0, 400.0, 0]}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, loads1)
    assert not np.allclose(u1, 0)
    assert not np.allclose(r1, 0)
    loads2 = {k: [2 * val for val in v] for (k, v) in loads1.items()}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, loads2)
    assert np.allclose(u2, 2 * u1)
    assert np.allclose(r2, 2 * r1)
    loads3 = {k: [-1 * val for val in v] for (k, v) in loads1.items()}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, loads3)
    assert np.allclose(u3, -1 * u1)
    assert np.allclose(r3, -1 * r1)
    P = np.zeros(6 * num_nodes)
    for (node_idx, load_vec) in loads1.items():
        P[node_idx * 6:(node_idx + 1) * 6] = load_vec
    external_forces_and_moments = (P + r1).reshape(num_nodes, 6)
    total_force = np.sum(external_forces_and_moments[:, :3], axis=0)
    assert np.allclose(total_force, 0, atol=1e-08)
    total_moment_about_origin = np.zeros(3)
    for i in range(num_nodes):
        pos = node_coords[i]
        force = external_forces_and_moments[i, :3]
        moment = external_forces_and_moments[i, 3:]
        total_moment_about_origin += moment + np.cross(pos, force)
    assert np.allclose(total_moment_about_origin, 0, atol=1e-08)

def test_ill_conditioned_due_to_under_constrained_structure(fcn):
    """Test that solve_linear_elastic_frame_3d raises a ValueError
when the structure is improperly constrained, leading to an
ill-conditioned free-free stiffness matrix (K_ff).
The solver should detect this (cond(K_ff) >= 1e16) and raise a ValueError."""
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': [0, 0, 1]}]
    boundary_conditions = {}
    nodal_loads = {}
    with pytest.raises(ValueError, match='(?i)ill-conditioned'):
        fcn(node_coords, elements, boundary_conditions, nodal_loads)
    boundary_conditions_pin = {0: [1, 1, 1, 0, 0, 0]}
    with pytest.raises(ValueError, match='(?i)ill-conditioned'):
        fcn(node_coords, elements, boundary_conditions_pin, nodal_loads)
    elements_flexible = [{'node_i': 0, 'node_j': 1, 'E': 1.0, 'nu': 0.3, 'A': 1e-06, 'I_y': 1e-12, 'I_z': 1e-12, 'J': 2e-12, 'local_z': [0, 0, 1]}]
    boundary_conditions_stable = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads_stable = {1: [0, 1, 0, 0, 0, 0]}
    try:
        fcn(node_coords, elements_flexible, boundary_conditions_stable, nodal_loads_stable)
    except ValueError:
        pytest.fail('ValueError raised for a stable but flexible structure.')