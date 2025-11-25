def test_simple_beam_discretized_axis_111(fcn):
    """Verification with resepct to a known analytical solution.
Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
Verify beam tip deflection with the appropriate analytical reference solution."""
    n_nodes = 11
    n_elements = 10
    node_coords = np.array([[i, i, i] for i in range(n_nodes)], dtype=float)
    E = 210000000000.0
    nu = 0.3
    A = 1.0
    I_y = 0.0001
    I_z = 0.0001
    J = 0.0002
    P_load = 1000.0
    local_z_vec = np.array([-1.0, 1.0, 0.0])
    local_z_vec /= np.linalg.norm(local_z_vec)
    elements = []
    for i in range(n_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_vec})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    force_direction = local_z_vec
    nodal_loads = {n_nodes - 1: list(P_load * force_direction) + [0, 0, 0]}
    L = 10.0 * np.sqrt(3.0)
    I = I_y
    delta_analytical = P_load * L ** 3 / (3 * E * I)
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    tip_node_idx = n_nodes - 1
    tip_disp_global = u[tip_node_idx * 6:tip_node_idx * 6 + 3]
    delta_computed = np.linalg.norm(tip_disp_global)
    assert np.isclose(delta_computed, delta_analytical, rtol=0.001)
    disp_direction = tip_disp_global / delta_computed
    assert np.allclose(disp_direction, force_direction)

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
Suggested Test stages:
1. Zero loads -> All displacements and reactions should be zero.
2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
3. Double the loads -> Displacements and reactions should double (linearity check).
4. Negate the original loads -> Displacements and reactions should flip sign.
5. Assure that reactions (forces and moments) satisfy global static equilibrium"""
    node_coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0], [4.0, 0.0, 5.0], [4.0, 6.0, 5.0], [0.0, 6.0, 5.0], [0.0, 6.0, 0.0]], dtype=float)
    props = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0002, 'J': 0.0003}
    elements = [{'node_i': 0, 'node_j': 1, **props, 'local_z': [1.0, 0.0, 0.0]}, {'node_i': 1, 'node_j': 2, **props, 'local_z': [0.0, 0.0, 1.0]}, {'node_i': 2, 'node_j': 3, **props, 'local_z': [0.0, 0.0, 1.0]}, {'node_i': 3, 'node_j': 4, **props, 'local_z': [0.0, 0.0, 1.0]}, {'node_i': 4, 'node_j': 1, **props, 'local_z': [0.0, 0.0, 1.0]}, {'node_i': 4, 'node_j': 5, **props, 'local_z': [1.0, 0.0, 0.0]}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1], 5: [1, 1, 1, 1, 1, 1]}
    nodal_loads_zero = {}
    (u0, r0) = fcn(node_coords, elements, boundary_conditions, nodal_loads_zero)
    assert np.all(u0 == 0)
    assert np.all(r0 == 0)
    nodal_loads_base = {2: np.array([1000.0, 2000.0, -1500.0, 5000.0, -3000.0, 4000.0])}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, nodal_loads_base)
    assert np.any(u1 != 0)
    assert np.any(r1 != 0)
    nodal_loads_double = {k: 2 * v for (k, v) in nodal_loads_base.items()}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, nodal_loads_double)
    assert np.allclose(u2, 2 * u1)
    assert np.allclose(r2, 2 * r1)
    nodal_loads_neg = {k: -1 * v for (k, v) in nodal_loads_base.items()}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, nodal_loads_neg)
    assert np.allclose(u3, -1 * u1)
    assert np.allclose(r3, -1 * r1)
    total_applied_force = np.sum([v[:3] for v in nodal_loads_base.values()], axis=0)
    total_reaction_force = r1[0:3] + r1[5 * 6:5 * 6 + 3]
    assert np.allclose(total_applied_force + total_reaction_force, 0, atol=1e-09)
    total_applied_moment = np.sum([v[3:] for v in nodal_loads_base.values()], axis=0)
    for (node_idx, load) in nodal_loads_base.items():
        total_applied_moment += np.cross(node_coords[node_idx], load[:3])
    total_reaction_moment = r1[3:6] + r1[5 * 6 + 3:5 * 6 + 6]
    total_reaction_moment += np.cross(node_coords[0], r1[0:3])
    total_reaction_moment += np.cross(node_coords[5], r1[5 * 6:5 * 6 + 3])
    assert np.allclose(total_applied_moment + total_reaction_moment, 0, atol=1e-09)

def test_ill_conditioned_due_to_under_constrained_structure(fcn):
    """Test that solve_linear_elastic_frame_3d raises a ValueError
when the structure is improperly constrained, leading to an
ill-conditioned free-free stiffness matrix (K_ff).
The solver should detect this (cond(K_ff) >= 1e16) and raise a ValueError."""
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 1.0, 'I_y': 1.0, 'I_z': 1.0, 'J': 1.0, 'local_z': [0.0, 0.0, 1.0]}]
    boundary_conditions = {}
    nodal_loads = {}
    with pytest.raises(ValueError, match='ill-conditioned'):
        fcn(node_coords, elements, boundary_conditions, nodal_loads)