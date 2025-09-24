def test_simple_beam_discretized_axis_111(fcn):
    """Verification with resepct to a known analytical solution.
Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
Verify beam tip deflection with the appropriate analytical reference solution."""
    L = 10.0
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I = 1e-05
    J = 2 * I
    P_load = 1000.0
    n_elem = 10
    n_nodes = n_elem + 1
    beam_axis_vec = np.array([1.0, 1.0, 1.0])
    unit_beam_axis = beam_axis_vec / np.linalg.norm(beam_axis_vec)
    node_coords = np.array([i * (L / n_elem) * unit_beam_axis for i in range(n_nodes)])
    elements = []
    for i in range(n_elem):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    force_vec = np.array([P_load, -P_load, 0.0])
    assert np.isclose(np.dot(force_vec, beam_axis_vec), 0.0)
    nodal_loads = {n_nodes - 1: list(force_vec) + [0.0, 0.0, 0.0]}
    P_eff = np.linalg.norm(force_vec)
    delta_analytical = P_eff * L ** 3 / (3 * E * I)
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    tip_disp_vec = u[(n_nodes - 1) * 6:(n_nodes - 1) * 6 + 3]
    delta_computed = np.linalg.norm(tip_disp_vec)
    assert delta_computed < delta_analytical
    assert np.isclose(delta_computed, delta_analytical, rtol=0.01)

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
Suggested Test stages:
1. Zero loads -> All displacements and reactions should be zero.
2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
3. Double the loads -> Displacements and reactions should double (linearity check).
4. Negate the original loads -> Displacements and reactions should flip sign.
5. Assure that reactions (forces and moments) satisfy global static equilibrium"""
    L = 1.0
    node_coords = np.array([[0, 0, 0], [L, 0, 0], [L, L, 0], [0, L, 0], [0, 0, L], [L, 0, L], [L, L, L], [0, L, L]])
    n_nodes = len(node_coords)
    elem_nodes = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    props = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05}
    elements = []
    for (i, j) in elem_nodes:
        axis = node_coords[j] - node_coords[i]
        if np.allclose(np.abs(axis / np.linalg.norm(axis)), [0, 0, 1]):
            local_z = [1, 0, 0]
        else:
            local_z = [0, 0, 1]
        elements.append({'node_i': i, 'node_j': j, **props, 'local_z': local_z})
    boundary_conditions = {i: [1, 1, 1, 1, 1, 1] for i in range(4)}
    (u0, r0) = fcn(node_coords, elements, boundary_conditions, nodal_loads={})
    assert np.allclose(u0, 0.0)
    assert np.allclose(r0, 0.0)
    nodal_loads_1 = {5: [10000.0, 0, 0, 0, 5000.0, 0], 7: [0, -20000.0, 0, 1000.0, 0, 0]}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, nodal_loads_1)
    assert not np.allclose(u1, 0.0)
    assert not np.allclose(r1, 0.0)
    nodal_loads_2 = {k: [2 * val for val in v] for (k, v) in nodal_loads_1.items()}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, nodal_loads_2)
    assert np.allclose(u2, 2 * u1)
    assert np.allclose(r2, 2 * r1)
    nodal_loads_3 = {k: [-val for val in v] for (k, v) in nodal_loads_1.items()}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, nodal_loads_3)
    assert np.allclose(u3, -u1)
    assert np.allclose(r3, -r1)
    sum_F_applied = np.sum([v[:3] for v in nodal_loads_1.values()], axis=0)
    r1_reshaped = r1.reshape((n_nodes, 6))
    sum_F_reactions = np.sum(r1_reshaped[:, :3], axis=0)
    assert np.allclose(sum_F_applied + sum_F_reactions, 0.0, atol=1e-06)
    sum_M_applied_origin = np.zeros(3)
    for (node_idx, loads) in nodal_loads_1.items():
        pos = node_coords[node_idx]
        sum_M_applied_origin += np.array(loads[3:]) + np.cross(pos, loads[:3])
    sum_M_reactions_origin = np.zeros(3)
    for i in range(n_nodes):
        pos = node_coords[i]
        sum_M_reactions_origin += r1_reshaped[i, 3:] + np.cross(pos, r1_reshaped[i, :3])
    assert np.allclose(sum_M_applied_origin + sum_M_reactions_origin, 0.0, atol=1e-06)

def test_ill_conditioned_due_to_under_constrained_structure(fcn):
    """Test that solve_linear_elastic_frame_3d raises a ValueError
when the structure is improperly constrained, leading to an
ill-conditioned free-free stiffness matrix (K_ff).
The solver should detect this (cond(K_ff) >= 1e16) and raise a ValueError."""
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': [0.0, 0.0, 1.0]}]
    nodal_loads = {}
    boundary_conditions_1 = {}
    with pytest.raises(ValueError, match='ill-conditioned'):
        fcn(node_coords, elements, boundary_conditions_1, nodal_loads)
    boundary_conditions_2 = {0: [1, 0, 0, 0, 0, 0]}
    with pytest.raises(ValueError, match='ill-conditioned'):
        fcn(node_coords, elements, boundary_conditions_2, nodal_loads)