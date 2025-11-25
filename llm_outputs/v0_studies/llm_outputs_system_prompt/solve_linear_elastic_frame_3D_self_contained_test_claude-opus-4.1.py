def test_simple_beam_discretized_axis_111(fcn):
    """Verification with resepct to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution."""
    import numpy as np
    L_total = 10.0
    n_elements = 10
    L_elem = L_total / n_elements
    E = 200000000000.0
    nu = 0.3
    d = 0.1
    A = np.pi * d ** 2 / 4
    I = np.pi * d ** 4 / 64
    direction = np.array([1, 1, 1]) / np.sqrt(3)
    node_coords = np.array([i * L_elem * direction for i in range(n_elements + 1)])
    elements = []
    for i in range(n_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': 2 * I, 'local_z': np.array([0, 0, 1])})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    F_magnitude = 1000.0
    force_direction = np.array([1, 1, -2]) / np.sqrt(6)
    nodal_loads = {n_elements: [F_magnitude * force_direction[0], F_magnitude * force_direction[1], F_magnitude * force_direction[2], 0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    delta_analytical = F_magnitude * L_total ** 3 / (3 * E * I)
    tip_node_idx = n_elements
    tip_disp = u[6 * tip_node_idx:6 * tip_node_idx + 3]
    tip_deflection_magnitude = np.linalg.norm(tip_disp)
    assert abs(tip_deflection_magnitude - delta_analytical) / delta_analytical < 0.05
    if tip_deflection_magnitude > 1e-10:
        disp_direction = tip_disp / tip_deflection_magnitude
        assert np.dot(disp_direction, force_direction) > 0.95

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Suggested Test stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium"""
    import numpy as np
    node_coords = np.array([[0, 0, 0], [2, 0, 0], [2, 3, 0], [2, 3, 4]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': None}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': None}, {'node_i': 2, 'node_j': 3, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': None}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1], 1: [0, 1, 1, 0, 0, 0]}
    nodal_loads = {}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u1, 0, atol=1e-12)
    assert np.allclose(r1, 0, atol=1e-12)
    nodal_loads = {2: [100, -50, 200, 10, -5, 15], 3: [-200, 100, -150, -20, 10, -25]}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert not np.allclose(u2, 0, atol=1e-10)
    assert not np.allclose(r2, 0, atol=1e-10)
    nodal_loads_double = {2: [200, -100, 400, 20, -10, 30], 3: [-400, 200, -300, -40, 20, -50]}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, nodal_loads_double)
    assert np.allclose(u3, 2 * u2, rtol=1e-10)
    assert np.allclose(r3, 2 * r2, rtol=1e-10)
    nodal_loads_neg = {2: [-100, 50, -200, -10, 5, -15], 3: [200, -100, 150, 20, -10, 25]}
    (u4, r4) = fcn(node_coords, elements, boundary_conditions, nodal_loads_neg)
    assert np.allclose(u4, -u2, rtol=1e-10)
    assert np.allclose(r4, -r2, rtol=1e-10)
    total_applied_force = np.zeros(3)
    total_applied_moment = np.zeros(3)
    for (node_idx, loads) in nodal_loads.items():
        total_applied_force += loads[:3]
        node_pos = node_coords[node_idx]
        total_applied_moment += loads[3:6] + np.cross(node_pos, loads[:3])
    total_reaction_force = np.zeros(3)
    total_reaction_moment = np.zeros(3)
    for node_idx in range(len(node_coords)):
        reaction = r2[6 * node_idx:6 * (node_idx + 1)]
        total_reaction_force += reaction[:3]
        node_pos = node_coords[node_idx]
        total_reaction_moment += reaction[3:6] + np.cross(node_pos, reaction[:3])
    assert np.allclose(total_applied_force + total_reaction_force, 0, atol=1e-08)
    assert np.allclose(total_applied_moment + total_reaction_moment, 0, atol=1e-08)

def test_ill_conditioned_due_to_under_constrained_structure(fcn):
    """Test that solve_linear_elastic_frame_3d raises a ValueError
    when the structure is improperly constrained, leading to an
    ill-conditioned free-free stiffness matrix (K_ff).
    The solver should detect this (cond(K_ff) >= 1e16) and raise a ValueError."""
    import numpy as np
    import pytest
    node_coords = np.array([[0, 0, 0], [1, 0, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': None}]
    boundary_conditions = {}
    nodal_loads = {1: [100, 0, 0, 0, 0, 0]}
    with pytest.raises(ValueError):
        fcn(node_coords, elements, boundary_conditions, nodal_loads)