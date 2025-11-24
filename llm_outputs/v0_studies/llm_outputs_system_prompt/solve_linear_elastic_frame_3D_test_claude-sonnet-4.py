def test_simple_beam_discretized_axis_111(fcn):
    """Verification with respect to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution."""
    import numpy as np
    L = 10.0
    n_elements = 10
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    I_y = I_z = 8.333e-06
    J = 1.667e-05
    direction = np.array([1, 1, 1]) / np.sqrt(3)
    n_nodes = n_elements + 1
    node_coords = np.zeros((n_nodes, 3))
    for i in range(n_nodes):
        node_coords[i] = i * L / n_elements * direction
    elements = []
    for i in range(n_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    F = 1000.0
    nodal_loads = {n_nodes - 1: [0, F, 0, 0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    tip_disp_y = u[6 * (n_nodes - 1) + 1]
    analytical_deflection = F * L ** 3 / (3 * E * I_z)
    assert abs(tip_disp_y - analytical_deflection) / abs(analytical_deflection) < 0.05
    assert np.allclose(u[:6], 0.0, atol=1e-12)
    reaction_y = r[1]
    assert abs(reaction_y + F) < 1e-06

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Suggested Test stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium"""
    import numpy as np
    node_coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [2.0, 2.0, 3.0]])
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    I_y = I_z = 8.333e-06
    J = 1.667e-05
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}, {'node_i': 2, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': None}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads_zero = {}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, nodal_loads_zero)
    assert np.allclose(u1, 0.0, atol=1e-12)
    assert np.allclose(r1, 0.0, atol=1e-12)
    nodal_loads_mixed = {1: [100.0, 0.0, 0.0, 0.0, 0.0, 500.0], 2: [0.0, 200.0, 0.0, 300.0, 0.0, 0.0], 3: [0.0, 0.0, 150.0, 0.0, 400.0, 0.0]}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, nodal_loads_mixed)
    assert not np.allclose(u2, 0.0, atol=1e-12)
    assert not np.allclose(r2, 0.0, atol=1e-12)
    nodal_loads_double = {1: [200.0, 0.0, 0.0, 0.0, 0.0, 1000.0], 2: [0.0, 400.0, 0.0, 600.0, 0.0, 0.0], 3: [0.0, 0.0, 300.0, 0.0, 800.0, 0.0]}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, nodal_loads_double)
    assert np.allclose(u3, 2.0 * u2, rtol=1e-12)
    assert np.allclose(r3, 2.0 * r2, rtol=1e-12)
    nodal_loads_negative = {1: [-100.0, 0.0, 0.0, 0.0, 0.0, -500.0], 2: [0.0, -200.0, 0.0, -300.0, 0.0, 0.0], 3: [0.0, 0.0, -150.0, 0.0, -400.0, 0.0]}
    (u4, r4) = fcn(node_coords, elements, boundary_conditions, nodal_loads_negative)
    assert np.allclose(u4, -u2, rtol=1e-12)
    assert np.allclose(r4, -r2, rtol=1e-12)
    total_applied_forces = np.array([100.0, 200.0, 150.0])
    total_applied_moments = np.array([300.0, 400.0, 500.0])
    reaction_forces = r2[:3]
    reaction_moments = r2[3:6]
    assert np.allclose(total_applied_forces + reaction_forces, 0.0, atol=1e-06)
    total_moment_from_forces = np.cross(node_coords[1], [100.0, 0.0, 0.0]) + np.cross(node_coords[2], [0.0, 200.0, 0.0]) + np.cross(node_coords[3], [0.0, 0.0, 150.0])
    total_applied_moments_global = total_applied_moments + total_moment_from_forces
    assert np.allclose(total_applied_moments_global + reaction_moments, 0.0, atol=0.001)