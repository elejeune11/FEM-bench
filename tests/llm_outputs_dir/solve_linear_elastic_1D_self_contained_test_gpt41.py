import numpy as np

def test_no_load_self_contained(fcn):
    """
    Test zero displacement and zero reaction in a fixed-free bar with no external load.

    A 2-element bar with uniform material, zero body force, and one fixed end should return:
    - Zero displacement at all nodes
    - Zero reaction at the fixed node
    - Correct output shapes and boundary condition enforcement
    """
    x_min = 0.0
    x_max = 2.0
    num_elements = 2
    E = 200.0
    A = 1.0
    material_regions = [{"coord_min": x_min, "coord_max": x_max, "E": E, "A": A}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{"x_location": 0.0, "u_prescribed": 0.0}]
    neumann_bc_list = []
    n_gauss = 2

    result = fcn(
        x_min=x_min,
        x_max=x_max,
        num_elements=num_elements,
        material_regions=material_regions,
        body_force_fn=body_force_fn,
        dirichlet_bc_list=dirichlet_bc_list,
        neumann_bc_list=neumann_bc_list,
        n_gauss=n_gauss,
    )

    displ = result["displacements"]
    reactions = result["reactions"]
    node_coords = result["node_coords"]
    reaction_nodes = result["reaction_nodes"]

    assert isinstance(displ, np.ndarray)
    assert isinstance(reactions, np.ndarray)
    assert isinstance(node_coords, np.ndarray)
    assert isinstance(reaction_nodes, np.ndarray)

    assert displ.shape == node_coords.shape
    assert reactions.shape == (1,)
    assert reaction_nodes.shape == (1,)

    # All displacements should be zero
    assert np.allclose(displ, 0.0, atol=1e-10)
    # Reaction at fixed node should be zero
    assert np.allclose(reactions[0], 0.0, atol=1e-10)
    # Displacement at x=0 node is exactly 0
    idx_fixed = np.argmin(np.abs(node_coords - 0.0))
    assert np.isclose(displ[idx_fixed], 0.0, atol=1e-10)
    # The right node should not be fixed (not in reaction_nodes)
    assert np.max(node_coords) not in node_coords[reaction_nodes]


def test_uniform_extension_analytical_self_contained(fcn):
    """
    Test displacement field against a known analytical solution.
    """
    # Bar 0 <= x <= 4, uniform E and A, fixed at x=0, pulled at x=4.
    # Analytical: u(x) = Fx / (EA)
    x_min = 0.0
    x_max = 4.0
    num_elements = 4
    E = 100.0
    A = 2.0
    F = 50.0
    material_regions = [{"coord_min": x_min, "coord_max": x_max, "E": E, "A": A}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{"x_location": 0.0, "u_prescribed": 0.0}]
    neumann_bc_list = [{"x_location": x_max, "load_mag": F}]
    n_gauss = 2

    result = fcn(
        x_min=x_min,
        x_max=x_max,
        num_elements=num_elements,
        material_regions=material_regions,
        body_force_fn=body_force_fn,
        dirichlet_bc_list=dirichlet_bc_list,
        neumann_bc_list=neumann_bc_list,
        n_gauss=n_gauss,
    )

    node_coords = result["node_coords"]
    displ = result["displacements"]
    reactions = result["reactions"]
    reaction_nodes = result["reaction_nodes"]

    assert node_coords.shape == displ.shape
    # The left node is fixed, rightmost is at x=4
    u_true = F * node_coords / (E * A)
    assert np.allclose(displ, u_true, atol=1e-8)
    # Reaction at fixed node should be -F (left), since right side is loaded by F
    assert reactions.shape == (1,)
    assert np.isclose(reactions[0], -F, atol=1e-8)
    # Node at x=0 displacement is zero
    idx_fixed = np.argmin(np.abs(node_coords - 0.0))
    assert np.isclose(displ[idx_fixed], 0.0, atol=1e-10)
    # Node at x=4 has max displacement
    idx_right = np.argmax(node_coords)
    assert np.isclose(displ[idx_right], F*x_max/(E*A), atol=1e-8)