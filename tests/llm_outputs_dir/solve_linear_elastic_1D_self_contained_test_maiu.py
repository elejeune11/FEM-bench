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
    x_max = 1.0
    num_elements = 2
    material_regions = [{"coord_min": x_min, "coord_max": x_max, "E": 1e7, "A": 1.0}]
    body_force_fn = lambda x: 0.0
    dirichlet_bc_list = [{"x_location": x_min, "u_prescribed": 0.0}]
    neumann_bc_list = None
    n_gauss = 2

    result = fcn(x_min, x_max, num_elements, material_regions, body_force_fn,
                 dirichlet_bc_list, neumann_bc_list, n_gauss)

    displacements = result["displacements"]
    reactions = result["reactions"]
    node_coords = result["node_coords"]
    reaction_nodes = result["reaction_nodes"]

    # Check zero displacement at all nodes
    assert np.allclose(displacements, 0.0, atol=1e-12)

    # Check zero reaction at fixed node (since no load)
    assert np.allclose(reactions, 0.0, atol=1e-12)

    # Check output shapes
    n_nodes_expected = num_elements + 1
    n_dirichlet_expected = len(dirichlet_bc_list)
    assert displacements.shape == (n_nodes_expected,)
    assert reactions.shape == (n_dirichlet_expected,)
    assert node_coords.shape == (n_nodes_expected,)
    assert reaction_nodes.shape == (n_dirichlet_expected,)

    # Check that reaction nodes correspond to Dirichlet BC node indices
    for bc in dirichlet_bc_list:
        x_loc = bc["x_location"]
        idx = np.argmin(np.abs(node_coords - x_loc))
        assert idx in reaction_nodes


def test_uniform_extension_analytical_self_contained(fcn):
    """
    Test displacement field against a known analytical solution.
    For a bar with uniform material properties and uniform body force,
    fixed at one end and free at the other with zero Neumann BC,
    the displacement is quadratic in x and can be compared to the FE solution.
    """
    x_min = 0.0
    x_max = 1.0
    num_elements = 10
    E = 210e9  # Young's modulus in Pa
    A = 0.01   # Cross-sectional area in m^2

    material_regions = [{"coord_min": x_min, "coord_max": x_max, "E": E, "A": A}]

    # Uniform body force per unit length (N/m)
    q = 1000.0
    body_force_fn = lambda x: q

    # Dirichlet BC: fixed at x=0 (u=0)
    dirichlet_bc_list = [{"x_location": x_min, "u_prescribed": 0.0}]

    # Neumann BC: free end at x=1 (no traction)
    neumann_bc_list = None

    n_gauss = 2

    result = fcn(x_min, x_max, num_elements, material_regions, body_force_fn,
                 dirichlet_bc_list, neumann_bc_list, n_gauss)

    displacements = result["displacements"]
    node_coords = result["node_coords"]

    # Analytical solution for displacement u(x) under uniform load q:
    # u(x) = (q / (2 * E * A)) * x * (L - x)
    L = x_max - x_min
    u_exact = (q / (2 * E * A)) * node_coords * (L - node_coords)

    # Compute L2 norm of error between FE and exact solution
    error_vec = displacements - u_exact
    l2_error = np.sqrt(np.sum(error_vec**2) / len(error_vec))

    # Assert that error is small (tolerance depends on mesh size)
    assert l2_error < 1e-5

    # Check that displacement at fixed node matches prescribed value exactly
    idx_fixed = np.argmin(np.abs(node_coords - x_min))
    assert np.isclose(displacements[idx_fixed], 0.0, atol=1e-12)