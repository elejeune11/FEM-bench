def test_no_load_self_contained(fcn):
    """Test zero displacement and zero reaction in a fixed-free bar with no external load.

    A 2-element bar with uniform material, zero body force, and one fixed end should return:
    - Zero displacement at all nodes
    - Zero reaction at the fixed node
    - Correct output shapes and boundary condition enforcement
    """
    import numpy as np

    # Setup: 2-element bar from 0 to 1
    x_min = 0.0
    x_max = 1.0
    num_elements = 2

    # Uniform material properties
    material_regions = [
        {"coord_min": 0.0, "coord_max": 1.0, "E": 1.0, "A": 1.0}
    ]

    # Zero body force
    def body_force_fn(x):
        return 0.0

    # Fixed at left end (x=0)
    dirichlet_bc_list = [
        {"x_location": 0.0, "u_prescribed": 0.0}
    ]

    # No applied loads
    neumann_bc_list = None

    # Single Gauss point
    n_gauss = 1

    # Solve
    result = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, 
                dirichlet_bc_list, neumann_bc_list, n_gauss)

    # Check output structure
    assert "displacements" in result
    assert "reactions" in result
    assert "node_coords" in result
    assert "reaction_nodes" in result

    # Check shapes
    n_nodes = num_elements + 1  # 3 nodes for 2 elements
    assert result["displacements"].shape == (n_nodes,)
    assert result["reactions"].shape == (1,)  # One Dirichlet BC
    assert result["node_coords"].shape == (n_nodes,)
    assert result["reaction_nodes"].shape == (1,)

    # Check zero displacement everywhere
    assert np.allclose(result["displacements"], 0.0)

    # Check zero reaction at fixed node
    assert np.allclose(result["reactions"], 0.0)

    # Check boundary condition enforcement
    assert result["reaction_nodes"][0] == 0  # Fixed node is at index 0
    assert result["displacements"][0] == 0.0  # Displacement at fixed node is zero


def test_uniform_extension_analytical_self_contained(fcn):
    """Test displacement field against a known analytical solution."""
    import numpy as np

    # Setup: Simple bar under uniform body force
    x_min = 0.0
    x_max = 1.0
    num_elements = 4

    # Uniform material properties
    E = 2.0
    A = 1.0
    material_regions = [
        {"coord_min": 0.0, "coord_max": 1.0, "E": E, "A": A}
    ]

    # Constant body force
    f = 1.0
    def body_force_fn(x):
        return f

    # Fixed at left end (x=0)
    dirichlet_bc_list = [
        {"x_location": 0.0, "u_prescribed": 0.0}
    ]

    # No applied loads
    neumann_bc_list = None

    # Two Gauss points for better accuracy
    n_gauss = 2

    # Solve
    result = fcn(x_min, x_max, num_elements, material_regions, body_force_fn, 
                dirichlet_bc_list, neumann_bc_list, n_gauss)

    # Analytical solution for u(x) = f*x*(L-x)/(2*E*A) for fixed-free bar
    # with uniform body force f, where L is the length
    L = x_max - x_min
    node_coords = result["node_coords"]

    # Analytical displacement at each node
    u_analytical = f * node_coords * (L - node_coords) / (2 * E * A)

    # Check displacement field matches analytical solution
    assert np.allclose(result["displacements"], u_analytical, rtol=1e-10)

    # Check reaction force at fixed end
    # For uniform body force, reaction = -f*L*A
    expected_reaction = -f * L * A
    assert np.allclose(result["reactions"], expected_reaction, rtol=1e-10)

    # Check boundary condition enforcement
    assert result["displacements"][0] == 0.0  # Fixed end displacement is zero