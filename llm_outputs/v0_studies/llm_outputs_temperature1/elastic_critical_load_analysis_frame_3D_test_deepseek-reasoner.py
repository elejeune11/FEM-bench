def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever value analytical solution.
    Use 10 elements, set tolerances to be appropriate for the anticipated discretization error at 10-5.
    """
    import numpy as np
    import math
    E = 200000000000.0
    nu = 0.3
    radii = [0.5, 0.75, 1.0]
    lengths = [10, 20, 40]
    n_elements = 10
    for r in radii:
        for L in lengths:
            A = math.pi * r ** 2
            I = math.pi * r ** 4 / 4
            J = 2 * I
            I_rho = I + I
            node_coords = np.zeros((n_elements + 1, 3))
            node_coords[:, 2] = np.linspace(0, L, n_elements + 1)
            elements = []
            for i in range(n_elements):
                ele = {'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'I_rho': I_rho, 'local_z': None}
                elements.append(ele)
            boundary_conditions = {0: [True, True, True, True, True, True]}
            tip_node = n_elements
            nodal_loads = {tip_node: [0, 0, -1, 0, 0, 0]}
            (critical_load_factor, _) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            P_euler = math.pi ** 2 * E * I / (4 * L ** 2)
            rel_error = abs(critical_load_factor - P_euler) / P_euler
            assert rel_error < 1e-05, f'Relative error {rel_error} too large for r={r}, L={L}'

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    The cantilever model is solved in its original orientation and again after applying
    a rigid-body rotation R to the geometry, element axes, and applied load. The critical
    load factor λ should be identical in both cases.
    The buckling mode from the rotated model should equal the base mode transformed by R:
      [ux, uy, uz] and rotational [θx, θy, θz] DOFs at each node.
    """
    import numpy as np
    import math
    E = 200000000000.0
    nu = 0.3
    L = 10
    n_elements = 8
    (b, h) = (0.5, 1.0)
    A = b * h
    I_y = b * h ** 3 / 12
    I_z = h * b ** 3 / 12
    J = (b * h ** 3 + h * b ** 3) / 12
    I_rho = I_y + I_z
    node_coords_base = np.zeros((n_elements + 1, 3))
    node_coords_base[:, 2] = np.linspace(0, L, n_elements + 1)
    elements_base = []