import numpy as np
from typing import Optional, Sequence


def solve_linear_elastic_frame_3D(
    node_coords: np.ndarray,
    elements: Sequence[dict],
    boundary_conditions: dict[int, Sequence[int]],
    nodal_loads: dict[int, Sequence[float]],
):
    """
    Small-displacement linear-elastic analysis of a 3D frame. Coordinate system follows the right hand rule.
    The condition number of the global stiffness matrix should be checked before solving.
    If the problem is ill-posed based on condition number, return a (6 N, ) zero array for both u and r.

    Parameters
    ----------
    node_coords : (N, 3) float ndarray
        Global coordinates of the N nodes (row i → [x, y, z] of node i, 0-based).
    elements : iterable of dict
        Each dict must contain
            'node_i', 'node_j' : int # end node indices (0-based)
            'E', 'nu', 'A', 'I_y', 'I_z', 'J' : float
            'local_z' : (3,) array | None # optional unit vector for local z
    boundary_conditions : dict[int, Sequence[int]]
        node index → 6-element 0/1 iterable (0 = free, 1 = fixed).
        Omitted nodes ⇒ all DOFs free.
    nodal_loads : dict[int, Sequence[float]]
        node index → 6-element [Fx, Fy, Fz, Mx, My, Mz] (forces (+) and moments).
        Omitted nodes ⇒ zero loads.

    Returns:
    u : (6 N,) ndarray
        Global displacement vector (UX, UY, UZ, RX, RY, RZ for each node in order).
    r : (6 N,) ndarray
        Global force/moment vector with support reactions filled in fixed DOFs.
    """
    def _kel(E, nu, A, L, Iy, Iz, J):
        k = np.zeros((12, 12))
        EA_L = E * A / L
        GJ_L = E * J / (2.0 * (1.0 + nu) * L)
        EIz_L = E * Iz
        EIy_L = E * Iy
        # axial
        k[0, 0] = k[6, 6] = EA_L
        k[0, 6] = k[6, 0] = -EA_L
        # torsion
        k[3, 3] = k[9, 9] = GJ_L
        k[3, 9] = k[9, 3] = -GJ_L
        # bending about z (local y‑displacements & rotations about z)
        k[1, 1] = k[7, 7] = 12.0 * EIz_L / L**3
        k[1, 7] = k[7, 1] = -12.0 * EIz_L / L**3
        k[1, 5] = k[5, 1] = k[1, 11] = k[11, 1] = 6.0 * EIz_L / L**2
        k[5, 7] = k[7, 5] = k[7, 11] = k[11, 7] = -6.0 * EIz_L / L**2
        k[5, 5] = k[11, 11] = 4.0 * EIz_L / L
        k[5, 11] = k[11, 5] = 2.0 * EIz_L / L
        # bending about y (local z‑displacements & rotations about y)
        k[2, 2] = k[8, 8] = 12.0 * EIy_L / L**3
        k[2, 8] = k[8, 2] = -12.0 * EIy_L / L**3
        k[2, 4] = k[4, 2] = k[2, 10] = k[10, 2] = -6.0 * EIy_L / L**2
        k[4, 8] = k[8, 4] = k[8, 10] = k[10, 8] = 6.0 * EIy_L / L**2
        k[4, 4] = k[10, 10] = 4.0 * EIy_L / L
        k[4, 10] = k[10, 4] = 2.0 * EIy_L / L
        return k
    
    def _gamma(x1, y1, z1, x2, y2, z2, ref_vec: Optional[np.ndarray]):
        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
        L = np.sqrt(dx*dx + dy*dy + dz*dz)
        if np.isclose(L, 0.0):
            raise ValueError("Beam length is zero.")
        ex = np.array([dx, dy, dz]) / L
        if ref_vec is None:
            ref_vec = np.array([0.0, 0.0, 1.0]) if not (np.isclose(ex[0], 0) and np.isclose(ex[1], 0)) \
                      else np.array([0.0, 1.0, 0.0])
        else:
            ref_vec = np.asarray(ref_vec, dtype=float)
            if ref_vec.shape != (3,):
                raise ValueError("local_z/reference_vector must be length‑3.")
            if not np.isclose(np.linalg.norm(ref_vec), 1.0):
                raise ValueError("reference_vector must be unit length.")
            if np.isclose(np.linalg.norm(np.cross(ref_vec, ex)), 0.0):
                raise ValueError("reference_vector parallel to beam axis.")

        ey = np.cross(ref_vec, ex)
        ey /= np.linalg.norm(ey)
        ez = np.cross(ex, ey)

        gamma = np.vstack((ex, ey, ez))  # 3×3
        return np.kron(np.eye(4), gamma)  # 12×12

    # Dimensions
    n_nodes = node_coords.shape[0]
    n_dof = 6 * n_nodes

    # DOF map
    def _node_dofs(n):  # 6 global DOFs for node n
        return list(range(6*n, 6*n + 6))

    # Assemble global stiffness K
    K = np.zeros((n_dof, n_dof))

    for ele in elements:
        ni, nj = int(ele['node_i']), int(ele['node_j'])
        xi, yi, zi = node_coords[ni]
        xj, yj, zj = node_coords[nj]

        L = np.linalg.norm([xj - xi, yj - yi, zj - zi])
        Gamma = _gamma(xi, yi, zi, xj, yj, zj, ele.get('local_z'))

        k_loc = _kel(ele['E'], ele['nu'], ele['A'], L,
                     ele['I_y'], ele['I_z'], ele['J'])
        k_glb = Gamma.T @ k_loc @ Gamma

        dofs = _node_dofs(ni) + _node_dofs(nj)
        K[np.ix_(dofs, dofs)] += k_glb

    # Assemble global load vector P
    P = np.zeros(n_dof)
    for n, load in nodal_loads.items():
        P[_node_dofs(n)] += np.asarray(load, dtype=float)

    # Free / fixed DOFs
    fixed = []
    for n in range(n_nodes):
        flags = boundary_conditions.get(n)
        if flags is not None:
            fixed.extend([6*n + i for i, f in enumerate(flags) if f])
    fixed = np.asarray(fixed, dtype=int)
    free = np.setdiff1d(np.arange(n_dof), fixed, assume_unique=True)

    # Solve
    K_ff = K[np.ix_(free,  free)]
    K_sf = K[np.ix_(fixed, free)]
    condition_number = np.linalg.cond(K_ff)
    if condition_number < 10 ** 16:
        u_f = np.linalg.solve(K_ff, P[free])
        u = np.zeros(n_dof)
        u[free] = u_f
        r = P.copy()
        r[fixed] = K_sf @ u_f
    else:
        u = np.zeros(n_dof)
        r = np.zeros(n_dof)
    return u, r


def solve_linear_elastic_frame_3D_all_zeros(
    node_coords: np.ndarray,
    elements: Sequence[dict],
    boundary_conditions: dict[int, Sequence[int]],
    nodal_loads: dict[int, Sequence[float]],
):
    """
    Dummy implementation - returns displacement and reaction vectors that are
    identically zero, regardless of the input.
    """
    n_dof = 6 * node_coords.shape[0]       # 6 DOF per node
    u = np.zeros(n_dof)
    r = np.zeros(n_dof)
    return u, r


def solve_linear_elastic_frame_3D_all_ones(
    node_coords: np.ndarray,
    elements: Sequence[dict],
    boundary_conditions: dict[int, Sequence[int]],
    nodal_loads: dict[int, Sequence[float]],
):
    """
    Dummy implementation - returns displacement and reaction vectors filled
    with ones, regardless of the input.
    """
    n_dof = 6 * node_coords.shape[0]
    u = np.ones(n_dof)
    r = np.ones(n_dof)
    return u, r


def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with resepct to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution.
    """
    # Geometry: 11 nodes along the [1,1,1] direction
    L_total = 1.0
    n_elems = 10
    n_nodes = n_elems + 1
    axis_unit = np.array([1.0, 1.0, 1.0])
    axis_unit /= np.linalg.norm(axis_unit)           # unit vector (≈ [0.577,0.577,0.577])
    node_coords = np.array([i * (L_total / n_elems) * axis_unit for i in range(n_nodes)])

    # Material & section (solid circular, I_y = I_z)
    E = 210e9
    r = 0.02
    A = np.pi * r**2
    I = np.pi * r**4 / 4.0
    J = 2.0 * I

    elements = [dict(node_i=e, node_j=e + 1,
                     E=E, nu=0.3,
                     A=A, I_y=I, I_z=I, J=J,
                     local_z=None)
                for e in range(n_elems)]

    # Boundary conditions & load (transverse, orthogonal to axis)
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}

    load_dir = np.array([1.0, -1.0, 0.0])
    load_dir /= np.linalg.norm(load_dir)
    P_load = 1_000.0

    nodal_loads = {n_nodes - 1: list(P_load * load_dir) + [0.0, 0.0, 0.0]}

    # Solve
    u, r = fcn(
        node_coords=node_coords,
        elements=elements,
        boundary_conditions=boundary_conditions,
        nodal_loads=nodal_loads,
    )

    # Numerical tip deflection component along load direction
    tip_node = n_nodes - 1
    disp_tip = u[6*tip_node : 6*tip_node + 3]        # [ux, uy, uz]
    delta_num = np.dot(disp_tip, load_dir)           # scalar projection onto load_dir

    # Analytical deflection (cantilever, point load, circular section)
    delta_exact = P_load * L_total**3 / (3.0 * E * I)

    # Assertion: within 2 %
    assert np.isclose(delta_num, delta_exact, rtol=0.02), (
        f"Tip deflection {delta_num:.6e} m differs from analytical "
        f"{delta_exact:.6e} m by more than 2 %")

    # Optional: check displacement component along beam axis is ~0
    axis_disp = np.dot(disp_tip, axis_unit)
    assert abs(axis_disp) < 1e-6


def test_complex_geometry_and_basic_loading(fcn):
    """
    Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Suggested Test stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium
    """
    # Node coordinates (tetrahedron, metres)
    node_coords = np.array([
        [0.0, 0.0, 0.0],   # Node 0 (fixed)
        [1.0, 0.0, 1.0],   # Node 1
        [0.0, 1.0, 1.0],   # Node 2
        [1.0, 1.0, 0.0],   # Node 3
    ])

    # Six frame elements connect the nodes
    E, nu = 210e9, 0.3
    r = 0.02
    A = np.pi * r**2
    I = np.pi * r**4 / 4
    J = 2.0 * I

    conn = [(0, 1), (0, 2), (0, 3),
            (1, 2), (1, 3), (2, 3)]
    elements = [dict(node_i=i, node_j=j,
                     E=E, nu=nu, A=A,
                     I_y=I, I_z=I, J=J,
                     local_z=None)
                for i, j in conn]

    # Boundary conditions – node 0 fully fixed
    bcs = {0: [1, 1, 1, 1, 1, 1]}

    # Stage 1 – zero loads
    u0, r0 = fcn(node_coords, elements, bcs, {})
    assert np.allclose(u0, 0.0, atol=1e-12)
    assert np.allclose(r0, 0.0, atol=1e-12)

    # Stage 2 – mixed forces + moments
    loads_1 = {
        1: [2500.0, -1000.0, 800.0, 50.0, 10.0, -25.0],
        2: [-3000.0, 2200.0, 1100.0, -40.0, 20.0, 30.0],
        3: [0.0, -1500.0, 50.0, 75.0, -60.0, 0.0],
    }
    u1, r1 = fcn(node_coords, elements, bcs, loads_1)
    assert np.linalg.norm(u1) > 0.0
    assert np.linalg.norm(r1) > 0.0

    # Stage 3 – double the loads (linearity check)
    loads_2 = {n: 2.0 * np.asarray(v) for n, v in loads_1.items()}
    u2, r2 = fcn(node_coords, elements, bcs, loads_2)
    assert np.allclose(u2, 2.0 * u1, rtol=1e-12, atol=1e-12)
    assert np.allclose(r2, 2.0 * r1, rtol=1e-12, atol=1e-12)

    # Stage 4 – negate the original loads (response should flip sign)
    loads_3 = {n: -np.asarray(v) for n, v in loads_1.items()}
    u3, r3 = fcn(node_coords, elements, bcs, loads_3)
    assert np.allclose(u3, -u1, rtol=1e-12, atol=1e-12), \
        "Displacements did not reverse sign with negated loads."
    assert np.allclose(r3, -r1, rtol=1e-12, atol=1e-12), \
        "Reactions did not reverse sign with negated loads."

    # Additional checks
    # Fixed node remains immovable
    assert np.allclose(u1[:6], 0.0, atol=1e-12)
    assert np.allclose(u2[:6], 0.0, atol=1e-12)
    assert np.allclose(u3[:6], 0.0, atol=1e-12)

    # Static equilibrium for Stage 2 (forces + moments)
    pos = node_coords - node_coords[0]
    total_F = np.zeros(3)
    total_M = np.zeros(3)
    for n, load in loads_1.items():
        F = np.asarray(load[:3])
        M = np.asarray(load[3:])
        total_F += F
        total_M += M + np.cross(pos[n], F)

    assert np.allclose(total_F + r1[:3], 0.0, atol=1e-8)
    assert np.allclose(total_M + r1[3:6], 0.0, atol=1e-6)


def test_condition_number(fcn):
    """
    Test the effect of boundary conditions on the stiffness matrix condition number.
    - Case 1: No boundary conditions → ill-posed system → zero solution expected.
    - Case 2: Fully fixed node → well-posed system → nonzero response expected.
    """

    node_coords = np.array([
        [0.0, 0.0, 0.0],  # Node 0
        [1.0, 0.0, 0.0],  # Node 1
    ])

    elements = [{
        'node_i': 0,
        'node_j': 1,
        'E': 210e9,
        'nu': 0.3,
        'A': 0.01,
        'I_y': 1e-6,
        'I_z': 1e-6,
        'J': 2e-6,
        'local_z': None
    }]

    nodal_loads = {
        1: [1000.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Force in global X at free end
    }

    # Case 1: No boundary conditions (should trigger ill-conditioning)
    bc_empty = {}

    u1, r1 = fcn(node_coords, elements, bc_empty, nodal_loads)

    assert np.allclose(u1, 0.0, atol=1e-12), "Expected zero displacements for ill-posed system"
    assert np.allclose(r1, 0.0, atol=1e-12), "Expected zero reactions for ill-posed system"

    # Case 2: Fully fixed node 0 (well-posed system)
    bc_fixed = {0: [1, 1, 1, 1, 1, 1]}

    u2, r2 = fcn(node_coords, elements, bc_fixed, nodal_loads)

    assert not np.allclose(u2, 0.0, atol=1e-12), "Expected non-zero displacements for well-posed system"
    assert not np.allclose(r2, 0.0, atol=1e-12), "Expected non-zero reactions for well-posed system"


def task_info():
    task_id = "solve_linear_elastic_frame_3D"
    task_short_description = "Self-contained function to solve a 3D linear elastic MSA problem"
    created_date = "2025-08-07"
    created_by = "elejeune11"
    main_fcn = solve_linear_elastic_frame_3D
    required_imports = [
        "import numpy as np",
        "from typing import Optional, Sequence",
        "import pytest"
    ]
    fcn_dependencies = []

    reference_verification_inputs = [

        # 1 ─ Skew cantilever ([1,1,1] beam with Z-shear)
        [
            np.array([[0.0, 0.0, 0.0]] + [0.15 * np.array([1, 1, 1]) * i for i in range(1, 10)]),
            [dict(node_i=i, node_j=i+1, E=210e9, nu=0.3, A=5e-4, I_y=1e-6, I_z=1e-6, J=2e-6, local_z=None)
             for i in range(9)],
            {0: [1, 1, 1, 1, 1, 1]},
            {9: [0.0, 0.0, -200.0, 0.0, 0.0, 0.0]}
        ],

        # 2 ─ Graded modulus beam
        [
            np.array([[i, 0.0, 0.0] for i in range(8)]),
            [dict(node_i=i, node_j=i+1, E=E_i, nu=0.3, A=1e-4, I_y=1e-8, I_z=1e-8, J=2e-8, local_z=None)
             for i, E_i in enumerate([70e9, 100e9, 140e9, 180e9, 220e9, 260e9, 300e9])],
            {0: [1, 1, 1, 1, 1, 1]},
            {7: [0.0, -400.0, 0.0, 0.0, 0.0, 0.0]}
        ],

        # 3 ─ Portal frame with mixed BCs and loads
        [
            np.array([
                [0, 0, 0],
                [0, 0, 3],
                [4, 0, 3],
                [4, 0, 0],
                [2, 0, 3],
                [2, 0, 0]
            ]),
            [
                *[dict(node_i=a, node_j=b, E=210e9, nu=0.3, A=6e-4, I_y=2e-6, I_z=2e-6, J=4e-6, local_z=None)
                  for a, b in [(0, 1), (3, 2), (1, 2), (0, 4), (3, 4)]],
                *[dict(node_i=a, node_j=b, E=210e9, nu=0.3, A=3e-4, I_y=8e-7, I_z=8e-7, J=1.6e-6, local_z=None)
                  for a, b in [(1, 5), (5, 3), (0, 5), (5, 2), (4, 5)]]
            ],
            {
                0: [1, 1, 1, 1, 1, 1],
                3: [1, 1, 1, 0, 0, 0]
            },
            {
                2: [20e3, 0.0, -50e3, 0, 0, 0],
                4: [0, 0, -10e3, 0, 0, 0]
            }
        ],

        # 4 ─ Cuboidal torsion loop
        [
            np.array([
                [0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0],
                [0, 0, 2], [2, 0, 2], [2, 2, 2], [0, 2, 2],
                [1, 1, 0], [1, 1, 2]
            ]),
            [dict(node_i=i, node_j=j, E=200e9, nu=0.3, A=1e-4, I_y=3e-8, I_z=3e-8, J=6e-8, local_z=None)
             for i, j in [(0, 1), (1, 2), (2, 3), (3, 0),
                          (4, 5), (5, 6), (6, 7), (7, 4),
                          (8, 9), (0, 4)]],
            {0: [1, 1, 1, 1, 1, 1]},
            {6: [0, 0, 0, 0, 0, 500.0]}
        ],

        # 5 ─ Free-floating beam (ill-conditioned)
        [
            np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]]),
            [dict(node_i=i, node_j=i+1, E=210e9, nu=0.3, A=5e-4, I_y=1e-6, I_z=1e-6, J=2e-6, local_z=None)
             for i in range(4)],
            {},
            {4: [0, 0, -100.0, 0, 0, 0]}
        ]
    ]

    test_cases = [
        {
            "test_code": test_simple_beam_discretized_axis_111,
            "expected_failures": [solve_linear_elastic_frame_3D_all_zeros, solve_linear_elastic_frame_3D_all_ones]
        },
        {
            "test_code": test_complex_geometry_and_basic_loading,
            "expected_failures": [solve_linear_elastic_frame_3D_all_zeros, solve_linear_elastic_frame_3D_all_ones]
        },
        {
            "test_code": test_condition_number,
            "expected_failures": [solve_linear_elastic_frame_3D_all_zeros, solve_linear_elastic_frame_3D_all_ones]
        }
    ]

    return (
        task_id,
        task_short_description,
        created_date,
        created_by,
        main_fcn,
        required_imports,
        fcn_dependencies,
        reference_verification_inputs,
        test_cases
    )
