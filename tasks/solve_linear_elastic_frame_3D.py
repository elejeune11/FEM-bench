import numpy as np
from typing import Optional, Sequence


def local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J):
    """
    Return the 12x12 local elastic stiffness matrix for a 3D Euler-Bernoulli beam element.

    The beam is assumed to be aligned with the local x-axis. The stiffness matrix
    relates local nodal displacements and rotations to forces and moments using the equation:

        [force_vector] = [stiffness_matrix] @ [displacement_vector]

    Degrees of freedom are ordered as:
        [u1, v1, w1, θx1, θy1, θz1, u2, v2, w2, θx2, θy2, θz2]

    Where:
        - u, v, w: displacements along local x, y, z
        - θx, θy, θz: rotations about local x, y, z
        - Subscripts 1 and 2 refer to node i and node j of the element

    Parameters:
        E (float): Young's modulus
        nu (float): Poisson's ratio (used for torsion only)
        A (float): Cross-sectional area
        L (float): Length of the beam element
        Iy (float): Second moment of area about the local y-axis
        Iz (float): Second moment of area about the local z-axis
        J (float): Torsional constant

    Returns:
        np.ndarray: A 12×12 symmetric stiffness matrix representing axial, torsional,
                    and bending stiffness in local coordinates.
    """
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


def beam_transformation_matrix_3D(x1, y1, z1, x2, y2, z2, ref_vec: Optional[np.ndarray]):
    """
    Compute the 12x12 transformation matrix Gamma for a 3D beam element.

    This transformation relates the element's local coordinate system to the global system:
        K_global = Gamma.T @ K_local @ Gamma
    where K_global is the global stiffness matrix and K_local is the local stiffness matrix.

    Parameters:
        x1, y1, z1 (float): Coordinates of the beam's start node in global space.
        x2, y2, z2 (float): Coordinates of the beam's end node in global space.
        reference_vector (np.ndarray of shape (3,), optional): A unit vector in global coordinates used to define
            the orientation of the local y-axis. The local y-axis is computed as the cross product
            of the reference vector and the local x-axis (beam axis). The local z-axis is then
            computed as the cross product of the local x-axis and the local y-axes.

            If not provided:
            - If the beam is aligned with the global z-axis, the global y-axis is used.
            - Otherwise, the global z-axis is used.

    Returns:
        Gamma (np.ndarray): A 12x12 local-to-global transformation matrix used to transform
            stiffness matrices, displacements, and forces. It is composed of four repeated
            3x3 direction cosine submatrices along the diagonal.

    Raises:
        ValueError: If `reference_vector` is not a unit vector.
        ValueError: If `reference_vector` is parallel to the beam axis.
        ValueError: If the `reference_vector` doesn't have shape (3,).
        ValueError: If the beam has zero length (start and end nodes coincide).

    Notes:
        All vectors must be specified in a right-handed global Cartesian coordinate system.
    """
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


def assemble_global_stiffness_matrix_linear_elastic_3D(node_coords, elements):
    """
    Assembles the global stiffness matrix for a 3D linear elastic frame structure composed of beam elements.

    Each beam element connects two nodes and contributes a 12x12 stiffness matrix (6 DOFs per node) to the
    global stiffness matrix. The element stiffness is computed in the local coordinate system using material
    and geometric properties, then transformed into the global coordinate system via a transformation matrix.

    Parameters
    ----------
    node_coords : ndarray of shape (n_nodes, 3)
        Array containing the (x, y, z) coordinates of each node.

    elements : list of dict
        A list of element dictionaries. Each dictionary must contain:
            - 'node_i', 'node_j' : int
                Indices of the start and end nodes.
            - 'E' : float
                Young's modulus of the element.
            - 'nu' : float
                Poisson's ratio of the element.
            - 'A' : float
                Cross-sectional area.
            - 'I_y', 'I_z' : float
                Second moments of area about local y and z axes.
            - 'J' : float
                Torsional constant.
            - 'local_z' (optional) : array-like
                Optional vector defining the local z-direction to resolve transformation ambiguity.

    Returns
    -------
    K : ndarray of shape (6 * n_nodes, 6 * n_nodes)
        The assembled global stiffness matrix of the structure, with 6 degrees of freedom per node.

    Helper Functions Required
    -------------------------
    - local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, I_y, I_z, J)
        Returns the 12x12 local stiffness matrix for a 3D beam element.

    - beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z=None)
        Returns the 12x12 transformation matrix to convert local stiffness to global coordinates.

    Notes
    -----
    - Degrees of freedom per node follow the order: [u_x, u_y, u_z, θ_x, θ_y, θ_z].
    - Assumes all elements are linearly elastic and connected via shared nodes.
    """
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
        Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, ele.get('local_z'))

        k_loc = local_elastic_stiffness_matrix_3D_beam(ele['E'], ele['nu'], ele['A'], L,
                     ele['I_y'], ele['I_z'], ele['J'])
        k_glb = Gamma.T @ k_loc @ Gamma

        dofs = _node_dofs(ni) + _node_dofs(nj)
        K[np.ix_(dofs, dofs)] += k_glb

    return K


def assemble_global_load_vector_linear_elastic_3D(nodal_loads, n_nodes):
    """
    Assembles the global load vector for a 3D linear elastic frame structure with 6 degrees of freedom per node.

    Each node can have a 6-component load vector corresponding to forces and moments in the x, y, z directions:
    [F_x, F_y, F_z, M_x, M_y, M_z]. The function maps these nodal loads into a global load vector with size
    6 * n_nodes.

    Parameters
    ----------
    nodal_loads : dict[int, array-like]
        Dictionary mapping node indices to applied loads.
        Each load must be an iterable of 6 components: [F_x, F_y, F_z, M_x, M_y, M_z].

    n_nodes : int
        Total number of nodes in the structure.

    Returns
    -------
    P : ndarray of shape (6 * n_nodes,)
        Global load vector containing all applied nodal loads, ordered by node index and degree of freedom.

    Notes
    -----
    - The function assumes 6 DOFs per node in the order:
      [u_x, u_y, u_z, θ_x, θ_y, θ_z].
    - Nodes without specified loads in `nodal_loads` contribute zero entries.
    - Internal helper:
        - `_node_dofs(n)`: Maps a node index to its global DOF indices.
    """
    n_dof = 6 * n_nodes

    def _node_dofs(n):  # 6 global DOFs for node n
        return list(range(6*n, 6*n + 6))

    P = np.zeros(n_dof)
    for n, load in nodal_loads.items():
        P[_node_dofs(n)] += np.asarray(load, dtype=float)
    return P


def partition_degrees_of_freedom(boundary_conditions, n_nodes):
    """
    Partitions the degrees of freedom (DOFs) into fixed and free sets for a 3D frame structure.

    Parameters
    ----------
    boundary_conditions : dict[int, array-like of bool]
        Dictionary mapping node indices to 6-element boolean arrays, where `True` indicates the DOF is fixed.
    
    n_nodes : int
        Total number of nodes in the structure.

    Returns
    -------
    fixed : ndarray of int
        Sorted array of fixed DOF indices.

    free : ndarray of int
        Sorted array of free DOF indices.
    """
    n_dof = n_nodes * 6
    fixed = []
    for n in range(n_nodes):
        flags = boundary_conditions.get(n)
        if flags is not None:
            fixed.extend([6*n + i for i, f in enumerate(flags) if f])
    fixed = np.asarray(fixed, dtype=int)
    free = np.setdiff1d(np.arange(n_dof), fixed, assume_unique=True)
    return fixed, free


def linear_solve(P_global, K_global, fixed, free):
    """
    Solves the linear system for displacements and internal nodal forces in a 3D linear elastic structure,
    using a partitioned approach based on fixed and free degrees of freedom (DOFs).

    The function solves for displacements at the free DOFs by inverting the corresponding submatrix
    of the global stiffness matrix (`K_ff`). A condition number check (`cond(K_ff) < 1e16`) is used
    to ensure numerical stability. If the matrix is well-conditioned, the system is solved and a nodal
    reaction vector is computed at the fixed DOFs.

    Parameters
    ----------
    P_global : ndarray of shape (n_dof,)
        The global load vector.

    K_global : ndarray of shape (n_dof, n_dof)
        The global stiffness matrix.

    fixed : array-like of int
        Indices of fixed degrees of freedom.

    free : array-like of int
        Indices of free degrees of freedom.

    Returns
    -------
    u : ndarray of shape (n_dof,)
        Displacement vector. Displacements are computed only for free DOFs; fixed DOFs are set to zero.

    nodal_reaction_vector : ndarray of shape (n_dof,)
        Nodal reaction vector. Reactions are computed only for fixed DOFs.

    Raises
    ------
    ValueError
        If the submatrix `K_ff` is ill-conditioned and the system cannot be reliably solved.
    """
    n_dof = len(fixed) + len(free)
    K_ff = K_global[np.ix_(free,  free)]
    K_sf = K_global[np.ix_(fixed, free)]
    condition_number = np.linalg.cond(K_ff)
    if condition_number < 10 ** 16:
        u_f = np.linalg.solve(K_ff, P_global[free])
        u = np.zeros(n_dof)
        u[free] = u_f
        nodal_reaction_vector = np.zeros(P_global.shape)
        nodal_reaction_vector[fixed] = K_sf @ u_f - P_global[fixed]
    else:
        raise ValueError(f"Cannot solve system: stiffness matrix is ill-conditioned (cond={condition_number:.2e})")
    return u, nodal_reaction_vector


def solve_linear_elastic_frame_3D(
    node_coords: np.ndarray,
    elements: Sequence[dict],
    boundary_conditions: dict[int, Sequence[int]],
    nodal_loads: dict[int, Sequence[float]],
):
    """
    Small-displacement linear-elastic analysis of a 3D frame using beam elements.

    Assumes global Cartesian coordinates and right-hand rule for orientation.
    This function assembles the global stiffness matrix and load vector, partitions
    the degrees of freedom (DOFs) into free and fixed sets, and solves the system
    for nodal displacements and support reactions.

    The system is solved using a partitioned approach. Displacements are computed
    at free DOFs, and true reaction forces (including contributions from both
    stiffness and applied loads) are computed at fixed DOFs. The system is only
    solved if the free-free stiffness matrix is well-conditioned
    (i.e., condition number < 1e16). If the matrix is ill-conditioned or singular,
    a ValueError is raised.

    Parameters
    ----------
    node_coords : (N, 3) float ndarray
        Global coordinates of the N nodes (row i → [x, y, z] of node i, 0-based).

    elements : iterable of dict
        Each dict must contain:
            'node_i', 'node_j' : int
                End node indices (0-based).
            'E', 'nu', 'A', 'I_y', 'I_z', 'J' : float
                Material and geometric properties.
            'local_z' : (3,) array or None
                Optional unit vector to define the local z-direction for transformation matrix orientation.

    boundary_conditions : dict[int, Sequence[int]]
        Maps node index to a 6-element iterable of 0 (free) or 1 (fixed) values.
        Omitted nodes are assumed to have all DOFs free.

    nodal_loads : dict[int, Sequence[float]]
        Maps node index to a 6-element array of applied loads:
        [Fx, Fy, Fz, Mx, My, Mz]. Omitted nodes are assumed to have zero loads.

    Returns
    -------
    u : (6 * N,) ndarray
        Global displacement vector. Entries are ordered as [UX, UY, UZ, RX, RY, RZ] for each node.
        Displacements are computed only at free DOFs; fixed DOFs are set to zero.

    r : (6 * N,) ndarray
        Global reaction force and moment vector. Nonzero values are present only at fixed DOFs
        and reflect the net support reactions, computed as internal elastic forces minus applied loads.

    Raises
    ------
    ValueError
        If the free-free stiffness matrix is ill-conditioned and the system cannot be reliably solved.

    Helper Functions Used
    ---------------------
    - assemble_global_stiffness_matrix_linear_elastic_3D(node_coords, elements)
        Assembles the global 6N x 6N stiffness matrix using local beam element stiffness and transformations.

    - assemble_global_load_vector_linear_elastic_3D(nodal_loads, n_nodes)
        Assembles the global load vector from nodal force/moment data.

    - partition_degrees_of_freedom(boundary_conditions, n_nodes)
        Identifies fixed and free degrees of freedom based on boundary condition flags.

    - linear_solve(P_global, K_global, fixed, free)
        Solves the reduced system for displacements and computes reaction forces.
        Raises a ValueError if the system is ill-conditioned.
    """
    # Dimensions
    n_nodes = node_coords.shape[0]

    # Assemble global stiffness matrix
    K = assemble_global_stiffness_matrix_linear_elastic_3D(node_coords, elements)

    # Assemble global load vector P
    P = assemble_global_load_vector_linear_elastic_3D(nodal_loads, n_nodes)

    # Free / fixed DOFs
    fixed, free = partition_degrees_of_freedom(boundary_conditions, n_nodes)

    # Solve
    u, r = linear_solve(P, K, fixed, free)

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

    Intended for *negative* tests (expected-failure scenarios).
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


def task_info():
    task_id = "solve_linear_elastic_frame_3D"
    task_short_description = "Self-contained function to solve a 3D linear elastic MSA problem"
    created_date = "2025-08-12"
    created_by = "elejeune11"
    main_fcn = solve_linear_elastic_frame_3D
    required_imports = [
        "import numpy as np",
        "from typing import Optional, Sequence",
        "import pytest"
    ]
    fcn_dependencies = [local_elastic_stiffness_matrix_3D_beam, beam_transformation_matrix_3D, assemble_global_stiffness_matrix_linear_elastic_3D, assemble_global_load_vector_linear_elastic_3D, partition_degrees_of_freedom, linear_solve]

    reference_verification_inputs = [
        [
            np.array([[0.0, 0.0, 0.0]] + [0.15 * np.array([1, 1, 1]) * i for i in range(1, 10)]),
            [dict(node_i=i, node_j=i+1, E=210e9, nu=0.3, A=5e-4, I_y=1e-6, I_z=1e-6, J=2e-6, local_z=None)
             for i in range(9)],
            {0: [1, 1, 1, 1, 1, 1]},
            {9: [0.0, 0.0, -200.0, 0.0, 0.0, 0.0]}
        ],
        [
            np.array([[i, 0.0, 0.0] for i in range(8)]),
            [dict(node_i=i, node_j=i+1, E=E_i, nu=0.3, A=1e-4, I_y=1e-8, I_z=1e-8, J=2e-8, local_z=None)
             for i, E_i in enumerate([70e9, 100e9, 140e9, 180e9, 220e9, 260e9, 300e9])],
            {0: [1, 1, 1, 1, 1, 1]},
            {7: [0.0, -400.0, 0.0, 0.0, 0.0, 0.0]}
        ],
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
        [
            np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.0, 0.5, 0.0]
            ]),
            [
                dict(node_i=0, node_j=1, E=200e9, nu=0.3, A=3e-4, I_y=2e-6, I_z=2e-6, J=4e-6, local_z=None),
                dict(node_i=1, node_j=2, E=200e9, nu=0.3, A=3e-4, I_y=2e-6, I_z=2e-6, J=4e-6, local_z=None),
                dict(node_i=1, node_j=3, E=200e9, nu=0.3, A=3e-4, I_y=2e-6, I_z=2e-6, J=4e-6, local_z=None),
                dict(node_i=0, node_j=3, E=200e9, nu=0.3, A=3e-4, I_y=2e-6, I_z=2e-6, J=4e-6, local_z=None),
                dict(node_i=2, node_j=3, E=200e9, nu=0.3, A=3e-4, I_y=2e-6, I_z=2e-6, J=4e-6, local_z=None),
                dict(node_i=3, node_j=4, E=200e9, nu=0.3, A=3e-4, I_y=2e-6, I_z=2e-6, J=4e-6, local_z=None),
                dict(node_i=1, node_j=4, E=200e9, nu=0.3, A=3e-4, I_y=2e-6, I_z=2e-6, J=4e-6, local_z=None),
            ],
            {
                0: [1, 1, 1, 1, 1, 1],
                2: [1, 1, 1, 1, 1, 1]
            },
            {
                3: [0.0, 0.0, -1000.0, 0.0, 0.0, 0.0]
            }
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
    ]

    return {
        "task_id": task_id,
        "task_short_description": task_short_description,
        "created_date": created_date,
        "created_by": created_by,
        "main_fcn": main_fcn,
        "required_imports": required_imports,
        "fcn_dependencies": fcn_dependencies,
        "reference_verification_inputs": reference_verification_inputs,
        "test_cases": test_cases,
        # "python_version": "version_number",
        # "package_versions": {"numpy": "version_number", },
    }
