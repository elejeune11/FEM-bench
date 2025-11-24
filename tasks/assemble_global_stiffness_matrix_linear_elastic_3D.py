import numpy as np
from typing import Optional


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


def assemble_global_stiffness_matrix_linear_elastic_3D_broken(node_coords, elements):
    n_nodes = node_coords.shape[0]
    n_dof = 6 * n_nodes
    K = np.zeros((n_dof, n_dof))
    K[0:12, 0:12] = np.ones((12, 12))
    return K


def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """
    Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.

    Covers multiple structural configurations, for example: single element, linear chain, triangle loop, and square loop.
    """
    # Case 1: One element, two nodes
    node_coords1 = np.array([[0, 0, 0], [1, 0, 0]])
    elements1 = [{
        'node_i': 0, 'node_j': 1, 'E': 1, 'nu': 0.3,
        'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1
    }]

    # Case 2: Two elements in series (3 nodes)
    node_coords2 = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    elements2 = [
        {'node_i': 0, 'node_j': 1, 'E': 1, 'nu': 0.3, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1},
        {'node_i': 1, 'node_j': 2, 'E': 1, 'nu': 0.3, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1}
    ]

    # Case 3: Triangle loop
    node_coords3 = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]])
    elements3 = [
        {'node_i': 0, 'node_j': 1, 'E': 1, 'nu': 0.3, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1},
        {'node_i': 1, 'node_j': 2, 'E': 1, 'nu': 0.3, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1},
        {'node_i': 2, 'node_j': 0, 'E': 1, 'nu': 0.3, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1}
    ]

    # Case 4: Square loop (4 nodes, 4 elements)
    node_coords4 = np.array([
        [0, 0, 0],    # Node 0: bottom-left
        [1, 0, 0],    # Node 1: bottom-right
        [1, 1, 0],    # Node 2: top-right
        [0, 1, 0]     # Node 3: top-left
    ])
    elements4 = [
        {'node_i': 0, 'node_j': 1, 'E': 1, 'nu': 0.3,
         'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1},  # bottom
        {'node_i': 1, 'node_j': 2, 'E': 1, 'nu': 0.3,
         'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1},  # right
        {'node_i': 2, 'node_j': 3, 'E': 1, 'nu': 0.3,
         'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1},  # top
        {'node_i': 3, 'node_j': 0, 'E': 1, 'nu': 0.3,
         'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1}   # left
    ]

    test_cases = [
        (node_coords1, elements1),
        (node_coords2, elements2),
        (node_coords3, elements3),
        (node_coords4, elements4),
    ]

    for node_coords, elements in test_cases:
        n_nodes = node_coords.shape[0]
        K = fcn(node_coords, elements)

        # Check shape
        expected_shape = (6 * n_nodes, 6 * n_nodes)
        assert K.shape == expected_shape, f"Incorrect shape: got {K.shape}, expected {expected_shape}"

        # Check symmetry
        assert np.allclose(K, K.T), "Stiffness matrix is not symmetric"

        # Confirm each element contributed a nonzero 12×12 block at its DOF location
        for ele in elements:
            ni, nj = ele["node_i"], ele["node_j"]
            dofs = list(range(6 * ni, 6 * ni + 6)) + list(range(6 * nj, 6 * nj + 6))
            K_block = K[np.ix_(dofs, dofs)]
            assert np.any(K_block != 0), (
                f"Element ({ni}, {nj}) appears to have contributed no stiffness to the global matrix"
            )


def task_info():
    task_id = "assemble_global_stiffness_matrix_linear_elastic_3D"
    task_short_description = "assembles the global stiffness matrix for a 3D linear elastic frame"
    created_date = "2025-08-11"
    created_by = "elejeune11"
    main_fcn = assemble_global_stiffness_matrix_linear_elastic_3D
    required_imports = ["import numpy as np", "from typing import Optional", "import pytest"]
    fcn_dependencies = [local_elastic_stiffness_matrix_3D_beam, beam_transformation_matrix_3D]
    reference_verification_inputs = [
        [
            np.array([[0.0, 0.0, 0.0]] + [0.15 * np.array([1, 1, 1]) * i for i in range(1, 10)]),
            [dict(node_i=i, node_j=i+1, E=210e9, nu=0.3, A=5e-4, I_y=1e-6, I_z=1e-6, J=2e-6, local_z=None)
            for i in range(9)]
        ],
        [
            np.array([[i, 0.0, 0.0] for i in range(8)]),
            [dict(node_i=i, node_j=i+1, E=E_i, nu=0.3, A=1e-4, I_y=1e-8, I_z=1e-8, J=2e-8, local_z=None)
            for i, E_i in enumerate([70e9, 100e9, 140e9, 180e9, 220e9, 260e9, 300e9])]
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
            ]
        ],
        [
            np.array([
                [0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0],
                [0, 0, 2], [2, 0, 2], [2, 2, 2], [0, 2, 2],
                [1, 1, 0], [1, 1, 2]
            ]),
            [dict(node_i=i, node_j=j, E=200e9, nu=0.3, A=1e-4, I_y=3e-8, I_z=3e-8, J=6e-8, local_z=None)
            for i, j in [(0, 1), (1, 2), (2, 3), (3, 0),
                        (4, 5), (5, 6), (6, 7), (7, 4),
                        (8, 9), (0, 4)]]
        ],
        [
            np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]]),
            [dict(node_i=i, node_j=i+1, E=210e9, nu=0.3, A=5e-4, I_y=1e-6, I_z=1e-6, J=2e-6, local_z=None)
            for i in range(4)]
        ]
    ]
    test_cases = [{"test_code": test_assemble_global_stiffness_matrix_shape_and_symmetry, "expected_failures": [assemble_global_stiffness_matrix_linear_elastic_3D_broken]}]
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
