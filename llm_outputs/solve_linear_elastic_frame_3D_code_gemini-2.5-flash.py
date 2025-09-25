import numpy as np
from typing import Optional, Sequence
import pytest
def local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J):
    """
    Return the 12x12 local elastic stiffness matrix for a 3D Euler-Bernoulli beam element.
    The beam is assumed to be aligned with the local x-axis. The stiffness matrix
    relates local nodal displacements and rotations to forces and moments using the equation:
        [force_vector] = [stiffness_matrix] @ [displacement_vector]
    Degrees of freedom are ordered as:
        [u1, v1, w1, θx1, θy1, θz1, u2, v2, w2, θx2, θy2, θz2]
    Where:
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
    k[0, 0] = k[6, 6] = EA_L
    k[0, 6] = k[6, 0] = -EA_L
    k[3, 3] = k[9, 9] = GJ_L
    k[3, 9] = k[9, 3] = -GJ_L
    k[1, 1] = k[7, 7] = 12.0 * EIz_L / L**3
    k[1, 7] = k[7, 1] = -12.0 * EIz_L / L**3
    k[1, 5] = k[5, 1] = k[1, 11] = k[11, 1] = 6.0 * EIz_L / L**2
    k[5, 7] = k[7, 5] = k[7, 11] = k[11, 7] = -6.0 * EIz_L / L**2
    k[5, 5] = k[11, 11] = 4.0 * EIz_L / L
    k[5, 11] = k[11, 5] = 2.0 * EIz_L / L
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
                Indices of the start and end nodes.
                Young's modulus of the element.
                Poisson's ratio of the element.
                Cross-sectional area.
                Second moments of area about local y and z axes.
                Torsional constant.
                Optional vector defining the local z-direction to resolve transformation ambiguity.
    Returns
    -------
    K : ndarray of shape (6 * n_nodes, 6 * n_nodes)
        The assembled global stiffness matrix of the structure, with 6 degrees of freedom per node.
    Helper Functions Required
    -------------------------
        Returns the 12x12 local stiffness matrix for a 3D beam element.
        Returns the 12x12 transformation matrix to convert local stiffness to global coordinates.
    Notes
    -----
    """
    n_nodes = node_coords.shape[0]
    n_dof = 6 * n_nodes
    def _node_dofs(n):  # 6 global DOFs for node n
        return list(range(6*n, 6*n + 6))
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
      [u_x, u_y, u_z, θ_x, θ_y, θ_z].
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
def solve_linear_elastic_frame_3D(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    """
    Small-displacement linear-elastic analysis of a 3D frame using beam elements.
    Assumes global Cartesian coordinates and right-hand rule for orientation.
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
                Material