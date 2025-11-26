import numpy as np
import pytest
from typing import Optional


def MSA_3D_transformation_matrix_CC0_H0_T0(
    x1: float, y1: float, z1: float,
    x2: float, y2: float, z2: float,
    reference_vector: Optional[np.ndarray] = None
) -> np.ndarray:
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
    L = np.sqrt(dx**2 + dy**2 + dz**2)
    
    if np.isclose(L, 0.0):
        raise ValueError("Beam has zero length (start and end coordinates are identical).")

    local_x = np.array([dx, dy, dz]) / L

    # choose a vector to orthonormalize the y axis if one is not given
    if reference_vector is None:
        # if the beam is oriented vertically, switch to the global y axis
        if np.isclose(local_x[0], 0.0) and np.isclose(local_x[1], 0.0):
            reference_vector = np.array([0.0, 1.0, 0.0])
        else:
            reference_vector = np.array([0.0, 0.0, 1.0])
    else:
        reference_vector = np.asarray(reference_vector)

        if reference_vector.shape != (3,):
            raise ValueError("Reference vector must have shape (3,).")

        if not np.isclose(np.linalg.norm(reference_vector), 1.0):
            raise ValueError("Expected a unit vector for reference vector.")

        if np.isclose(np.linalg.norm(np.cross(reference_vector, local_x)), 0.0):
            raise ValueError("Reference vector is parallel to beam axis.")

    # compute local y and z axes
    local_y = np.cross(reference_vector, local_x)
    local_y /= np.linalg.norm(local_y)

    local_z = np.cross(local_x, local_y)
    local_z /= np.linalg.norm(local_z)

    # assemble gamma and Gamma
    gamma = np.vstack((local_x, local_y, local_z))
    Gamma = np.kron(np.eye(4), gamma)

    return Gamma


def beam_transformation_matrix_3D_xp_wrong(
    x1: float, y1: float, z1: float,
    x2: float, y2: float, z2: float,
    reference_vector: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Incorrect version: defines local_y using cross(local_x, reference_vector),
    which flips the orientation of the local frame.
    """
    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
    L = np.sqrt(dx**2 + dy**2 + dz**2)
    
    if np.isclose(L, 0.0):
        raise ValueError("Beam has zero length (start and end coordinates are identical).")

    local_x = np.array([dx, dy, dz]) / L

    if reference_vector is None:
        if np.isclose(local_x[0], 0.0) and np.isclose(local_x[1], 0.0):
            reference_vector = np.array([0.0, 1.0, 0.0])
        else:
            reference_vector = np.array([0.0, 0.0, 1.0])

    # Cross product order is wrong here
    local_y = np.cross(local_x, reference_vector)
    local_y = local_y / np.linalg.norm(local_y)

    local_z = np.cross(local_x, local_y)
    local_z = local_z / np.linalg.norm(local_z)

    gamma = np.vstack((local_x, local_y, local_z))
    Gamma = np.kron(np.eye(4), gamma)
    return Gamma


def beam_transformation_matrix_3D_no_error_check(
    x1: float, y1: float, z1: float,
    x2: float, y2: float, z2: float,
    reference_vector: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Will not raise expected ValueErrors.
    """
    L = np.sqrt((x2 - x1) ** 2.0 + (y2 - y1) ** 2.0 + (z2 - z1) ** 2.0)
    lxp = (x2 - x1) / L
    mxp = (y2 - y1) / L
    nxp = (z2 - z1) / L
    local_x = np.asarray([lxp, mxp, nxp])

    # choose a vector to orthonormalize the y axis if one is not given
    if reference_vector is None:
        # if the beam is oriented vertically, switch to the global y axis
        if np.isclose(lxp, 0.0) and np.isclose(mxp, 0.0):
            reference_vector = np.array([0, 1.0, 0.0])
        else:
            # otherwise use the global z axis
            reference_vector = np.array([0, 0, 1.0])

    # compute the local y axis
    local_y = np.cross(reference_vector, local_x)
    local_y = local_y / np.linalg.norm(local_y)

    # compute the local z axis
    local_z = np.cross(local_x, local_y)
    local_z = local_z / np.linalg.norm(local_z)

    # assemble gamma
    gamma = np.vstack((local_x, local_y, local_z))

    # assemble Gamma
    Gamma = np.zeros((12, 12))
    Gamma[0:3, 0:3] = gamma
    Gamma[3:6, 3:6] = gamma
    Gamma[6:9, 6:9] = gamma
    Gamma[9:12, 9:12] = gamma
    return Gamma


def beam_transformation_matrix_3D_gamma_T(
    x1: float, y1: float, z1: float,
    x2: float, y2: float, z2: float,
    reference_vector: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Incorrect version: transposes `gamma` before inserting into Gamma,
    effectively inverting the rotation direction.
    """
    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
    L = np.sqrt(dx**2 + dy**2 + dz**2)
    if np.isclose(L, 0.0):
        raise ValueError("Beam has zero length")

    local_x = np.array([dx, dy, dz]) / L

    if reference_vector is None:
        if np.isclose(local_x[0], 0.0) and np.isclose(local_x[1], 0.0):
            reference_vector = np.array([0.0, 1.0, 0.0])
        else:
            reference_vector = np.array([0.0, 0.0, 1.0])
    else:
        reference_vector = np.asarray(reference_vector)
        if reference_vector.shape != (3,):
            raise ValueError("Reference vector must have shape (3,).")
        if not np.isclose(np.linalg.norm(reference_vector), 1.0):
            raise ValueError("Expected a unit vector for reference vector.")
        if np.isclose(np.linalg.norm(np.cross(reference_vector, local_x)), 0.0):
            raise ValueError("Reference vector is parallel to beam axis.")

    local_y = np.cross(reference_vector, local_x)
    local_y = local_y / np.linalg.norm(local_y)
    local_z = np.cross(local_x, local_y)
    local_z = local_z / np.linalg.norm(local_z)

    gamma = np.vstack((local_x, local_y, local_z))

    # Incorrect: transpose gamma before repeating
    Gamma = np.kron(np.eye(4), gamma.T)
    return Gamma


def test_cardinal_axis_alignment(fcn):
    """
    Test transformation matrices for beams aligned with global coordinate axes.
    
    Verifies that the 3x3 direction cosine matrix (first block of the 12x12 transformation
    matrix) has the expected orientation for beams aligned with each cardinal direction.
    
    The expected orientations are determined by:
    - local_x: beam direction (normalized)
    - reference_vector: global z-axis for non-vertical beams, global y-axis for vertical beams
    - local_y: cross(reference_vector, local_x), normalized
    - local_z: cross(local_x, local_y), normalized
    
    Test cases:
    - X-axis beam, use global z-axis as reference vector.
    - Y-axis beam, use global z-axis as reference vector.
    - Z-axis beam, use global y-axis as reference vector due to vertical beam logic.
    """
    test_cases = [
        # (coords, expected_gamma, description)
        ((0, 0, 0, 1, 0, 0), np.eye(3), "x-axis alignment"),
        ((0, 0, 0, 0, 1, 0), np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]), "y-axis alignment"),
        ((0, 0, 0, 0, 0, 1), np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]), "z-axis alignment"),
    ]
    
    for coords, expected, description in test_cases:
        Gamma = fcn(*coords)
        gamma = Gamma[:3, :3]
        assert np.allclose(gamma, expected, atol=1e-12), f"Gamma block incorrect for {description}"


def test_transformation_matrix_properties(fcn):
    """
    Test fundamental mathematical properties of transformation matrices for multiple beam configurations.
    Test specific examples with known behavior and verifies that the tranformation is correct.
    """
    # Fundamental mathematical properties
    beam_cases = [
        # (coords, reference_vector, label)
        ((0, 0, 0, 1, 0, 0), None, "X-axis"),
        ((0, 0, 0, 0, 1, 0), None, "Y-axis"),
        ((0, 0, 0, 0, 0, 1), None, "Z-axis"),
        ((0, 0, 0, 1, 1, 1), None, "Diagonal XYZ"),
        ((1, 2, 3, 4, 5, 6), np.array([0, 0, 1]), "Arbitrary + Z ref"),
        ((0, 0, 0, 1, 2, 3), None, "Skewed diagonal"),
    ]

    for coords, reference_vector, label in beam_cases:
        Gamma = fcn(*coords, reference_vector=reference_vector)
        gamma = Gamma[:3, :3]

        # (1) Orthonormality
        assert np.allclose(gamma.T @ gamma, np.eye(3), atol=1e-12), \
            f"Not orthonormal: {fcn.__name__} [{label}]"

        # (2) Proper rotation
        det = np.linalg.det(gamma)
        assert np.isclose(det, 1.0, atol=1e-12), \
            f"Improper rotation (det != 1): det={det} in {fcn.__name__} [{label}]"

        # (3) Block structure
        for i in range(1, 4):
            block = Gamma[3*i:3*(i+1), 3*i:3*(i+1)]
            assert np.allclose(block, gamma, atol=1e-12), \
                f"Block {i} mismatch in {fcn.__name__} [{label}]"

        # (4) Round-trip check
        v_local = np.random.rand(12)
        v_global = Gamma @ v_local
        v_recovered = Gamma.T @ v_global
        assert np.allclose(v_local, v_recovered, atol=1e-12), \
            f"Round-trip failed in {fcn.__name__} [{label}]"

        # (5) Shape
        assert Gamma.shape == (12, 12), \
            f"Incorrect shape {Gamma.shape} in {fcn.__name__} [{label}]"

    # Known examples
    Gamma = fcn(0, 0, 0, 1, 1, 0, reference_vector=np.array([0, 0, 1]))
    gamma = Gamma[:3, :3]
    sqrt2 = np.sqrt(2)
    expected_gamma = np.array([
        [1/sqrt2,  1/sqrt2,  0],
        [-1/sqrt2, 1/sqrt2,  0],
        [0,        0,        1]
    ])
    assert np.allclose(gamma, expected_gamma, atol=1e-12), \
        f"Known gamma mismatch: got\n{gamma}\nexpected\n{expected_gamma}"

    Gamma = fcn(0, 0, 0, 0, 1, 1, reference_vector=np.array([1, 0, 0]))
    gamma = Gamma[:3, :3]
    sqrt2 = np.sqrt(2)
    expected_2 = np.array([
        [0,         1/np.sqrt(2),  1/np.sqrt(2)],
        [0,        -1/np.sqrt(2),  1/np.sqrt(2)],
        [1,         0,             0           ]
    ])
    assert np.allclose(gamma, expected_2, atol=1e-12), \
        f"Known gamma mismatch (diag YZ):\nExpected:\n{expected_2}\nGot:\n{gamma}"


def test_beam_transformation_matrix_error_messages(fcn):
    """
    Test that the function raises appropriate ValueError exceptions for invalid reference vectors.
    Verifies proper error handling for two invalid reference vector conditions:
    1. Non-unit vector: When the reference vector's magnitude is not equal to 1.0
    2. Parallel reference vector: When the reference vector is parallel to the beam axis
        - This would result in a zero cross product, making it impossible to define orthogonal local y and z axes
    Also veriefies proper error handling for:
    3. Zero-length beam: start and end coordinates are identical
    All conditions should raise ValueError
    """
    # --- Not a unit vector ---
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 0, 0, reference_vector=np.array([1.0, 1.0, 0.0]))

    # --- Parallel reference vector ---
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 0, 0, reference_vector=np.array([1.0, 0.0, 0.0]))

    # --- Zero-length beam ---
    with pytest.raises(ValueError):
        fcn(1, 1, 1, 1, 1, 1)


def task_info():
    task_id = "MSA_3D_transformation_matrix_CC0_H0_T0"
    task_short_description = "creates a 3D beam element transformation matrix"
    created_date = "2025-08-02"
    created_by = "elejeune11"
    main_fcn = MSA_3D_transformation_matrix_CC0_H0_T0
    required_imports = ["import numpy as np", "from typing import Callable, Optional", "import pytest"]
    fcn_dependencies = []
    reference_verification_inputs = [
        # 1. Horizontal beam along global x-axis, no reference vector
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, None],

        # 2. Vertical beam along global z-axis, no reference vector (should fallback to y-axis)
        [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, None],

        # 3. Beam at 45 degrees in x-y plane, with z-up reference
        [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, np.array([0.0, 0.0, 1.0])],

        # 4. Arbitrary 3D skewed beam, y-up reference
        [1.0, 2.0, 3.0, 4.0, 6.0, 5.0, np.array([0.0, 1.0, 0.0])],

        # 5. Negative-direction beam along global x-axis
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.array([0.0, 0.0, 1.0])],

        # 6. Negative-direction skewed beam
        [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, np.array([0.0, 0.0, 1.0])]
    ]
    test_cases = [
        {"test_code": test_cardinal_axis_alignment,
         "expected_failures": [beam_transformation_matrix_3D_xp_wrong]},
        {"test_code": test_transformation_matrix_properties,
         "expected_failures": [beam_transformation_matrix_3D_xp_wrong, beam_transformation_matrix_3D_gamma_T]},
        {"test_code": test_beam_transformation_matrix_error_messages,
         "expected_failures": [beam_transformation_matrix_3D_no_error_check]},
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