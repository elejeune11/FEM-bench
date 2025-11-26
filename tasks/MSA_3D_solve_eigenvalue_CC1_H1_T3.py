import numpy as np
import pytest
import scipy


def MSA_3D_solve_eigenvalue_CC1_H1_T3(K_e_global: np.ndarray, K_g_global: np.ndarray, boundary_conditions: dict, n_nodes: int):
    """
    Compute the smallest positive elastic critical load factor and corresponding
    global buckling mode shape for a 3D frame/beam model.

    The generalized eigenproblem is solved on the free DOFs:
        K_e_ff * phi = -lambda * K_g_ff * phi
    where K_e_ff and K_g_ff are the partitions of the elastic and geometric
    stiffness matrices after applying boundary conditions.

    Parameters
    ----------
    K_e_global : ndarray, shape (6*n_nodes, 6*n_nodes)
        Global elastic stiffness matrix.
    K_g_global : ndarray, shape (6*n_nodes, 6*n_nodes)
        Global geometric stiffness matrix at the reference load state.
    boundary_conditions : dict[int, array-like of bool]
        Dictionary mapping each node index (0-based) to a 6-element boolean array
        defining the constrained DOFs:
            True  → fixed (prescribed zero displacement/rotation)
            False → free (unknown)
        Nodes not listed are assumed fully free.
    n_nodes : int
        Number of nodes in the model (assumes 6 DOFs per node, ordered
        [u_x, u_y, u_z, theta_x, theta_y, theta_z] per node).

    Returns
    -------
    elastic_critical_load_factor : float
        The smallest positive eigenvalue lambda (> 0), interpreted as the elastic
        critical load factor (i.e., P_cr = lambda * P_ref, if K_g_global was
        formed at reference load P_ref).
    deformed_shape_vector : ndarray, shape (6*n_nodes,)
        Global buckling mode vector with entries on constrained DOFs set to zero.
        No normalization is applied (matches original behavior).

    Raises
    ------
    ValueError
        - If the reduced matrices are ill-conditioned/singular beyond tolerance.
            Use a tolerence of 1e16
        - If no positive eigenvalue is found.
        - If eigenpairs contain non-negligible complex parts.
    """

    def _partition_degrees_of_freedom(boundary_conditions: dict, n_nodes: int):
        n_dof = n_nodes * 6
        fixed = []
        for n in range(n_nodes):
            flags = boundary_conditions.get(n)
            if flags is not None:
                fixed.extend([6*n + i for i, f in enumerate(flags) if f])
        fixed = np.asarray(fixed, dtype=int)
        free = np.setdiff1d(np.arange(n_dof), fixed, assume_unique=True)
        return fixed, free

    # --- Constants / tolerances ---
    dof_per_node = 6
    pos_tol = 1e-10               # smallest positive eigenvalue threshold
    cond_limit_e = 1e16           # condition number for elastic stiffness
    cond_limit_g = 1e16           # condition number for geometric stiffness
    real_tol = 1e3                # tolerance for np.real_if_close

    # --- DOF partitioning ---
    _, free = _partition_degrees_of_freedom(boundary_conditions, n_nodes)
    free = np.asarray(free, dtype=int)

    # --- Free-free blocks ---
    K_e_ff = K_e_global[np.ix_(free, free)]
    K_g_ff = K_g_global[np.ix_(free, free)]

    # --- Conditioning checks (before solve) ---
    cond_e = np.linalg.cond(K_e_ff)
    if not np.isfinite(cond_e) or cond_e > cond_limit_e:
        raise ValueError(
            f"Elastic stiffness (free-free) is ill-conditioned (cond={cond_e:.3e}). "
            "Check boundary conditions (remove rigid-body modes) and material/section inputs."
        )
    cond_g = np.linalg.cond(K_g_ff)
    if not np.isfinite(cond_g) or cond_g > cond_limit_g:
        raise ValueError(
            f"Geometric stiffness (free-free) is ill-conditioned (cond={cond_g:.3e}). "
            "Ensure the reference load state and element forces produce a valid K_g."
        )

    # --- Generalized eigenvalue solve: K_e_ff v = -lambda K_g_ff v ---
    eig_vals, eig_vecs = scipy.linalg.eig(K_e_ff, -1.0 * K_g_ff)

    # Coerce to real if imaginary parts are negligible
    eig_vals = np.real_if_close(eig_vals, tol=real_tol)
    eig_vecs = np.real_if_close(eig_vecs, tol=real_tol)

    # Verify coercion succeeded (catch genuinely complex modes)
    if np.iscomplexobj(eig_vals) or np.iscomplexobj(eig_vecs):
        raise ValueError(
            "Eigen-solution returned significantly complex eigenpairs. "
            "This may indicate inconsistent matrices or severe roundoff issues."
        )

    # --- Select smallest positive eigenvalue ---
    vals = eig_vals.astype(float, copy=False)
    mask = np.isfinite(vals) & (vals > pos_tol)
    if not np.any(mask):
        raise ValueError(
            "No positive buckling factors found. "
            "Verify the sign convention (K_e phi = -lambda K_g phi), boundary conditions, and K_g assembly."
        )
    ix = np.argmin(vals[mask])
    true_indices = np.flatnonzero(mask)
    ix = true_indices[ix]

    eig_value = float(vals[ix])
    eig_vector_free = eig_vecs[:, ix].astype(float, copy=False)

    # --- Embed back into global vector (constrained DOFs set to zero) ---
    num_dofs = dof_per_node * n_nodes
    deformed_shape_vector = np.zeros((num_dofs,), dtype=float)
    deformed_shape_vector[free] = eig_vector_free

    elastic_critical_load_factor = eig_value
    return elastic_critical_load_factor, deformed_shape_vector


def eigenvalue_analysis_bad_allzeros(K_e_global, K_g_global, boundary_conditions, n_nodes):
    """
    Always returns λ=0 and a zero mode.
    Fails: known-answer (λ>0), scaling, and error raising.
    """
    ndof = 6 * n_nodes
    return 0.0, np.zeros(ndof)


def eigenvalue_analysis_bad_skip_checks(K_e_global, K_g_global, boundary_conditions, n_nodes):
    """
    Ignores error raising and returns a canned answer.
    """
    ndof = 6 * n_nodes
    return 1.0, np.ones(ndof)


def eigenvalue_analysis_bad_wrong_scaling(K_e_global, K_g_global, boundary_conditions, n_nodes):
    """
    Makes λ proportional to ||K_g|| (wrong sign/scaling convention).
    Fails: invariance-to-reference-load-scaling (λ should scale as 1/c, not ∝ c).
    """
    lam = float(np.linalg.norm(K_g_global, ord='fro'))
    ndof = 6 * n_nodes
    return lam, np.zeros(ndof)


def test_eigen_known_answer(fcn):
    """
    Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF.
    """
    n_nodes = 2
    ndof = 6 * n_nodes

    # Boundary conditions: node 0 fully fixed; node 1 -> free (uy, rz)
    bc = {
        0: [True, True, True, True, True, True],
        1: [True, False, True, True, True, False],
    }

    # Compute constrained DOF indices (inline; True = fixed)
    constrained = []
    for n in range(n_nodes):
        flags = bc.get(n)
        if flags is not None:
            idx = np.nonzero(np.asarray(flags, dtype=bool))[0]
            constrained.extend((6 * n + idx).tolist())
    constrained = np.asarray(sorted(constrained), dtype=int)

    # Diagonal Ke (strictly increasing) and Kg = -I
    Ke = np.diag(np.linspace(2.0, 20.0, ndof))
    Kg = -np.eye(ndof)

    # Free DOFs implied by BC above
    free = np.array([7, 11], dtype=int)

    # Closed-form expected eigenvalue: min diag among free DOFs
    diag_free = np.array([Ke[d, d] for d in free])
    expected_lambda = float(np.min(diag_free))
    # DOF index (in 'free') where the minimum occurs
    imin_local = int(np.argmin(diag_free))
    dof_min = int(free[imin_local])
    dof_other = int(free[1 - imin_local])

    lam, mode = fcn(Ke, Kg, bc, n_nodes)

    # Eigenvalue should match the analytic value
    assert np.isfinite(lam) and lam > 0.0
    assert np.isclose(lam, expected_lambda, rtol=1e-12, atol=0.0)

    # Mode size and constrained zeros
    assert mode.shape == (ndof,)
    assert np.allclose(mode[constrained], 0.0)

    # Mode should be aligned with the minimal-diagonal DOF (up to scale)
    #    i.e., |mode[dof_min]| >> |mode[dof_other]|
    amp_min = abs(mode[dof_min])
    amp_other = abs(mode[dof_other])
    # allow near-zero overall scaling but require relative dominance if nonzero
    if amp_min > 0.0 or amp_other > 0.0:
        assert amp_min >= 1e3 * amp_other, (
            f"Mode not aligned with expected DOF: |mode[{dof_min}]|={amp_min:.3e}, "
            f"|mode[{dof_other}]|={amp_other:.3e}"
        )


def test_eigen_singluar_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned.
    """
    n_nodes = 2
    ndof = 6 * n_nodes

    # No constraints
    bc = {}

    Ke = np.ones((ndof, ndof), dtype=float)  # singular
    Kg = -np.eye(ndof)

    with pytest.raises(ValueError, match=r"Elastic stiffness .* ill-conditioned"):
        _ = fcn(Ke, Kg, bc, n_nodes)


def test_eigen_complex_eigenpairs_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs.
    """
    n_nodes = 2
    ndof = 6 * n_nodes

    # Clamp node 0; at node 1 keep (uy, rz) free
    bc = {
        0: [True, True, True, True, True, True],
        1: [True, False, True, True, True, False],
    }

    Ke = np.zeros((ndof, ndof), dtype=float)
    Kg = np.zeros((ndof, ndof), dtype=float)

    # Free DOFs (node 1): uy (7), rz (11)
    ff = np.array([7, 11], dtype=int)

    # Symmetric Positive Definite elastic sub-block
    Ke_ff = np.array([[3.0, 0.2],
                      [0.2, 4.0]], dtype=float)
    # Strongly non-symmetric geometric sub-block
    Kg_ff = np.array([[0.0,  1.0],
                      [-2.0, 0.0]], dtype=float)

    Ke[np.ix_(ff, ff)] = Ke_ff
    Kg[np.ix_(ff, ff)] = Kg_ff

    with pytest.raises(ValueError, match=r"significantly complex eigenpairs"):
        _ = fcn(Ke, Kg, bc, n_nodes)


def test_eigen_no_positive_eigenvalues_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present.
    """
    n_nodes = 2
    ndof = 6 * n_nodes

    # Boundary conditions: node 0 fully fixed, node 1 free in (uy, rz)
    bc = {
        0: [True, True, True, True, True, True],
        1: [True, False, True, True, True, False],
    }

    # Diagonal Symmetric Positive Definite elastic stiffness
    Ke = np.diag(np.linspace(2.0, 20.0, ndof))
    # Make geometric stiffness +I instead of -I → leads to negative eigenvalues
    Kg = np.eye(ndof)

    with pytest.raises(ValueError, match=r"No positive buckling factors found"):
        _ = fcn(Ke, Kg, bc, n_nodes)


def test_eigen_invariance_to_reference_load_scaling(fcn):
    """
    Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size.
    """
    n_nodes = 2
    ndof = 6 * n_nodes

    bc = {
        0: [True, True, True, True, True, True],
        1: [True, False, True, True, True, False],
    }

    Ke = np.diag(np.linspace(2.0, 20.0, ndof))  # SPD diagonal
    Kg = -np.eye(ndof)

    lam_base, mode_base = fcn(Ke, Kg, bc, n_nodes)

    c = 10.0
    lam_scaled, mode_scaled = fcn(Ke, c * Kg, bc, n_nodes)

    assert np.isfinite(lam_base) and lam_base > 0.0
    assert np.isfinite(lam_scaled) and lam_scaled > 0.0
    assert np.isclose(lam_scaled, lam_base / c, rtol=1e-12, atol=0.0)
    assert mode_base.shape == (ndof,)
    assert mode_scaled.shape == (ndof,)
    assert np.all(np.isfinite(mode_base)) and np.all(np.isfinite(mode_scaled))


def task_info():
    task_id = "MSA_3D_solve_eigenvalue_CC1_H1_T3"
    task_short_description = "performs eigenvalue analysis given global stiffness matricies and boundary conditions"
    created_date = "2025-09-17"
    created_by = "elejeune11"
    main_fcn = MSA_3D_solve_eigenvalue_CC1_H1_T3
    required_imports = ["import numpy as np", "import scipy", "import pytest", "from typing import Callable"]
    fcn_dependencies = []
    reference_verification_inputs = [
        # 1) 2 nodes, diagonal Ke, Kg = -I, free set: node 1 (uy, rz)
        [
            np.diag(np.linspace(2.0, 20.0, 12)),
            -np.eye(12),
            {
                0: [True, True, True, True, True, True],
                1: [True, False, True, True, True, False],
            },
            2
        ],
        # 2) 2 nodes, SPD Ke from fixed random matrix, Kg = -5 I
        [
            (np.array([
                [1.76405235, 0.40015721, 0.97873798, 2.2408932 , 1.86755799, -0.97727788,
                0.95008842, -0.15135721, -0.10321885, 0.4105985 , 0.14404357, 1.45427351],
                [0.76103773, 0.12167502, 0.44386323, 0.33367433, 1.49407907, -0.20515826,
                0.3130677 , -0.85409574, -2.55298982, 0.6536186 , 0.8644362 , -0.74216502],
                [2.26975462, -1.45436567, 0.04575852, -0.18718385, 1.53277921, 1.46935877,
                0.15494743, 0.37816252, -0.88778575, -1.98079647, -0.34791215, 0.15634897],
                [1.23029068, 1.20237985, -0.38732682, -0.30230275, -1.04855297, -1.42001794,
                -1.70627019, 1.9507754 , -0.50965218, -0.4380743 , -1.25279536, 0.77749036],
                [-1.61389785, -0.21274028, -0.89546656, 0.3869025 , -0.51080514, -1.18063218,
                -0.02818223, 0.42833187, 0.06651722, 0.3024719 , -0.63432209, -0.36274117],
                [-0.67246045, -0.35955316, -0.81314628, -1.7262826 , 0.17742614, -0.40178094,
                -1.63019835, 0.46278226, -0.90729836, 0.0519454 , 0.72909056, 0.12898291],
                [1.13940068, -1.23482582, 0.40234164, -0.68481009, -0.87079715, -0.57884966,
                -0.31155253, 0.05616534, -1.16514984, 0.90082649, 0.46566244, -1.53624369],
                [1.48825219, 1.89588918, 1.17877957, -0.17992484, -1.07075262, 1.05445173,
                -0.40317695, 1.22244507, 0.20827498, 0.97663904, 0.3563664 , 0.70657317],
                [0.01050002, 1.78587049, 0.12691209, 0.40198936, 1.8831507 , -1.34775906,
                -1.270485  , 0.96939671, -1.17312341, 1.94362119, -0.41361898, -0.74745481],
                [1.92294203, 1.48051479, 1.86755896, 0.90604466, -0.86122569, 1.91006495,
                -0.26800337, 0.8024564 , 0.94725197, -0.15501009, 0.61407937, 0.92220667],
                [0.37642553, -1.09940079, 0.29823817, 1.3263859 , -0.69456786, -0.14963454,
                -0.43515355, 1.84926373, 0.67229476, 0.40746184, -0.76991607, 0.53924919],
                [-0.67433266, 0.03183056, -0.63584608, 0.67643329, 0.57659082, -0.20829876,
                0.39600671, -1.09306151, -1.49125759, 0.4393917 , 0.1666735 , 0.63503144],
            ]).T @ np.array([
                [1.76405235, 0.40015721, 0.97873798, 2.2408932 , 1.86755799, -0.97727788,
                0.95008842, -0.15135721, -0.10321885, 0.4105985 , 0.14404357, 1.45427351],
                [0.76103773, 0.12167502, 0.44386323, 0.33367433, 1.49407907, -0.20515826,
                0.3130677 , -0.85409574, -2.55298982, 0.6536186 , 0.8644362 , -0.74216502],
                [2.26975462, -1.45436567, 0.04575852, -0.18718385, 1.53277921, 1.46935877,
                0.15494743, 0.37816252, -0.88778575, -1.98079647, -0.34791215, 0.15634897],
                [1.23029068, 1.20237985, -0.38732682, -0.30230275, -1.04855297, -1.42001794,
                -1.70627019, 1.9507754 , -0.50965218, -0.4380743 , -1.25279536, 0.77749036],
                [-1.61389785, -0.21274028, -0.89546656, 0.3869025 , -0.51080514, -1.18063218,
                -0.02818223, 0.42833187, 0.06651722, 0.3024719 , -0.63432209, -0.36274117],
                [-0.67246045, -0.35955316, -0.81314628, -1.7262826 , 0.17742614, -0.40178094,
                -1.63019835, 0.46278226, -0.90729836, 0.0519454 , 0.72909056, 0.12898291],
                [1.13940068, -1.23482582, 0.40234164, -0.68481009, -0.87079715, -0.57884966,
                -0.31155253, 0.05616534, -1.16514984, 0.90082649, 0.46566244, -1.53624369],
                [1.48825219, 1.89588918, 1.17877957, -0.17992484, -1.07075262, 1.05445173,
                -0.40317695, 1.22244507, 0.20827498, 0.97663904, 0.3563664 , 0.70657317],
                [0.01050002, 1.78587049, 0.12691209, 0.40198936, 1.8831507 , -1.34775906,
                -1.270485  , 0.96939671, -1.17312341, 1.94362119, -0.41361898, -0.74745481],
                [1.92294203, 1.48051479, 1.86755896, 0.90604466, -0.86122569, 1.91006495,
                -0.26800337, 0.8024564 , 0.94725197, -0.15501009, 0.61407937, 0.92220667],
                [0.37642553, -1.09940079, 0.29823817, 1.3263859 , -0.69456786, -0.14963454,
                -0.43515355, 1.84926373, 0.67229476, 0.40746184, -0.76991607, 0.53924919],
                [-0.67433266, 0.03183056, -0.63584608, 0.67643329, 0.57659082, -0.20829876,
                0.39600671, -1.09306151, -1.49125759, 0.4393917 , 0.1666735 , 0.63503144],
            ])) + 0.5 * np.eye(12),
            -5.0 * np.eye(12),
            {
                0: [True, True, True, True, True, True],
                1: [True, False, True, True, True, False],
            },
            2
        ],
        # 3) 3 nodes, diagonal Ke, Kg = -2 I, free set: node 2 (uy, rz)
        [
            np.diag(np.linspace(10.0, 40.0, 18)),
            -2.0 * np.eye(18),
            {
                0: [True, True, True, True, True, True],
                1: [True, True, True, True, True, True],
                2: [True, False, True, True, True, False],
            },
            3
        ],
        # 4) 4 nodes, diagonal SPD Ke, Kg = -I, free set: node 3 (uy, rz)
        [
            np.diag(np.linspace(1.0, 24.0, 24)),
            -np.eye(24),
            {
                0: [True, True, True, True, True, True],
                1: [True, True, True, True, True, True],
                2: [True, True, True, True, True, True],
                3: [True, False, True, True, True, False],
            },
            4
        ],
        # 5) 2 nodes, diagonal Ke, Kg = -3 I, free set: node 1 (ux, ry)
        [
            np.diag(np.linspace(5.0, 25.0, 12)),
            -3.0 * np.eye(12),
            {
                0: [True, True, True, True, True, True],
                1: [False, True, True, True, False, True],  # free: ux, ry
            },
            2
        ],
    ]
    test_cases = [{"test_code": test_eigen_known_answer, "expected_failures": [eigenvalue_analysis_bad_allzeros]},
                  {"test_code": test_eigen_singluar_detected, "expected_failures": [eigenvalue_analysis_bad_skip_checks]},
                  {"test_code": test_eigen_complex_eigenpairs_detected, "expected_failures": [eigenvalue_analysis_bad_skip_checks]},
                  {"test_code": test_eigen_no_positive_eigenvalues_detected, "expected_failures": [eigenvalue_analysis_bad_skip_checks]},
                  {"test_code": test_eigen_invariance_to_reference_load_scaling, "expected_failures": [eigenvalue_analysis_bad_wrong_scaling]}
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
