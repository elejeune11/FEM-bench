def MSA_3D_partition_DOFs_CC0_H0_T0(boundary_conditions: dict, n_nodes: int):
    all_dofs = np.arange(6 * n_nodes)
    fixed_dofs = []
    for node_idx in range(n_nodes):
        if node_idx in boundary_conditions:
            bc = boundary_conditions[node_idx]
            for local_dof in range(6):
                if bc[local_dof]:
                    dof_index = 6 * node_idx + local_dof
                    fixed_dofs.append(dof_index)
    fixed = np.sort(np.unique(fixed_dofs))
    free = np.setdiff1d(all_dofs, fixed, assume_unique=False)
    return (fixed, free)