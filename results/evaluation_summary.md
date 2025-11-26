### Function Correctness (✓ = Match)

| Task                                                      | gemini-2.5-pro   | gemini-3-pro-preview   |
|:----------------------------------------------------------|:-----------------|:-----------------------|
| FEM_1D_linear_elastic_CC0_H0_T0                           | ✓                | ✓                      |
| FEM_1D_local_elastic_stiffness_CC0_H3_T1                  | ✓                | ✓                      |
| FEM_1D_uniform_mesh_CC0_H0_T0                             | ✓                | ✓                      |
| FEM_2D_quad8_element_distributed_load_CC0_H0_T0           | ✓                | ✓                      |
| FEM_2D_quad8_integral_of_derivative_CC0_H3_T3             | ✓                | ✓                      |
| FEM_2D_quad8_mesh_rectangle_CC0_H0_T0                     | ✓                | ✓                      |
| FEM_2D_quad8_physical_gradient_CC0_H1_T3                  | ✓                | ✓                      |
| FEM_2D_quad8_shape_fcns_and_derivatives_CC0_H0_T0         | ✓                | ✓                      |
| FEM_2D_quad_quadrature_CC0_H0_T0                          | ✓                | ✓                      |
| FEM_2D_tri6_mesh_rectangle_CC0_H0_T0                      | ✓                | ✓                      |
| FEM_2D_tri6_shape_fcns_and_derivatives_CC0_H0_T0          | ✓                | ✓                      |
| FEM_2D_tri_quadrature_CC0_H0_T0                           | ✓                | ✓                      |
| MSA_3D_assemble_global_geometric_stiffness_CC1_H4_T1      | ✓                | ✓                      |
| MSA_3D_assemble_global_geometric_stiffness_CC1_H4_T2      | ×                | ×                      |
| MSA_3D_assemble_global_geometric_stiffness_CC1_H4_T3      | ×                | ×                      |
| MSA_3D_assemble_global_linear_elastic_stiffness_CC0_H2_T1 | ✓                | ✓                      |
| MSA_3D_assemble_global_linear_elastic_stiffness_CC0_H2_T3 | ×                | ✓                      |
| MSA_3D_assemble_global_load_CC0_H0_T0                     | ✓                | ✓                      |
| MSA_3D_elastic_critical_load_CC1_H10_T1                   | ✓                | ✓                      |
| MSA_3D_elastic_critical_load_CC1_H10_T2                   | ×                | ×                      |
| MSA_3D_elastic_critical_load_CC1_H10_T3                   | ×                | ×                      |
| MSA_3D_linear_elastic_CC0_H6_T1                           | ✓                | ✓                      |
| MSA_3D_linear_elastic_CC0_H6_T3                           | ✓                | ✓                      |
| MSA_3D_local_elastic_stiffness_CC0_H0_T0                  | ✓                | ✓                      |
| MSA_3D_local_element_loads_CC0_H2_T1                      | ✓                | ✓                      |
| MSA_3D_local_element_loads_CC0_H2_T3                      | ✓                | ✓                      |
| MSA_3D_local_geometric_stiffness_CC1_H0_T0                | ×                | ×                      |
| MSA_3D_partition_DOFs_CC0_H0_T0                           | ✓                | ✓                      |
| MSA_3D_solve_eigenvalue_CC1_H1_T1                         | ✓                | ✓                      |
| MSA_3D_solve_eigenvalue_CC1_H1_T3                         | ✓                | ✓                      |
| MSA_3D_solve_linear_CC0_H1_T1                             | ✓                | ✓                      |
| MSA_3D_solve_linear_CC0_H1_T3                             | ✓                | ✓                      |
| MSA_3D_transformation_matrix_CC0_H0_T0                    | ×                | ✓                      |
| Total                                                     | 26/33            | 28/33                  |

### Joint Test Success Rate (%)

| Task                                                      | gemini-2.5-pro   | gemini-3-pro-preview   |
|:----------------------------------------------------------|:-----------------|:-----------------------|
| FEM_1D_linear_elastic_CC0_H0_T0                           | 100.0%           | 100.0%                 |
| FEM_1D_local_elastic_stiffness_CC0_H3_T1                  | 100.0%           | 100.0%                 |
| FEM_1D_uniform_mesh_CC0_H0_T0                             | 100.0%           | 100.0%                 |
| FEM_2D_quad8_element_distributed_load_CC0_H0_T0           | 100.0%           | 100.0%                 |
| FEM_2D_quad8_integral_of_derivative_CC0_H3_T3             | 66.7%            | 66.7%                  |
| FEM_2D_quad8_mesh_rectangle_CC0_H0_T0                     | 66.7%            | 100.0%                 |
| FEM_2D_quad8_physical_gradient_CC0_H1_T3                  | 100.0%           | 100.0%                 |
| FEM_2D_quad8_shape_fcns_and_derivatives_CC0_H0_T0         | 83.3%            | 100.0%                 |
| FEM_2D_quad_quadrature_CC0_H0_T0                          | 100.0%           | 40.0%                  |
| FEM_2D_tri6_mesh_rectangle_CC0_H0_T0                      | 100.0%           | 66.7%                  |
| FEM_2D_tri6_shape_fcns_and_derivatives_CC0_H0_T0          | 100.0%           | 100.0%                 |
| FEM_2D_tri_quadrature_CC0_H0_T0                           | 40.0%            | 40.0%                  |
| MSA_3D_assemble_global_geometric_stiffness_CC1_H4_T1      | 50.0%            | 0.0%                   |
| MSA_3D_assemble_global_geometric_stiffness_CC1_H4_T2      | 0.0%             | 0.0%                   |
| MSA_3D_assemble_global_geometric_stiffness_CC1_H4_T3      | 0.0%             | 0.0%                   |
| MSA_3D_assemble_global_linear_elastic_stiffness_CC0_H2_T1 | 100.0%           | 100.0%                 |
| MSA_3D_assemble_global_linear_elastic_stiffness_CC0_H2_T3 | 100.0%           | 100.0%                 |
| MSA_3D_assemble_global_load_CC0_H0_T0                     | –                | 100.0%                 |
| MSA_3D_elastic_critical_load_CC1_H10_T1                   | 0.0%             | 0.0%                   |
| MSA_3D_elastic_critical_load_CC1_H10_T2                   | 0.0%             | 0.0%                   |
| MSA_3D_elastic_critical_load_CC1_H10_T3                   | 0.0%             | 0.0%                   |
| MSA_3D_linear_elastic_CC0_H6_T1                           | 100.0%           | 100.0%                 |
| MSA_3D_linear_elastic_CC0_H6_T3                           | 50.0%            | 100.0%                 |
| MSA_3D_local_elastic_stiffness_CC0_H0_T0                  | –                | 50.0%                  |
| MSA_3D_local_element_loads_CC0_H2_T1                      | 50.0%            | 100.0%                 |
| MSA_3D_local_element_loads_CC0_H2_T3                      | 100.0%           | 100.0%                 |
| MSA_3D_local_geometric_stiffness_CC1_H0_T0                | –                | 50.0%                  |
| MSA_3D_partition_DOFs_CC0_H0_T0                           | 100.0%           | 100.0%                 |
| MSA_3D_solve_eigenvalue_CC1_H1_T1                         | 40.0%            | 100.0%                 |
| MSA_3D_solve_eigenvalue_CC1_H1_T3                         | 80.0%            | 100.0%                 |
| MSA_3D_solve_linear_CC0_H1_T1                             | 100.0%           | 50.0%                  |
| MSA_3D_solve_linear_CC0_H1_T3                             | 100.0%           | 100.0%                 |
| MSA_3D_transformation_matrix_CC0_H0_T0                    | 66.7%            | 100.0%                 |
| Avg Joint Success %                                       | 63.4%            | 71.6%                  |

