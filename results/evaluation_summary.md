### Function Correctness (✓ = Match)

| Task                                                        | claude-3-5   | deepseek-chat   | gemini-flash   | gemini-pro   | gpt-4o   |
|:------------------------------------------------------------|:-------------|:----------------|:---------------|:-------------|:---------|
| assemble_global_geometric_stiffness_3D_beam                 | ✓            | ×               | ✓              | ✓            | ✓        |
| assemble_global_stiffness_matrix_linear_elastic_3D          | ✓            | ✓               | ✓              | ×            | ✓        |
| beam_transformation_matrix_3D                               | ×            | ×               | ×              | ×            | ✓        |
| compute_local_element_loads_beam_3D                         | ✓            | ✓               | ✓              | ×            | ✓        |
| compute_physical_gradient_quad8                             | ×            | ×               | ×              | ×            | ×        |
| eigenvalue_analysis_msa_3D                                  | ×            | ×               | ×              | ×            | ×        |
| elastic_critical_load_analysis_frame_3D                     | ✓            | ✓               | ×              | ×            | ✓        |
| elastic_critical_load_analysis_frame_3D_part_self_contained | ×            | ×               | ×              | ×            | ×        |
| elastic_critical_load_analysis_frame_3D_self_contained      | ×            | ×               | ×              | ×            | ×        |
| element_stiffness_linear_elastic_1D                         | ✓            | ✓               | ✓              | ×            | ✓        |
| generate_quad8_rectangular_mesh                             | ✓            | ✓               | ×              | ×            | ✓        |
| generate_tri6_rectangular_mesh                              | ×            | ✓               | ×              | ×            | ×        |
| linear_solve                                                | ✓            | ×               | ×              | ✓            | ✓        |
| linear_uniform_mesh_1D                                      | ✓            | ✓               | ✓              | ✓            | ✓        |
| local_elastic_stiffness_matrix_3D_beam                      | ✓            | ×               | ×              | ×            | ✓        |
| local_geometric_stiffness_matrix_3D_beam                    | ×            | ×               | ×              | ×            | ×        |
| quad8_shape_functions_and_derivatives                       | ×            | ✓               | ✓              | ×            | ×        |
| quad_quadrature_2D                                          | ✓            | ✓               | ×              | ✓            | ✓        |
| solve_linear_elastic_1D_self_contained                      | ✓            | ✓               | ×              | ×            | ✓        |
| solve_linear_elastic_frame_3D                               | ✓            | ✓               | ✓              | ×            | ✓        |
| solve_linear_elastic_frame_3D_self_contained                | ✓            | ×               | ×              | ×            | ×        |
| tri6_shape_functions_and_derivatives                        | ×            | ✓               | ✓              | ×            | ×        |
| triangle_quadrature_2D                                      | ✓            | ×               | ×              | ✓            | ✓        |
| Total                                                       | 14/23        | 12/23           | 8/23           | 5/23         | 14/23    |

### Joint Test Success Rate (%)

| Task                                                        | claude-3-5   | deepseek-chat   | gemini-flash   | gemini-pro   | gpt-4o   |
|:------------------------------------------------------------|:-------------|:----------------|:---------------|:-------------|:---------|
| assemble_global_geometric_stiffness_3D_beam                 | 0.0%         | 0.0%            | 0.0%           | 0.0%         | 0.0%     |
| assemble_global_stiffness_matrix_linear_elastic_3D          | 100.0%       | 0.0%            | 100.0%         | 0.0%         | 0.0%     |
| beam_transformation_matrix_3D                               | 33.3%        | 66.7%           | 33.3%          | 33.3%        | 33.3%    |
| compute_local_element_loads_beam_3D                         | 0.0%         | 75.0%           | –              | 0.0%         | 75.0%    |
| compute_physical_gradient_quad8                             | 100.0%       | 100.0%          | 100.0%         | 100.0%       | 100.0%   |
| eigenvalue_analysis_msa_3D                                  | 20.0%        | 0.0%            | 0.0%           | 0.0%         | 0.0%     |
| elastic_critical_load_analysis_frame_3D                     | –            | 0.0%            | –              | –            | 0.0%     |
| elastic_critical_load_analysis_frame_3D_part_self_contained | –            | –               | –              | –            | 0.0%     |
| elastic_critical_load_analysis_frame_3D_self_contained      | 0.0%         | –               | –              | –            | 0.0%     |
| element_stiffness_linear_elastic_1D                         | 100.0%       | 100.0%          | 100.0%         | 100.0%       | 100.0%   |
| generate_quad8_rectangular_mesh                             | 100.0%       | 100.0%          | –              | 33.3%        | 66.7%    |
| generate_tri6_rectangular_mesh                              | 66.7%        | 33.3%           | –              | 100.0%       | 66.7%    |
| linear_solve                                                | 0.0%         | 0.0%            | 0.0%           | 0.0%         | 0.0%     |
| linear_uniform_mesh_1D                                      | 100.0%       | 100.0%          | 100.0%         | 100.0%       | 100.0%   |
| local_elastic_stiffness_matrix_3D_beam                      | 0.0%         | –               | 0.0%           | 100.0%       | 50.0%    |
| local_geometric_stiffness_matrix_3D_beam                    | 50.0%        | –               | 50.0%          | 0.0%         | 50.0%    |
| quad8_shape_functions_and_derivatives                       | 66.7%        | 66.7%           | –              | –            | 83.3%    |
| quad_quadrature_2D                                          | 60.0%        | –               | –              | 60.0%        | 40.0%    |
| solve_linear_elastic_1D_self_contained                      | 0.0%         | –               | 50.0%          | 100.0%       | 50.0%    |
| solve_linear_elastic_frame_3D                               | 100.0%       | 100.0%          | 50.0%          | 0.0%         | 0.0%     |
| solve_linear_elastic_frame_3D_self_contained                | 66.7%        | –               | 33.3%          | 33.3%        | 33.3%    |
| tri6_shape_functions_and_derivatives                        | 50.0%        | 66.7%           | 50.0%          | 33.3%        | 83.3%    |
| triangle_quadrature_2D                                      | 40.0%        | –               | 20.0%          | 40.0%        | 60.0%    |
| Avg Joint Success %                                         | 45.8%        | 35.1%           | 29.9%          | 36.2%        | 43.1%    |

