### Function Correctness (✓ = Match)

| Task                                                        | gpt-4o   | gpt-5   | gemini-1.5-flash   | gemini-2.5-pro   | claude-3-5   | claude-sonnet-4   | claude-opus-4.1   | deepseek-chat   | deepseek-reasoner   |
|:------------------------------------------------------------|:---------|:--------|:-------------------|:-----------------|:-------------|:------------------|:------------------|:----------------|:--------------------|
| assemble_global_geometric_stiffness_3D_beam                 | ×        | ✓       | ×                  | ✓                | ✓            | ✓                 | ×                 | ✓               | ✓                   |
| assemble_global_stiffness_matrix_linear_elastic_3D          | ✓        | ✓       | ×                  | ✓                | ✓            | ✓                 | ✓                 | ✓               | ✓                   |
| beam_transformation_matrix_3D                               | ✓        | ×       | ×                  | ✓                | ×            | ✓                 | ✓                 | ×               | ×                   |
| compute_integral_of_derivative_quad8                        | ✓        | ✓       | ×                  | ✓                | ✓            | ✓                 | ✓                 | ✓               | ✓                   |
| compute_local_element_loads_beam_3D                         | ✓        | ×       | ×                  | ✓                | ✓            | ✓                 | ✓                 | ✓               | ✓                   |
| compute_physical_gradient_quad8                             | ×        | ✓       | ×                  | ×                | ×            | ✓                 | ✓                 | ✓               | ✓                   |
| eigenvalue_analysis_msa_3D                                  | ×        | ×       | ×                  | ×                | ×            | ×                 | ×                 | ×               | ×                   |
| elastic_critical_load_analysis_frame_3D                     | ✓        | ✓       | ×                  | ✓                | ✓            | ✓                 | ✓                 | ✓               | ✓                   |
| elastic_critical_load_analysis_frame_3D_part_self_contained | ×        | ×       | ×                  | ×                | ×            | ×                 | ×                 | ×               | ×                   |
| elastic_critical_load_analysis_frame_3D_self_contained      | ×        | ×       | ×                  | ×                | ×            | ×                 | ×                 | ×               | ×                   |
| element_distributed_load_quad8                              | ✓        | ✓       | ×                  | ✓                | ✓            | ✓                 | ✓                 | ×               | ×                   |
| element_stiffness_linear_elastic_1D                         | ✓        | ×       | ×                  | ✓                | ✓            | ✓                 | ✓                 | ✓               | ✓                   |
| generate_quad8_rectangular_mesh                             | ✓        | ✓       | ×                  | ✓                | ✓            | ✓                 | ✓                 | ✓               | ✓                   |
| generate_tri6_rectangular_mesh                              | ✓        | ✓       | ×                  | ✓                | ×            | ×                 | ✓                 | ✓               | ×                   |
| linear_solve                                                | ✓        | ✓       | ×                  | ✓                | ✓            | ×                 | ×                 | ×               | ×                   |
| linear_uniform_mesh_1D                                      | ✓        | ×       | ×                  | ×                | ✓            | ✓                 | ✓                 | ✓               | ✓                   |
| local_elastic_stiffness_matrix_3D_beam                      | ✓        | ✓       | ×                  | ✓                | ✓            | ✓                 | ✓                 | ✓               | ✓                   |
| local_geometric_stiffness_matrix_3D_beam                    | ×        | ×       | ×                  | ×                | ×            | ×                 | ×                 | ×               | ×                   |
| quad8_shape_functions_and_derivatives                       | ×        | ✓       | ×                  | ✓                | ×            | ×                 | ×                 | ✓               | ✓                   |
| quad_quadrature_2D                                          | ✓        | ✓       | ×                  | ✓                | ✓            | ×                 | ×                 | ×               | ✓                   |
| solve_linear_elastic_1D_self_contained                      | ✓        | ×       | ×                  | ×                | ✓            | ×                 | ✓                 | ×               | ×                   |
| solve_linear_elastic_frame_3D                               | ✓        | ✓       | ×                  | ✓                | ✓            | ✓                 | ✓                 | ✓               | ✓                   |
| solve_linear_elastic_frame_3D_self_contained                | ×        | ✓       | ×                  | ✓                | ×            | ×                 | ×                 | ×               | ✓                   |
| tri6_shape_functions_and_derivatives                        | ×        | ✓       | ×                  | ✓                | ×            | ×                 | ✓                 | ✓               | ✓                   |
| triangle_quadrature_2D                                      | ×        | ✓       | ×                  | ×                | ✓            | ✓                 | ✓                 | ✓               | ✓                   |
| Total                                                       | 15/25    | 16/25   | 0/25               | 17/25            | 15/25        | 14/25             | 16/25             | 15/25           | 16/25               |

### Joint Test Success Rate (%)

| Task                                                        | gpt-4o   | gpt-5   | gemini-1.5-flash   | gemini-2.5-pro   | claude-3-5   | claude-sonnet-4   | claude-opus-4.1   | deepseek-chat   | deepseek-reasoner   |
|:------------------------------------------------------------|:---------|:--------|:-------------------|:-----------------|:-------------|:------------------|:------------------|:----------------|:--------------------|
| assemble_global_geometric_stiffness_3D_beam                 | 100.0%   | 100.0%  | –                  | 0.0%             | 50.0%        | 100.0%            | 50.0%             | 100.0%          | 100.0%              |
| assemble_global_stiffness_matrix_linear_elastic_3D          | 0.0%     | 0.0%    | –                  | –                | 0.0%         | 0.0%              | 100.0%            | 0.0%            | 100.0%              |
| beam_transformation_matrix_3D                               | 33.3%    | 66.7%   | –                  | –                | 33.3%        | 66.7%             | 33.3%             | 33.3%           | 33.3%               |
| compute_integral_of_derivative_quad8                        | 0.0%     | 66.7%   | –                  | 66.7%            | –            | 66.7%             | 66.7%             | 66.7%           | 66.7%               |
| compute_local_element_loads_beam_3D                         | 75.0%    | 25.0%   | –                  | 75.0%            | 75.0%        | 100.0%            | 0.0%              | 75.0%           | 75.0%               |
| compute_physical_gradient_quad8                             | 0.0%     | 100.0%  | –                  | 100.0%           | 100.0%       | 100.0%            | 100.0%            | 100.0%          | 100.0%              |
| eigenvalue_analysis_msa_3D                                  | 0.0%     | 100.0%  | –                  | 0.0%             | 0.0%         | 60.0%             | 0.0%              | 0.0%            | 0.0%                |
| elastic_critical_load_analysis_frame_3D                     | 0.0%     | 0.0%    | –                  | 0.0%             | 0.0%         | –                 | –                 | –               | –                   |
| elastic_critical_load_analysis_frame_3D_part_self_contained | 0.0%     | 0.0%    | –                  | –                | 0.0%         | –                 | –                 | –               | –                   |
| elastic_critical_load_analysis_frame_3D_self_contained      | 0.0%     | 0.0%    | –                  | –                | 0.0%         | –                 | –                 | –               | –                   |
| element_distributed_load_quad8                              | 100.0%   | 100.0%  | –                  | 100.0%           | 100.0%       | 50.0%             | 100.0%            | 50.0%           | 100.0%              |
| element_stiffness_linear_elastic_1D                         | 100.0%   | 100.0%  | –                  | 100.0%           | 0.0%         | 100.0%            | 100.0%            | 100.0%          | 100.0%              |
| generate_quad8_rectangular_mesh                             | 100.0%   | 100.0%  | –                  | 100.0%           | 66.7%        | 100.0%            | 66.7%             | 66.7%           | 100.0%              |
| generate_tri6_rectangular_mesh                              | 66.7%    | 100.0%  | –                  | –                | 66.7%        | 100.0%            | –                 | 100.0%          | 66.7%               |
| linear_solve                                                | 0.0%     | 0.0%    | –                  | 100.0%           | 0.0%         | 50.0%             | 0.0%              | 0.0%            | 50.0%               |
| linear_uniform_mesh_1D                                      | 100.0%   | 100.0%  | –                  | 100.0%           | 100.0%       | 100.0%            | 100.0%            | 100.0%          | 100.0%              |
| local_elastic_stiffness_matrix_3D_beam                      | 0.0%     | 50.0%   | –                  | –                | 0.0%         | 0.0%              | –                 | –               | –                   |
| local_geometric_stiffness_matrix_3D_beam                    | 0.0%     | 100.0%  | –                  | –                | 50.0%        | –                 | –                 | –               | –                   |
| quad8_shape_functions_and_derivatives                       | 83.3%    | 100.0%  | –                  | –                | 83.3%        | 100.0%            | 66.7%             | 100.0%          | 83.3%               |
| quad_quadrature_2D                                          | 40.0%    | 100.0%  | –                  | –                | 60.0%        | 100.0%            | 100.0%            | 100.0%          | 100.0%              |
| solve_linear_elastic_1D_self_contained                      | 50.0%    | 100.0%  | –                  | 100.0%           | 100.0%       | 100.0%            | –                 | –               | 100.0%              |
| solve_linear_elastic_frame_3D                               | 0.0%     | 100.0%  | –                  | 50.0%            | 100.0%       | 50.0%             | 100.0%            | 50.0%           | 0.0%                |
| solve_linear_elastic_frame_3D_self_contained                | 33.3%    | 100.0%  | –                  | 100.0%           | 100.0%       | 66.7%             | 100.0%            | 33.3%           | 66.7%               |
| tri6_shape_functions_and_derivatives                        | 66.7%    | 100.0%  | –                  | –                | 50.0%        | 100.0%            | 100.0%            | 50.0%           | 50.0%               |
| triangle_quadrature_2D                                      | 60.0%    | 40.0%   | –                  | –                | 40.0%        | 40.0%             | 40.0%             | 60.0%           | 100.0%              |
| Avg Joint Success %                                         | 40.3%    | 69.9%   | 0.0%               | 39.7%            | 47.0%        | 62.0%             | 48.9%             | 47.4%           | 59.7%               |

