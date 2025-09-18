### Function Correctness (✓ = Match)

| Task                                                        | claude-3-5   | deepseek-chat   | gemini-flash   | gemini-pro   | gpt-4o   |
|:------------------------------------------------------------|:-------------|:----------------|:---------------|:-------------|:---------|
| assemble_global_geometric_stiffness_3D_beam                 | ✓            | ×               | ✓              | ✓            | ✓        |
| assemble_global_stiffness_matrix_linear_elastic_3D          | ✓            | ✓               | ✓              | ×            | ✓        |
| beam_transformation_matrix_3D                               | ×            | ×               | ×              | ×            | ✓        |
| compute_local_element_loads_beam_3D                         | ✓            | ✓               | ✓              | ×            | ✓        |
| eigenvalue_analysis_msa_3D                                  | ×            | ×               | ×              | ×            | ×        |
| elastic_critical_load_analysis_frame_3D                     | ✓            | ✓               | ×              | ×            | ✓        |
| elastic_critical_load_analysis_frame_3D_part_self_contained | ×            | ×               | ×              | ×            | ×        |
| elastic_critical_load_analysis_frame_3D_self_contained      | ×            | ×               | ×              | ×            | ×        |
| element_stiffness_linear_elastic_1D                         | ✓            | ✓               | ✓              | ×            | ✓        |
| linear_solve                                                | ✓            | ×               | ×              | ✓            | ✓        |
| linear_uniform_mesh_1D                                      | ✓            | ✓               | ✓              | ✓            | ✓        |
| local_elastic_stiffness_matrix_3D_beam                      | ✓            | ×               | ×              | ×            | ✓        |
| local_geometric_stiffness_matrix_3D_beam                    | ×            | ×               | ×              | ×            | ×        |
| solve_linear_elastic_1D_self_contained                      | ✓            | ✓               | ×              | ×            | ✓        |
| solve_linear_elastic_frame_3D                               | ✓            | ✓               | ✓              | ×            | ✓        |
| solve_linear_elastic_frame_3D_self_contained                | ✓            | ×               | ×              | ×            | ×        |
| Total                                                       | 11/16        | 7/16            | 6/16           | 3/16         | 11/16    |

### Joint Test Success Rate (%)

| Task                                                        | claude-3-5   | deepseek-chat   | gemini-flash   | gemini-pro   | gpt-4o   |
|:------------------------------------------------------------|:-------------|:----------------|:---------------|:-------------|:---------|
| assemble_global_geometric_stiffness_3D_beam                 | 0.0%         | 0.0%            | 0.0%           | 0.0%         | 0.0%     |
| assemble_global_stiffness_matrix_linear_elastic_3D          | 100.0%       | 0.0%            | 100.0%         | 0.0%         | 0.0%     |
| beam_transformation_matrix_3D                               | 33.3%        | 66.7%           | 33.3%          | 33.3%        | 33.3%    |
| compute_local_element_loads_beam_3D                         | 0.0%         | 75.0%           | –              | 0.0%         | 75.0%    |
| eigenvalue_analysis_msa_3D                                  | 20.0%        | 0.0%            | 0.0%           | 0.0%         | 0.0%     |
| elastic_critical_load_analysis_frame_3D                     | –            | 0.0%            | –              | –            | 0.0%     |
| elastic_critical_load_analysis_frame_3D_part_self_contained | –            | –               | –              | –            | 0.0%     |
| elastic_critical_load_analysis_frame_3D_self_contained      | 0.0%         | –               | –              | –            | 0.0%     |
| element_stiffness_linear_elastic_1D                         | 100.0%       | 100.0%          | 100.0%         | 100.0%       | 100.0%   |
| linear_solve                                                | 0.0%         | 0.0%            | 0.0%           | 0.0%         | 0.0%     |
| linear_uniform_mesh_1D                                      | 100.0%       | 100.0%          | 100.0%         | 100.0%       | 100.0%   |
| local_elastic_stiffness_matrix_3D_beam                      | 0.0%         | –               | 0.0%           | 100.0%       | 50.0%    |
| local_geometric_stiffness_matrix_3D_beam                    | 50.0%        | –               | 50.0%          | 0.0%         | 50.0%    |
| solve_linear_elastic_1D_self_contained                      | 0.0%         | –               | 50.0%          | 100.0%       | 50.0%    |
| solve_linear_elastic_frame_3D                               | 100.0%       | 100.0%          | 50.0%          | 0.0%         | 0.0%     |
| solve_linear_elastic_frame_3D_self_contained                | 66.7%        | –               | 33.3%          | 33.3%        | 33.3%    |
| Avg Joint Success %                                         | 35.6%        | 27.6%           | 32.3%          | 29.2%        | 30.7%    |

