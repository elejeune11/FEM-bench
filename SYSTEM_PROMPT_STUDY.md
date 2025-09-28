# FEM-bench preliminary system prompts study

## Table of Contents
* [Results with no system prompt](#no)
* [Results with System Prompt 1](#s1)
* [Results with System Prompt 2](#s2)
* [Summay of Results](#summary)

## Results with no system prompt, temperature = 0 <a name="no"></a>

### Function Correctness (✓ = Match)

| Task                                                        | gpt-4o   | gpt-5   | gemini-1.5-flash   | gemini-2.5-pro   | claude-3-5   | claude-sonnet-4   | claude-opus-4.1   | deepseek-chat   | deepseek-reasoner   |
|:------------------------------------------------------------|:---------|:--------|:-------------------|:-----------------|:-------------|:------------------|:------------------|:----------------|:--------------------|
| assemble_global_geometric_stiffness_3D_beam                 | ✓        | ✓       | ✓                  | ✓                | ✓            | ✓                 | ×                 | ×               | ✓                   |
| assemble_global_stiffness_matrix_linear_elastic_3D          | ✓        | ✓       | ✓                  | ✓                | ✓            | ✓                 | ✓                 | ✓               | ✓                   |
| beam_transformation_matrix_3D                               | ✓        | ×       | ×                  | ×                | ×            | ✓                 | ✓                 | ×               | ×                   |
| compute_integral_of_derivative_quad8                        | ✓        | ✓       | ×                  | ✓                | ✓            | ✓                 | ✓                 | ✓               | ✓                   |
| compute_local_element_loads_beam_3D                         | ✓        | ×       | ✓                  | ✓                | ✓            | ✓                 | ✓                 | ✓               | ✓                   |
| compute_physical_gradient_quad8                             | ×        | ✓       | ×                  | ✓                | ×            | ×                 | ✓                 | ×               | ✓                   |
| eigenvalue_analysis_msa_3D                                  | ×        | ×       | ×                  | ×                | ×            | ×                 | ×                 | ×               | ×                   |
| elastic_critical_load_analysis_frame_3D                     | ✓        | ✓       | ✓                  | ✓                | ✓            | ✓                 | ✓                 | ✓               | ✓                   |
| elastic_critical_load_analysis_frame_3D_part_self_contained | ×        | ×       | ×                  | ×                | ×            | ×                 | ×                 | ×               | ×                   |
| elastic_critical_load_analysis_frame_3D_self_contained      | ×        | ×       | ×                  | ×                | ×            | ×                 | ×                 | ×               | ×                   |
| element_distributed_load_quad8                              | ✓        | ✓       | ×                  | ✓                | ✓            | ✓                 | ✓                 | ×               | ✓                   |
| element_stiffness_linear_elastic_1D                         | ✓        | ×       | ✓                  | ✓                | ✓            | ✓                 | ✓                 | ✓               | ✓                   |
| generate_quad8_rectangular_mesh                             | ✓        | ✓       | ×                  | ✓                | ✓            | ✓                 | ✓                 | ✓               | ✓                   |
| generate_tri6_rectangular_mesh                              | ×        | ✓       | ×                  | ✓                | ×            | ✓                 | ✓                 | ✓               | ✓                   |
| linear_solve                                                | ✓        | ✓       | ×                  | ×                | ✓            | ×                 | ×                 | ×               | ×                   |
| linear_uniform_mesh_1D                                      | ✓        | ✓       | ✓                  | ✓                | ✓            | ✓                 | ✓                 | ✓               | ✓                   |
| local_elastic_stiffness_matrix_3D_beam                      | ✓        | ✓       | ×                  | ✓                | ✓            | ✓                 | ✓                 | ×               | ✓                   |
| local_geometric_stiffness_matrix_3D_beam                    | ×        | ×       | ×                  | ×                | ×            | ×                 | ×                 | ×               | ×                   |
| quad8_shape_functions_and_derivatives                       | ×        | ✓       | ✓                  | ✓                | ×            | ×                 | ×                 | ✓               | ✓                   |
| quad_quadrature_2D                                          | ✓        | ✓       | ×                  | ✓                | ✓            | ×                 | ×                 | ✓               | ×                   |
| solve_linear_elastic_1D_self_contained                      | ✓        | ×       | ×                  | ✓                | ✓            | ×                 | ✓                 | ✓               | ×                   |
| solve_linear_elastic_frame_3D                               | ✓        | ✓       | ✓                  | ✓                | ✓            | ✓                 | ✓                 | ✓               | ✓                   |
| solve_linear_elastic_frame_3D_self_contained                | ×        | ×       | ×                  | ✓                | ✓            | ×                 | ×                 | ×               | ×                   |
| tri6_shape_functions_and_derivatives                        | ×        | ✓       | ✓                  | ✓                | ×            | ×                 | ✓                 | ✓               | ✓                   |
| triangle_quadrature_2D                                      | ✓        | ✓       | ×                  | ✓                | ✓            | ✓                 | ✓                 | ×               | ✓                   |
| Total                                                       | 16/25    | 16/25   | 9/25               | 19/25            | 16/25        | 14/25             | 16/25             | 13/25           | 16/25               |

### Joint Test Success Rate (%)

| Task                                                        | gpt-4o   | gpt-5   | gemini-1.5-flash   | gemini-2.5-pro   | claude-3-5   | claude-sonnet-4   | claude-opus-4.1   | deepseek-chat   | deepseek-reasoner   |
|:------------------------------------------------------------|:---------|:--------|:-------------------|:-----------------|:-------------|:------------------|:------------------|:----------------|:--------------------|
| assemble_global_geometric_stiffness_3D_beam                 | 0.0%     | 50.0%   | 50.0%              | 0.0%             | 50.0%        | 0.0%              | 50.0%             | 50.0%           | 50.0%               |
| assemble_global_stiffness_matrix_linear_elastic_3D          | 0.0%     | 0.0%    | 100.0%             | 100.0%           | 100.0%       | 100.0%            | 100.0%            | 0.0%            | 0.0%                |
| beam_transformation_matrix_3D                               | 33.3%    | 33.3%   | 33.3%              | 0.0%             | 33.3%        | 33.3%             | 33.3%             | 66.7%           | 33.3%               |
| compute_integral_of_derivative_quad8                        | 33.3%    | 66.7%   | 33.3%              | 66.7%            | 66.7%        | 66.7%             | 66.7%             | 66.7%           | 66.7%               |
| compute_local_element_loads_beam_3D                         | 50.0%    | 50.0%   | 0.0%               | 50.0%            | 0.0%         | 50.0%             | 0.0%              | 75.0%           | 75.0%               |
| compute_physical_gradient_quad8                             | 100.0%   | 100.0%  | 50.0%              | 100.0%           | 100.0%       | 100.0%            | 100.0%            | 100.0%          | 100.0%              |
| eigenvalue_analysis_msa_3D                                  | 0.0%     | 0.0%    | 0.0%               | 0.0%             | 20.0%        | 0.0%              | 0.0%              | 0.0%            | 0.0%                |
| elastic_critical_load_analysis_frame_3D                     | 0.0%     | 0.0%    | –                  | 0.0%             | –            | –                 | –                 | 0.0%            | –                   |
| elastic_critical_load_analysis_frame_3D_part_self_contained | 0.0%     | 0.0%    | –                  | –                | –            | –                 | –                 | –               | –                   |
| elastic_critical_load_analysis_frame_3D_self_contained      | 0.0%     | 0.0%    | –                  | –                | 0.0%         | –                 | –                 | –               | –                   |
| element_distributed_load_quad8                              | 100.0%   | 50.0%   | 50.0%              | 100.0%           | 0.0%         | 50.0%             | 100.0%            | 50.0%           | 50.0%               |
| element_stiffness_linear_elastic_1D                         | 100.0%   | 100.0%  | 100.0%             | 100.0%           | 100.0%       | 100.0%            | 100.0%            | 100.0%          | 100.0%              |
| generate_quad8_rectangular_mesh                             | 66.7%    | 100.0%  | 66.7%              | 100.0%           | 100.0%       | 100.0%            | 100.0%            | 100.0%          | 33.3%               |
| generate_tri6_rectangular_mesh                              | 66.7%    | 100.0%  | –                  | 100.0%           | 66.7%        | 100.0%            | –                 | 33.3%           | 100.0%              |
| linear_solve                                                | 0.0%     | 50.0%   | 0.0%               | 0.0%             | 0.0%         | 50.0%             | 0.0%              | 0.0%            | 0.0%                |
| linear_uniform_mesh_1D                                      | 100.0%   | 100.0%  | 100.0%             | 0.0%             | 100.0%       | 100.0%            | 100.0%            | 100.0%          | 100.0%              |
| local_elastic_stiffness_matrix_3D_beam                      | 50.0%    | 50.0%   | 0.0%               | –                | 0.0%         | 0.0%              | –                 | –               | 50.0%               |
| local_geometric_stiffness_matrix_3D_beam                    | 50.0%    | 100.0%  | 50.0%              | 50.0%            | 50.0%        | –                 | –                 | –               | –                   |
| quad8_shape_functions_and_derivatives                       | 83.3%    | 100.0%  | –                  | –                | 66.7%        | 100.0%            | 100.0%            | 66.7%           | 83.3%               |
| quad_quadrature_2D                                          | 40.0%    | 100.0%  | 40.0%              | –                | 60.0%        | 100.0%            | 100.0%            | –               | 100.0%              |
| solve_linear_elastic_1D_self_contained                      | 50.0%    | 100.0%  | 50.0%              | 100.0%           | 0.0%         | 100.0%            | –                 | –               | 100.0%              |
| solve_linear_elastic_frame_3D                               | 0.0%     | 100.0%  | 0.0%               | 50.0%            | 100.0%       | 50.0%             | 100.0%            | 100.0%          | 50.0%               |
| solve_linear_elastic_frame_3D_self_contained                | 33.3%    | 100.0%  | 0.0%               | 100.0%           | 66.7%        | 66.7%             | 100.0%            | –               | 33.3%               |
| tri6_shape_functions_and_derivatives                        | 83.3%    | 83.3%   | 50.0%              | 16.7%            | 50.0%        | 100.0%            | 100.0%            | 66.7%           | 50.0%               |
| triangle_quadrature_2D                                      | 60.0%    | 100.0%  | 20.0%              | –                | 40.0%        | 40.0%             | 40.0%             | –               | 100.0%              |
| Avg Joint Success %                                         | 44.0%    | 65.3%   | 31.7%              | 41.3%            | 46.8%        | 56.3%             | 51.6%             | 39.0%           | 51.0%               |


## Results with system prompt 1, temperature = 0 <a name="s1"></a>

System prompt:
```
You are an expert Python engineer in scientific computing, with a specialization in Finite Element Analysis (FEM).
Before responding, reason privately to ensure mathematical correctness and edge-case coverage, but do not output your reasoning.
Always return only executable Python code—no commentary, markdown fences, or extra text.
Follow the user’s instructions exactly; if there is any conflict, prefer the user’s explicit task rules.
Never add imports beyond those specified, and never reimplement or alter helper functions.
If the task is to write tests, output only pytest test functions.
Be precise, deterministic, and correctness-focused.
```

### Function Correctness (✓ = Match)

| Task                                                        | gpt-4o   | gpt-5   | gemini-1.5-flash   | gemini-2.5-pro   | claude-3-5   | claude-sonnet-4   | claude-opus-4.1   | deepseek-chat   | deepseek-reasoner   |
|:------------------------------------------------------------|:---------|:--------|:-------------------|:-----------------|:-------------|:------------------|:------------------|:----------------|:--------------------|
| assemble_global_geometric_stiffness_3D_beam                 | ✓        | ✓       | ✓                  | ✓                | ✓            | ✓                 | ×                 | ✓               | ✓                   |
| assemble_global_stiffness_matrix_linear_elastic_3D          | ✓        | ✓       | ✓                  | ✓                | ✓            | ✓                 | ✓                 | ✓               | ✓                   |
| beam_transformation_matrix_3D                               | ✓        | ✓       | ×                  | ✓                | ✓            | ✓                 | ✓                 | ×               | ×                   |
| compute_integral_of_derivative_quad8                        | ✓        | ✓       | ×                  | ✓                | ✓            | ✓                 | ✓                 | ✓               | ×                   |
| compute_local_element_loads_beam_3D                         | ✓        | ✓       | ✓                  | ✓                | ✓            | ✓                 | ✓                 | ✓               | ✓                   |
| compute_physical_gradient_quad8                             | ×        | ✓       | ×                  | ✓                | ×            | ×                 | ×                 | ×               | ✓                   |
| eigenvalue_analysis_msa_3D                                  | ×        | ×       | ×                  | ×                | ×            | ×                 | ×                 | ×               | ×                   |
| elastic_critical_load_analysis_frame_3D                     | ✓        | ✓       | ✓                  | ✓                | ✓            | ✓                 | ✓                 | ✓               | ✓                   |
| elastic_critical_load_analysis_frame_3D_part_self_contained | ×        | ×       | ×                  | ×                | ×            | ×                 | ×                 | ×               | ×                   |
| elastic_critical_load_analysis_frame_3D_self_contained      | ×        | ×       | ×                  | ×                | ×            | ×                 | ×                 | ×               | ×                   |
| element_distributed_load_quad8                              | ✓        | ✓       | ×                  | ✓                | ✓            | ✓                 | ✓                 | ×               | ✓                   |
| element_stiffness_linear_elastic_1D                         | ✓        | ×       | ✓                  | ×                | ✓            | ✓                 | ✓                 | ✓               | ✓                   |
| generate_quad8_rectangular_mesh                             | ✓        | ✓       | ×                  | ✓                | ✓            | ✓                 | ✓                 | ✓               | ✓                   |
| generate_tri6_rectangular_mesh                              | ×        | ✓       | ×                  | ✓                | ×            | ✓                 | ✓                 | ✓               | ✓                   |
| linear_solve                                                | ✓        | ✓       | ×                  | ✓                | ×            | ×                 | ×                 | ×               | ×                   |
| linear_uniform_mesh_1D                                      | ✓        | ✓       | ✓                  | ✓                | ✓            | ✓                 | ✓                 | ✓               | ✓                   |
| local_elastic_stiffness_matrix_3D_beam                      | ✓        | ✓       | ×                  | ✓                | ✓            | ✓                 | ✓                 | ✓               | ×                   |
| local_geometric_stiffness_matrix_3D_beam                    | ×        | ×       | ×                  | ×                | ×            | ×                 | ×                 | ×               | ×                   |
| quad8_shape_functions_and_derivatives                       | ×        | ✓       | ×                  | ✓                | ×            | ✓                 | ×                 | ✓               | ✓                   |
| quad_quadrature_2D                                          | ✓        | ×       | ×                  | ×                | ✓            | ×                 | ✓                 | ✓               | ×                   |
| solve_linear_elastic_1D_self_contained                      | ✓        | ×       | ×                  | ×                | ✓            | ×                 | ✓                 | ✓               | ✓                   |
| solve_linear_elastic_frame_3D                               | ✓        | ✓       | ✓                  | ✓                | ✓            | ✓                 | ✓                 | ✓               | ✓                   |
| solve_linear_elastic_frame_3D_self_contained                | ×        | ✓       | ×                  | ×                | ×            | ×                 | ×                 | ✓               | ×                   |
| tri6_shape_functions_and_derivatives                        | ×        | ✓       | ✓                  | ✓                | ✓            | ×                 | ✓                 | ✓               | ✓                   |
| triangle_quadrature_2D                                      | ✓        | ✓       | ×                  | ✓                | ✓            | ✓                 | ✓                 | ✓               | ✓                   |
| Total                                                       | 16/25    | 18/25   | 8/25               | 17/25            | 16/25        | 15/25             | 16/25             | 17/25           | 15/25               |

### Joint Test Success Rate (%)

| Task                                                        | gpt-4o   | gpt-5   | gemini-1.5-flash   | gemini-2.5-pro   | claude-3-5   | claude-sonnet-4   | claude-opus-4.1   | deepseek-chat   | deepseek-reasoner   |
|:------------------------------------------------------------|:---------|:--------|:-------------------|:-----------------|:-------------|:------------------|:------------------|:----------------|:--------------------|
| assemble_global_geometric_stiffness_3D_beam                 | 0.0%     | 0.0%    | 0.0%               | 0.0%             | 0.0%         | 0.0%              | 0.0%              | 0.0%            | 0.0%                |
| assemble_global_stiffness_matrix_linear_elastic_3D          | 0.0%     | 100.0%  | 0.0%               | 100.0%           | 100.0%       | 100.0%            | 100.0%            | 0.0%            | 0.0%                |
| beam_transformation_matrix_3D                               | 33.3%    | 100.0%  | 33.3%              | 0.0%             | 33.3%        | 33.3%             | –                 | 33.3%           | 33.3%               |
| compute_integral_of_derivative_quad8                        | 33.3%    | 66.7%   | 33.3%              | 66.7%            | 66.7%        | 66.7%             | 66.7%             | 66.7%           | 66.7%               |
| compute_local_element_loads_beam_3D                         | 75.0%    | 100.0%  | 0.0%               | 100.0%           | 0.0%         | 75.0%             | 75.0%             | 25.0%           | 100.0%              |
| compute_physical_gradient_quad8                             | 100.0%   | 100.0%  | 100.0%             | 100.0%           | 100.0%       | 100.0%            | 100.0%            | 100.0%          | 100.0%              |
| eigenvalue_analysis_msa_3D                                  | 0.0%     | 0.0%    | 0.0%               | 0.0%             | 20.0%        | 0.0%              | 0.0%              | 0.0%            | 0.0%                |
| elastic_critical_load_analysis_frame_3D                     | 0.0%     | 0.0%    | –                  | –                | 0.0%         | –                 | –                 | –               | –                   |
| elastic_critical_load_analysis_frame_3D_part_self_contained | 0.0%     | 0.0%    | –                  | –                | 0.0%         | –                 | –                 | –               | –                   |
| elastic_critical_load_analysis_frame_3D_self_contained      | 0.0%     | 0.0%    | –                  | –                | 0.0%         | –                 | –                 | –               | –                   |
| element_distributed_load_quad8                              | 100.0%   | 50.0%   | 50.0%              | 50.0%            | 100.0%       | 50.0%             | 100.0%            | 50.0%           | 50.0%               |
| element_stiffness_linear_elastic_1D                         | 100.0%   | 100.0%  | 100.0%             | 100.0%           | 0.0%         | 100.0%            | 100.0%            | 100.0%          | 100.0%              |
| generate_quad8_rectangular_mesh                             | 66.7%    | 100.0%  | 0.0%               | 66.7%            | 66.7%        | 100.0%            | 66.7%             | 33.3%           | –                   |
| generate_tri6_rectangular_mesh                              | 66.7%    | 100.0%  | 0.0%               | 100.0%           | 100.0%       | 66.7%             | 100.0%            | 100.0%          | –                   |
| linear_solve                                                | 0.0%     | 50.0%   | 0.0%               | 50.0%            | 0.0%         | 0.0%              | 50.0%             | 0.0%            | 50.0%               |
| linear_uniform_mesh_1D                                      | 100.0%   | 100.0%  | 100.0%             | 100.0%           | 100.0%       | 100.0%            | 100.0%            | 100.0%          | 100.0%              |
| local_elastic_stiffness_matrix_3D_beam                      | 0.0%     | 50.0%   | 0.0%               | –                | –            | –                 | –                 | 0.0%            | –                   |
| local_geometric_stiffness_matrix_3D_beam                    | 50.0%    | 50.0%   | 50.0%              | –                | 50.0%        | –                 | –                 | –               | –                   |
| quad8_shape_functions_and_derivatives                       | 66.7%    | 100.0%  | 33.3%              | –                | –            | 100.0%            | 100.0%            | 100.0%          | 100.0%              |
| quad_quadrature_2D                                          | 40.0%    | 100.0%  | 60.0%              | 100.0%           | 60.0%        | 80.0%             | 100.0%            | 100.0%          | 100.0%              |
| solve_linear_elastic_1D_self_contained                      | 50.0%    | 100.0%  | 100.0%             | –                | 100.0%       | 100.0%            | –                 | 100.0%          | 100.0%              |
| solve_linear_elastic_frame_3D                               | 0.0%     | 100.0%  | 0.0%               | 100.0%           | 100.0%       | 50.0%             | 100.0%            | 50.0%           | 50.0%               |
| solve_linear_elastic_frame_3D_self_contained                | 33.3%    | 100.0%  | 0.0%               | 100.0%           | 33.3%        | 66.7%             | 100.0%            | 33.3%           | 33.3%               |
| tri6_shape_functions_and_derivatives                        | 83.3%    | 100.0%  | 33.3%              | 100.0%           | –            | 100.0%            | 100.0%            | 66.7%           | 50.0%               |
| triangle_quadrature_2D                                      | 100.0%   | 100.0%  | 20.0%              | 100.0%           | 40.0%        | 40.0%             | 100.0%            | 40.0%           | 80.0%               |
| Avg Joint Success %                                         | 43.9%    | 70.7%   | 28.5%              | 53.3%            | 42.8%        | 53.1%             | 58.3%             | 43.9%           | 44.5%               |

## Results with System Prompt 2, temperature = 0 <a name="s2"></a>

System prompt:
```
You are an expert in finite element analysis and scientific computing. You completed your PhD under Tom Hughes and have spent over 10 years at Sandia National Laboratories working on computational mechanics problems.
Focus on producing robust, correct, production-quality Python code. Your solutions should demonstrate both mathematical rigor and practical engineering judgment.
Output only executable Python code—no markdown, comments, or extra text.
Follow the user's task rules exactly: match the given function signatures and docstrings, respect import limits, and never alter helper functions.
If the task is to write tests, output only pytest tests with meaningful assertions.
```

### Function Correctness (✓ = Match)

| Task                                                        | gpt-4o   | gpt-5   | gemini-1.5-flash   | gemini-2.5-pro   | claude-3-5   | claude-sonnet-4   | claude-opus-4.1   | deepseek-chat   | deepseek-reasoner   |
|:------------------------------------------------------------|:---------|:--------|:-------------------|:-----------------|:-------------|:------------------|:------------------|:----------------|:--------------------|
| assemble_global_geometric_stiffness_3D_beam                 | ✓        | ✓       | ×                  | ✓                | ✓            | ✓                 | ×                 | ✓               | ✓                   |
| assemble_global_stiffness_matrix_linear_elastic_3D          | ✓        | ✓       | ×                  | ✓                | ✓            | ✓                 | ✓                 | ✓               | ✓                   |
| beam_transformation_matrix_3D                               | ✓        | ✓       | ×                  | ✓                | ✓            | ×                 | ✓                 | ×               | ×                   |
| compute_integral_of_derivative_quad8                        | ✓        | ✓       | ×                  | ✓                | ✓            | ✓                 | ✓                 | ✓               | ✓                   |
| compute_local_element_loads_beam_3D                         | ✓        | ×       | ×                  | ✓                | ✓            | ✓                 | ✓                 | ✓               | ✓                   |
| compute_physical_gradient_quad8                             | ×        | ✓       | ×                  | ×                | ✓            | ×                 | ×                 | ×               | ×                   |
| eigenvalue_analysis_msa_3D                                  | ×        | ×       | ×                  | ×                | ×            | ×                 | ×                 | ×               | ×                   |
| elastic_critical_load_analysis_frame_3D                     | ✓        | ✓       | ×                  | ✓                | ✓            | ✓                 | ✓                 | ✓               | ✓                   |
| elastic_critical_load_analysis_frame_3D_part_self_contained | ×        | ×       | ×                  | ×                | ×            | ×                 | ×                 | ×               | ×                   |
| elastic_critical_load_analysis_frame_3D_self_contained      | ×        | ×       | ×                  | ×                | ×            | ×                 | ×                 | ×               | ×                   |
| element_distributed_load_quad8                              | ×        | ✓       | ×                  | ✓                | ×            | ✓                 | ✓                 | ✓               | ✓                   |
| element_stiffness_linear_elastic_1D                         | ✓        | ×       | ×                  | ×                | ✓            | ✓                 | ✓                 | ✓               | ✓                   |
| generate_quad8_rectangular_mesh                             | ✓        | ✓       | ×                  | ✓                | ✓            | ✓                 | ✓                 | ✓               | ✓                   |
| generate_tri6_rectangular_mesh                              | ×        | ✓       | ×                  | ✓                | ✓            | ✓                 | ✓                 | ✓               | ✓                   |
| linear_solve                                                | ✓        | ✓       | ×                  | ✓                | ✓            | ×                 | ✓                 | ×               | ✓                   |
| linear_uniform_mesh_1D                                      | ✓        | ×       | ×                  | ✓                | ✓            | ✓                 | ✓                 | ✓               | ✓                   |
| local_elastic_stiffness_matrix_3D_beam                      | ✓        | ✓       | ×                  | ✓                | ✓            | ✓                 | ✓                 | ×               | ✓                   |
| local_geometric_stiffness_matrix_3D_beam                    | ×        | ×       | ×                  | ×                | ×            | ×                 | ×                 | ×               | ×                   |
| quad8_shape_functions_and_derivatives                       | ×        | ✓       | ×                  | ✓                | ×            | ✓                 | ×                 | ×               | ✓                   |
| quad_quadrature_2D                                          | ✓        | ✓       | ×                  | ×                | ✓            | ✓                 | ✓                 | ×               | ×                   |
| solve_linear_elastic_1D_self_contained                      | ×        | ✓       | ×                  | ×                | ✓            | ×                 | ✓                 | ✓               | ✓                   |
| solve_linear_elastic_frame_3D                               | ✓        | ✓       | ×                  | ✓                | ✓            | ✓                 | ✓                 | ✓               | ✓                   |
| solve_linear_elastic_frame_3D_self_contained                | ×        | ✓       | ×                  | ✓                | ✓            | ✓                 | ✓                 | ×               | ✓                   |
| tri6_shape_functions_and_derivatives                        | ×        | ✓       | ×                  | ✓                | ✓            | ×                 | ✓                 | ✓               | ✓                   |
| triangle_quadrature_2D                                      | ×        | ✓       | ×                  | ✓                | ✓            | ✓                 | ✓                 | ✓               | ×                   |
| Total                                                       | 13/25    | 18/25   | 0/25               | 17/25            | 19/25        | 16/25             | 18/25             | 14/25           | 17/25               |

### Joint Test Success Rate (%)

| Task                                                        | gpt-4o   | gpt-5   | gemini-1.5-flash   | gemini-2.5-pro   | claude-3-5   | claude-sonnet-4   | claude-opus-4.1   | deepseek-chat   | deepseek-reasoner   |
|:------------------------------------------------------------|:---------|:--------|:-------------------|:-----------------|:-------------|:------------------|:------------------|:----------------|:--------------------|
| assemble_global_geometric_stiffness_3D_beam                 | 0.0%     | 0.0%    | –                  | 0.0%             | 0.0%         | 0.0%              | 0.0%              | 0.0%            | 0.0%                |
| assemble_global_stiffness_matrix_linear_elastic_3D          | 0.0%     | 100.0%  | –                  | 100.0%           | 0.0%         | 100.0%            | 100.0%            | 0.0%            | 0.0%                |
| beam_transformation_matrix_3D                               | 33.3%    | 100.0%  | –                  | 0.0%             | 0.0%         | 33.3%             | –                 | 33.3%           | 66.7%               |
| compute_integral_of_derivative_quad8                        | 0.0%     | 66.7%   | –                  | 66.7%            | 66.7%        | 66.7%             | 66.7%             | 66.7%           | 66.7%               |
| compute_local_element_loads_beam_3D                         | 75.0%    | 75.0%   | –                  | 75.0%            | 0.0%         | 75.0%             | 50.0%             | 75.0%           | 75.0%               |
| compute_physical_gradient_quad8                             | 100.0%   | 100.0%  | –                  | 100.0%           | 0.0%         | 100.0%            | 100.0%            | 100.0%          | 100.0%              |
| eigenvalue_analysis_msa_3D                                  | 0.0%     | 0.0%    | –                  | 0.0%             | 60.0%        | 0.0%              | 0.0%              | 0.0%            | 0.0%                |
| elastic_critical_load_analysis_frame_3D                     | 0.0%     | 0.0%    | –                  | –                | 0.0%         | –                 | –                 | –               | –                   |
| elastic_critical_load_analysis_frame_3D_part_self_contained | 0.0%     | 0.0%    | –                  | –                | 0.0%         | –                 | –                 | –               | –                   |
| elastic_critical_load_analysis_frame_3D_self_contained      | 0.0%     | 0.0%    | –                  | –                | 0.0%         | –                 | –                 | –               | –                   |
| element_distributed_load_quad8                              | 100.0%   | 50.0%   | –                  | 50.0%            | 0.0%         | 50.0%             | 100.0%            | 50.0%           | 0.0%                |
| element_stiffness_linear_elastic_1D                         | 100.0%   | 100.0%  | –                  | 100.0%           | 0.0%         | 100.0%            | 100.0%            | 100.0%          | 100.0%              |
| generate_quad8_rectangular_mesh                             | 66.7%    | 100.0%  | –                  | –                | 66.7%        | 100.0%            | 66.7%             | 66.7%           | 66.7%               |
| generate_tri6_rectangular_mesh                              | 66.7%    | 100.0%  | –                  | 100.0%           | 100.0%       | 100.0%            | –                 | –               | 66.7%               |
| linear_solve                                                | 0.0%     | 50.0%   | –                  | 50.0%            | 0.0%         | 50.0%             | 50.0%             | 0.0%            | 0.0%                |
| linear_uniform_mesh_1D                                      | 100.0%   | 100.0%  | –                  | 100.0%           | 100.0%       | 100.0%            | 100.0%            | 100.0%          | 100.0%              |
| local_elastic_stiffness_matrix_3D_beam                      | 0.0%     | 50.0%   | –                  | –                | 0.0%         | –                 | –                 | –               | –                   |
| local_geometric_stiffness_matrix_3D_beam                    | 0.0%     | 100.0%  | –                  | –                | 0.0%         | –                 | –                 | –               | –                   |
| quad8_shape_functions_and_derivatives                       | 66.7%    | 66.7%   | –                  | 100.0%           | 83.3%        | 100.0%            | 100.0%            | 83.3%           | 66.7%               |
| quad_quadrature_2D                                          | 60.0%    | 100.0%  | –                  | –                | 60.0%        | 100.0%            | 100.0%            | 100.0%          | 100.0%              |
| solve_linear_elastic_1D_self_contained                      | 100.0%   | 100.0%  | –                  | 100.0%           | 100.0%       | 100.0%            | –                 | –               | –                   |
| solve_linear_elastic_frame_3D                               | 0.0%     | 100.0%  | –                  | 50.0%            | 0.0%         | 50.0%             | 100.0%            | 50.0%           | 50.0%               |
| solve_linear_elastic_frame_3D_self_contained                | 66.7%    | 100.0%  | –                  | 66.7%            | 33.3%        | 66.7%             | 66.7%             | 100.0%          | 66.7%               |
| tri6_shape_functions_and_derivatives                        | 66.7%    | 100.0%  | –                  | –                | –            | 100.0%            | 100.0%            | 50.0%           | 50.0%               |
| triangle_quadrature_2D                                      | 60.0%    | 100.0%  | –                  | –                | 40.0%        | 100.0%            | 100.0%            | 40.0%           | 80.0%               |
| Avg Joint Success %                                         | 42.5%    | 70.3%   | 0.0%               | 42.3%            | 28.4%        | 59.7%             | 52.0%             | 40.6%           | 42.2%               |


## Summary of Results <a name="summary"></a>

### Combined Aggregate (Function Corectness % / Joint Success %)

The Combined Aggregate (Function Corectness % / Joint Success %) table shows, for each model, how often its outputs exactly matched the reference functions (Function Correctness) and how often its generated code and tests passed jointly (Joint Success). Bold entries highlight the best performance per model across the three conditions (no system prompt, System Prompt 1, and System Prompt 2), making it easy to see whether adding a system prompt improved or reduced reliability for that model.


| Model             | No System Prompt | System Prompt 1 | System Prompt 2 |
|:------------------|:----------------:|:---------------:|:---------------:|
| gpt-4o            | **64 / 44.0**    | 64 / 43.9       | 52 / 42.5       |
| gpt-5             | 64 / 65.3        | **72 / 70.7**   | 72 / 70.3       |
| gemini-1.5-flash  | **36 / 31.7**    | 32 / 28.5       | 0 / 0.0         |
| gemini-2.5-pro    | **76** / 41.3    | 68 / **53.3**   | 68 / 42.3       |
| claude-3-5        | 64 / **46.8**    | 64 / 42.8       | **76** / 28.4   |
| claude-sonnet-4   | 56 / 56.3        | 60 / 53.1       | **64 / 59.7**   |
| claude-opus-4.1   | 64 / 51.6        | 64 / **58.3**   | **72** / 52.0   |
| deepseek-chat     | 52 / 39.0        | **68 / 43.9**   | 56 / 40.6       |
| deepseek-reasoner | 64 / **51.0**    | 60 / 44.5       | **68** / 42.2   |


### Task-by-task Accuracy @ 3 (best of all options)

This table shows, for each FEM task, whether each model produced a correct implementation in at least one of k independent runs, with the final row reporting overall Accuracy@k across all 25 tasks for temperature = 0.25.

### Task-by-task Accuracy@3 (best of 3 seeds)

| Task                                                        | gpt-4o   | gpt-5   | gemini-1.5-flash   | gemini-2.5-pro   | claude-3-5   | claude-sonnet-4   | claude-opus-4.1   | deepseek-chat   | deepseek-reasoner   |
|:------------------------------------------------------------|:---------|:--------|:-------------------|:-----------------|:-------------|:------------------|:------------------|:----------------|:--------------------|
| assemble_global_geometric_stiffness_3D_beam                 | 3/3 ✓    | 3/3 ✓   | 2/3 ✓              | 3/3 ✓            | 3/3 ✓        | 3/3 ✓             | 0/3 ×             | 2/3 ✓           | 3/3 ✓               |
| assemble_global_stiffness_matrix_linear_elastic_3D          | 3/3 ✓    | 3/3 ✓   | 2/3 ✓              | 3/3 ✓            | 3/3 ✓        | 3/3 ✓             | 3/3 ✓             | 3/3 ✓           | 3/3 ✓               |
| beam_transformation_matrix_3D                               | 3/3 ✓    | 2/3 ✓   | 0/3 ×              | 2/3 ✓            | 2/3 ✓        | 2/3 ✓             | 3/3 ✓             | 0/3 ×           | 0/3 ×               |
| compute_integral_of_derivative_quad8                        | 3/3 ✓    | 3/3 ✓   | 0/3 ×              | 3/3 ✓            | 3/3 ✓        | 3/3 ✓             | 3/3 ✓             | 3/3 ✓           | 2/3 ✓               |
| compute_local_element_loads_beam_3D                         | 3/3 ✓    | 1/3 ✓   | 2/3 ✓              | 3/3 ✓            | 3/3 ✓        | 3/3 ✓             | 3/3 ✓             | 3/3 ✓           | 3/3 ✓               |
| compute_physical_gradient_quad8                             | 0/3 ×    | 3/3 ✓   | 0/3 ×              | 2/3 ✓            | 1/3 ✓        | 0/3 ×             | 1/3 ✓             | 0/3 ×           | 2/3 ✓               |
| eigenvalue_analysis_msa_3D                                  | 0/3 ×    | 0/3 ×   | 0/3 ×              | 0/3 ×            | 0/3 ×        | 0/3 ×             | 0/3 ×             | 0/3 ×           | 0/3 ×               |
| elastic_critical_load_analysis_frame_3D                     | 3/3 ✓    | 3/3 ✓   | 2/3 ✓              | 3/3 ✓            | 3/3 ✓        | 3/3 ✓             | 3/3 ✓             | 3/3 ✓           | 3/3 ✓               |
| elastic_critical_load_analysis_frame_3D_part_self_contained | 0/3 ×    | 0/3 ×   | 0/3 ×              | 0/3 ×            | 0/3 ×        | 0/3 ×             | 0/3 ×             | 0/3 ×           | 0/3 ×               |
| elastic_critical_load_analysis_frame_3D_self_contained      | 0/3 ×    | 0/3 ×   | 0/3 ×              | 0/3 ×            | 0/3 ×        | 0/3 ×             | 0/3 ×             | 0/3 ×           | 0/3 ×               |
| element_distributed_load_quad8                              | 2/3 ✓    | 3/3 ✓   | 0/3 ×              | 3/3 ✓            | 2/3 ✓        | 3/3 ✓             | 3/3 ✓             | 1/3 ✓           | 3/3 ✓               |
| element_stiffness_linear_elastic_1D                         | 3/3 ✓    | 0/3 ×   | 2/3 ✓              | 1/3 ✓            | 3/3 ✓        | 3/3 ✓             | 3/3 ✓             | 3/3 ✓           | 3/3 ✓               |
| generate_quad8_rectangular_mesh                             | 3/3 ✓    | 3/3 ✓   | 0/3 ×              | 3/3 ✓            | 3/3 ✓        | 3/3 ✓             | 3/3 ✓             | 3/3 ✓           | 3/3 ✓               |
| generate_tri6_rectangular_mesh                              | 0/3 ×    | 3/3 ✓   | 0/3 ×              | 3/3 ✓            | 1/3 ✓        | 3/3 ✓             | 3/3 ✓             | 3/3 ✓           | 3/3 ✓               |
| linear_solve                                                | 3/3 ✓    | 3/3 ✓   | 0/3 ×              | 2/3 ✓            | 2/3 ✓        | 0/3 ×             | 1/3 ✓             | 0/3 ×           | 1/3 ✓               |
| linear_uniform_mesh_1D                                      | 3/3 ✓    | 2/3 ✓   | 2/3 ✓              | 3/3 ✓            | 3/3 ✓        | 3/3 ✓             | 3/3 ✓             | 3/3 ✓           | 3/3 ✓               |
| local_elastic_stiffness_matrix_3D_beam                      | 3/3 ✓    | 3/3 ✓   | 0/3 ×              | 3/3 ✓            | 3/3 ✓        | 3/3 ✓             | 3/3 ✓             | 1/3 ✓           | 2/3 ✓               |
| local_geometric_stiffness_matrix_3D_beam                    | 0/3 ×    | 0/3 ×   | 0/3 ×              | 0/3 ×            | 0/3 ×        | 0/3 ×             | 0/3 ×             | 0/3 ×           | 0/3 ×               |
| quad8_shape_functions_and_derivatives                       | 0/3 ×    | 3/3 ✓   | 1/3 ✓              | 3/3 ✓            | 0/3 ×        | 2/3 ✓             | 0/3 ×             | 2/3 ✓           | 3/3 ✓               |
| quad_quadrature_2D                                          | 3/3 ✓    | 2/3 ✓   | 0/3 ×              | 1/3 ✓            | 3/3 ✓        | 1/3 ✓             | 2/3 ✓             | 2/3 ✓           | 0/3 ×               |
| solve_linear_elastic_1D_self_contained                      | 2/3 ✓    | 1/3 ✓   | 0/3 ×              | 1/3 ✓            | 3/3 ✓        | 0/3 ×             | 3/3 ✓             | 3/3 ✓           | 2/3 ✓               |
| solve_linear_elastic_frame_3D                               | 3/3 ✓    | 3/3 ✓   | 2/3 ✓              | 3/3 ✓            | 3/3 ✓        | 3/3 ✓             | 3/3 ✓             | 3/3 ✓           | 3/3 ✓               |
| solve_linear_elastic_frame_3D_self_contained                | 0/3 ×    | 2/3 ✓   | 0/3 ×              | 2/3 ✓            | 2/3 ✓        | 1/3 ✓             | 1/3 ✓             | 1/3 ✓           | 1/3 ✓               |
| tri6_shape_functions_and_derivatives                        | 0/3 ×    | 3/3 ✓   | 2/3 ✓              | 3/3 ✓            | 2/3 ✓        | 0/3 ×             | 3/3 ✓             | 3/3 ✓           | 3/3 ✓               |
| triangle_quadrature_2D                                      | 2/3 ✓    | 3/3 ✓   | 0/3 ×              | 3/3 ✓            | 3/3 ✓        | 3/3 ✓             | 3/3 ✓             | 2/3 ✓           | 2/3 ✓               |
| **Overall Accuracy@3**                                      | 16/25    | 20/25   | 9/25               | 21/25            | 20/25        | 17/25             | 19/25             | 18/25           | 19/25               |


### Joint Test Accuracy @ 3 per test (grouped by task) (best of all options)

This table breaks down individual test functions, showing for each task::test combination whether a model simultaneously passed the reference test and failed its paired expected-failure test in at least one of k runs, with a final summary row giving overall Joint@k coverage per model.


| Task::Test                                                                                                                | gpt-4o        | gpt-5         | gemini-1.5-flash   | gemini-2.5-pro   | claude-3-5    | claude-sonnet-4   | claude-opus-4.1   | deepseek-chat   | deepseek-reasoner   |
|:--------------------------------------------------------------------------------------------------------------------------|:--------------|:--------------|:-------------------|:-----------------|:--------------|:------------------|:------------------|:----------------|:--------------------|
| assemble_global_geometric_stiffness_3D_beam::test_frame_objectivity_under_global_rotation                                 | 0/3 ×         | 0/3 ×         | 0/3 ×              | 0/3 ×            | 0/3 ×         | 0/3 ×             | 0/3 ×             | 0/3 ×           | 0/3 ×               |
| assemble_global_geometric_stiffness_3D_beam::test_multi_element_core_correctness_assembly                                 | 0/3 ×         | 1/3 ✓         | 1/3 ✓              | 0/3 ×            | 1/3 ✓         | 0/3 ×             | 1/3 ✓             | 1/3 ✓           | 1/3 ✓               |
| assemble_global_stiffness_matrix_linear_elastic_3D::test_assemble_global_stiffness_matrix_shape_and_symmetry              | 0/3 ×         | 2/3 ✓         | 1/3 ✓              | 3/3 ✓            | 2/3 ✓         | 3/3 ✓             | 3/3 ✓             | 0/3 ×           | 0/3 ×               |
| beam_transformation_matrix_3D::test_beam_transformation_matrix_error_messages                                             | 3/3 ✓         | 3/3 ✓         | 2/3 ✓              | 0/3 ×            | 2/3 ✓         | 3/3 ✓             | 1/3 ✓             | 3/3 ✓           | 3/3 ✓               |
| beam_transformation_matrix_3D::test_cardinal_axis_alignment                                                               | 0/3 ×         | 2/3 ✓         | 0/3 ×              | 0/3 ×            | 0/3 ×         | 0/3 ×             | 0/3 ×             | 0/3 ×           | 0/3 ×               |
| beam_transformation_matrix_3D::test_transformation_matrix_properties                                                      | 0/3 ×         | 2/3 ✓         | 0/3 ×              | 0/3 ×            | 0/3 ×         | 0/3 ×             | 0/3 ×             | 1/3 ✓           | 1/3 ✓               |
| compute_integral_of_derivative_quad8::test_integral_of_derivative_quad8_affine_linear_field                               | 2/3 ✓         | 3/3 ✓         | 1/3 ✓              | 3/3 ✓            | 3/3 ✓         | 3/3 ✓             | 3/3 ✓             | 3/3 ✓           | 3/3 ✓               |
| compute_integral_of_derivative_quad8::test_integral_of_derivative_quad8_identity_cubic                                    | 0/3 ×         | 3/3 ✓         | 1/3 ✓              | 3/3 ✓            | 3/3 ✓         | 3/3 ✓             | 3/3 ✓             | 3/3 ✓           | 3/3 ✓               |
| compute_integral_of_derivative_quad8::test_integral_of_derivative_quad8_order_check_asymmetric_curved                     | 0/3 ×         | 0/3 ×         | 0/3 ×              | 0/3 ×            | 0/3 ×         | 0/3 ×             | 0/3 ×             | 0/3 ×           | 0/3 ×               |
| compute_local_element_loads_beam_3D::test_coordinate_invariance_global_rotation                                           | 2/3 ✓         | 2/3 ✓         | 0/3 ×              | 2/3 ✓            | 0/3 ×         | 2/3 ✓             | 2/3 ✓             | 2/3 ✓           | 3/3 ✓               |
| compute_local_element_loads_beam_3D::test_rigid_body_motion_zero_loads                                                    | 3/3 ✓         | 1/3 ✓         | 0/3 ×              | 3/3 ✓            | 0/3 ×         | 0/3 ×             | 0/3 ×             | 1/3 ✓           | 2/3 ✓               |
| compute_local_element_loads_beam_3D::test_superposition_linearity                                                         | 3/3 ✓         | 3/3 ✓         | 0/3 ×              | 3/3 ✓            | 0/3 ×         | 3/3 ✓             | 2/3 ✓             | 3/3 ✓           | 3/3 ✓               |
| compute_local_element_loads_beam_3D::test_unit_responses_axial_shear_torsion                                              | 0/3 ×         | 3/3 ✓         | 0/3 ×              | 1/3 ✓            | 0/3 ×         | 3/3 ✓             | 1/3 ✓             | 1/3 ✓           | 2/3 ✓               |
| compute_physical_gradient_quad8::test_q8_gradient_identity_mapping_matches_quadratic_analytic                             | 3/3 ✓         | 3/3 ✓         | 1/3 ✓              | 3/3 ✓            | 2/3 ✓         | 3/3 ✓             | 3/3 ✓             | 3/3 ✓           | 3/3 ✓               |
| compute_physical_gradient_quad8::test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant                  | 3/3 ✓         | 3/3 ✓         | 2/3 ✓              | 3/3 ✓            | 2/3 ✓         | 3/3 ✓             | 3/3 ✓             | 3/3 ✓           | 3/3 ✓               |
| eigenvalue_analysis_msa_3D::test_eigen_complex_eigenpairs_detected                                                        | 0/3 ×         | 0/3 ×         | 0/3 ×              | 0/3 ×            | 1/3 ✓         | 0/3 ×             | 0/3 ×             | 0/3 ×           | 0/3 ×               |
| eigenvalue_analysis_msa_3D::test_eigen_invariance_to_reference_load_scaling                                               | 0/3 ×         | 0/3 ×         | 0/3 ×              | 0/3 ×            | 3/3 ✓         | 0/3 ×             | 0/3 ×             | 0/3 ×           | 0/3 ×               |
| eigenvalue_analysis_msa_3D::test_eigen_known_answer                                                                       | 0/3 ×         | 0/3 ×         | 0/3 ×              | 0/3 ×            | 0/3 ×         | 0/3 ×             | 0/3 ×             | 0/3 ×           | 0/3 ×               |
| eigenvalue_analysis_msa_3D::test_eigen_no_positive_eigenvalues_detected                                                   | 0/3 ×         | 0/3 ×         | 0/3 ×              | 0/3 ×            | 0/3 ×         | 0/3 ×             | 0/3 ×             | 0/3 ×           | 0/3 ×               |
| eigenvalue_analysis_msa_3D::test_eigen_singluar_detected                                                                  | 0/3 ×         | 0/3 ×         | 0/3 ×              | 0/3 ×            | 1/3 ✓         | 0/3 ×             | 0/3 ×             | 0/3 ×           | 0/3 ×               |
| elastic_critical_load_analysis_frame_3D::test_cantilever_euler_buckling_mesh_convergence                                  | 0/3 ×         | 0/3 ×         | 0/3 ×              | 0/3 ×            | 0/3 ×         | 0/3 ×             | 0/3 ×             | 0/3 ×           | 0/3 ×               |
| elastic_critical_load_analysis_frame_3D::test_euler_buckling_cantilever_circular_param_sweep                              | 0/3 ×         | 0/3 ×         | 0/3 ×              | 0/3 ×            | 0/3 ×         | 0/3 ×             | 0/3 ×             | 0/3 ×           | 0/3 ×               |
| elastic_critical_load_analysis_frame_3D::test_orientation_invariance_cantilever_buckling_rect_section                     | 0/3 ×         | 0/3 ×         | 0/3 ×              | 0/3 ×            | 0/3 ×         | 0/3 ×             | 0/3 ×             | 0/3 ×           | 0/3 ×               |
| elastic_critical_load_analysis_frame_3D_part_self_contained::test_cantilever_euler_buckling_mesh_convergence              | 0/3 ×         | 0/3 ×         | 0/3 ×              | 0/3 ×            | 0/3 ×         | 0/3 ×             | 0/3 ×             | 0/3 ×           | 0/3 ×               |
| elastic_critical_load_analysis_frame_3D_part_self_contained::test_euler_buckling_cantilever_circular_param_sweep          | 0/3 ×         | 0/3 ×         | 0/3 ×              | 0/3 ×            | 0/3 ×         | 0/3 ×             | 0/3 ×             | 0/3 ×           | 0/3 ×               |
| elastic_critical_load_analysis_frame_3D_part_self_contained::test_orientation_invariance_cantilever_buckling_rect_section | 0/3 ×         | 0/3 ×         | 0/3 ×              | 0/3 ×            | 0/3 ×         | 0/3 ×             | 0/3 ×             | 0/3 ×           | 0/3 ×               |
| elastic_critical_load_analysis_frame_3D_self_contained::test_cantilever_euler_buckling_mesh_convergence                   | 0/3 ×         | 0/3 ×         | 0/3 ×              | 0/3 ×            | 0/3 ×         | 0/3 ×             | 0/3 ×             | 0/3 ×           | 0/3 ×               |
| elastic_critical_load_analysis_frame_3D_self_contained::test_euler_buckling_cantilever_circular_param_sweep               | 0/3 ×         | 0/3 ×         | 0/3 ×              | 0/3 ×            | 0/3 ×         | 0/3 ×             | 0/3 ×             | 0/3 ×           | 0/3 ×               |
| elastic_critical_load_analysis_frame_3D_self_contained::test_orientation_invariance_cantilever_buckling_rect_section      | 0/3 ×         | 0/3 ×         | 0/3 ×              | 0/3 ×            | 0/3 ×         | 0/3 ×             | 0/3 ×             | 0/3 ×           | 0/3 ×               |
| element_distributed_load_quad8::test_edl_q8_analytic_straight_edges_total_force_scaled_all_faces                          | 3/3 ✓         | 3/3 ✓         | 2/3 ✓              | 3/3 ✓            | 1/3 ✓         | 3/3 ✓             | 3/3 ✓             | 3/3 ✓           | 2/3 ✓               |
| element_distributed_load_quad8::test_edl_q8_constant_traction_total_force_on_curved_parabolic_edge                        | 3/3 ✓         | 0/3 ×         | 0/3 ×              | 1/3 ✓            | 1/3 ✓         | 0/3 ×             | 3/3 ✓             | 0/3 ×           | 0/3 ×               |
| element_stiffness_linear_elastic_1D::test_element_stiffness_comprehensive                                                 | 3/3 ✓         | 3/3 ✓         | 2/3 ✓              | 3/3 ✓            | 1/3 ✓         | 3/3 ✓             | 3/3 ✓             | 3/3 ✓           | 3/3 ✓               |
| generate_quad8_rectangular_mesh::test_quad8_mesh_basic_structure_and_determinism                                          | 3/3 ✓         | 3/3 ✓         | 1/3 ✓              | 1/3 ✓            | 1/3 ✓         | 3/3 ✓             | 1/3 ✓             | 2/3 ✓           | 1/3 ✓               |
| generate_quad8_rectangular_mesh::test_quad8_mesh_geometry_and_conformity                                                  | 0/3 ×         | 3/3 ✓         | 0/3 ×              | 2/3 ✓            | 3/3 ✓         | 3/3 ✓             | 3/3 ✓             | 1/3 ✓           | 0/3 ×               |
| generate_quad8_rectangular_mesh::test_quad8_mesh_invalid_inputs                                                           | 3/3 ✓         | 3/3 ✓         | 1/3 ✓              | 2/3 ✓            | 3/3 ✓         | 3/3 ✓             | 3/3 ✓             | 3/3 ✓           | 2/3 ✓               |
| generate_tri6_rectangular_mesh::test_tri6_mesh_basic_structure_and_determinism                                            | 2/3 ✓         | 3/3 ✓         | 0/3 ×              | 3/3 ✓            | 2/3 ✓         | 2/3 ✓             | 1/3 ✓             | 1/3 ✓           | 2/3 ✓               |
| generate_tri6_rectangular_mesh::test_tri6_mesh_geometry_and_conformity                                                    | 1/3 ✓         | 3/3 ✓         | 0/3 ×              | 3/3 ✓            | 3/3 ✓         | 3/3 ✓             | 1/3 ✓             | 1/3 ✓           | 1/3 ✓               |
| generate_tri6_rectangular_mesh::test_tri6_mesh_invalid_inputs                                                             | 3/3 ✓         | 3/3 ✓         | 0/3 ×              | 3/3 ✓            | 3/3 ✓         | 3/3 ✓             | 1/3 ✓             | 2/3 ✓           | 2/3 ✓               |
| linear_solve::test_linear_solve_arbitrary_solvable_cases                                                                  | 0/3 ×         | 3/3 ✓         | 0/3 ×              | 2/3 ✓            | 0/3 ×         | 1/3 ✓             | 1/3 ✓             | 0/3 ×           | 1/3 ✓               |
| linear_solve::test_linear_solve_raises_on_ill_conditioned_matrix                                                          | 0/3 ×         | 0/3 ×         | 0/3 ×              | 0/3 ×            | 0/3 ×         | 1/3 ✓             | 1/3 ✓             | 0/3 ×           | 0/3 ×               |
| linear_uniform_mesh_1D::test_basic_mesh_creation                                                                          | 3/3 ✓         | 3/3 ✓         | 2/3 ✓              | 2/3 ✓            | 3/3 ✓         | 3/3 ✓             | 3/3 ✓             | 3/3 ✓           | 3/3 ✓               |
| linear_uniform_mesh_1D::test_single_element_mesh                                                                          | 3/3 ✓         | 3/3 ✓         | 2/3 ✓              | 2/3 ✓            | 3/3 ✓         | 3/3 ✓             | 3/3 ✓             | 3/3 ✓           | 3/3 ✓               |
| local_elastic_stiffness_matrix_3D_beam::test_cantilever_deflection_matches_euler_bernoulli                                | 0/3 ×         | 3/3 ✓         | 0/3 ×              | 0/3 ×            | 0/3 ×         | 0/3 ×             | 0/3 ×             | 0/3 ×           | 1/3 ✓               |
| local_elastic_stiffness_matrix_3D_beam::test_local_stiffness_3D_beam                                                      | 1/3 ✓         | 0/3 ×         | 0/3 ×              | 0/3 ×            | 0/3 ×         | 0/3 ×             | 0/3 ×             | 0/3 ×           | 0/3 ×               |
| local_geometric_stiffness_matrix_3D_beam::test_euler_buckling_cantilever_column                                           | 2/3 ✓         | 2/3 ✓         | 0/3 ×              | 0/3 ×            | 0/3 ×         | 0/3 ×             | 0/3 ×             | 0/3 ×           | 0/3 ×               |
| local_geometric_stiffness_matrix_3D_beam::test_local_geometric_stiffness_matrix_3D_beam_comprehensive                     | 0/3 ×         | 3/3 ✓         | 2/3 ✓              | 1/3 ✓            | 2/3 ✓         | 0/3 ×             | 0/3 ×             | 0/3 ×           | 0/3 ×               |
| quad8_shape_functions_and_derivatives::test_derivative_partition_of_unity_quad8                                           | 3/3 ✓         | 3/3 ✓         | 1/3 ✓              | 1/3 ✓            | 2/3 ✓         | 3/3 ✓             | 3/3 ✓             | 3/3 ✓           | 3/3 ✓               |
| quad8_shape_functions_and_derivatives::test_gradient_completeness_quad8                                                   | 0/3 ×         | 2/3 ✓         | 0/3 ×              | 1/3 ✓            | 0/3 ×         | 3/3 ✓             | 3/3 ✓             | 2/3 ✓           | 2/3 ✓               |
| quad8_shape_functions_and_derivatives::test_kronecker_delta_at_nodes_quad8                                                | 3/3 ✓         | 3/3 ✓         | 0/3 ×              | 1/3 ✓            | 2/3 ✓         | 3/3 ✓             | 3/3 ✓             | 3/3 ✓           | 3/3 ✓               |
| quad8_shape_functions_and_derivatives::test_partition_of_unity_quad8                                                      | 3/3 ✓         | 3/3 ✓         | 1/3 ✓              | 1/3 ✓            | 2/3 ✓         | 3/3 ✓             | 3/3 ✓             | 3/3 ✓           | 3/3 ✓               |
| quad8_shape_functions_and_derivatives::test_quad8_shape_functions_and_derivatives_input_errors                            | 1/3 ✓         | 3/3 ✓         | 0/3 ×              | 1/3 ✓            | 2/3 ✓         | 3/3 ✓             | 3/3 ✓             | 3/3 ✓           | 3/3 ✓               |
| quad8_shape_functions_and_derivatives::test_value_completeness_quad8                                                      | 3/3 ✓         | 2/3 ✓         | 0/3 ×              | 1/3 ✓            | 1/3 ✓         | 3/3 ✓             | 3/3 ✓             | 1/3 ✓           | 1/3 ✓               |
| quad_quadrature_2D::test_quad_quadrature_2D_basics                                                                        | 3/3 ✓         | 3/3 ✓         | 2/3 ✓              | 1/3 ✓            | 3/3 ✓         | 3/3 ✓             | 3/3 ✓             | 2/3 ✓           | 3/3 ✓               |
| quad_quadrature_2D::test_quad_quadrature_2D_degree_exactness_1pt                                                          | 0/3 ×         | 3/3 ✓         | 1/3 ✓              | 1/3 ✓            | 3/3 ✓         | 3/3 ✓             | 3/3 ✓             | 2/3 ✓           | 3/3 ✓               |
| quad_quadrature_2D::test_quad_quadrature_2D_degree_exactness_2x2                                                          | 1/3 ✓         | 3/3 ✓         | 0/3 ×              | 1/3 ✓            | 0/3 ×         | 3/3 ✓             | 3/3 ✓             | 2/3 ✓           | 3/3 ✓               |
| quad_quadrature_2D::test_quad_quadrature_2D_degree_exactness_3x3                                                          | 0/3 ×         | 3/3 ✓         | 0/3 ×              | 1/3 ✓            | 0/3 ×         | 2/3 ✓             | 3/3 ✓             | 2/3 ✓           | 3/3 ✓               |
| quad_quadrature_2D::test_quad_quadrature_2D_invalid_inputs                                                                | 3/3 ✓         | 3/3 ✓         | 2/3 ✓              | 1/3 ✓            | 3/3 ✓         | 3/3 ✓             | 3/3 ✓             | 2/3 ✓           | 3/3 ✓               |
| solve_linear_elastic_1D_self_contained::test_no_load_self_contained                                                       | 3/3 ✓         | 3/3 ✓         | 2/3 ✓              | 2/3 ✓            | 2/3 ✓         | 3/3 ✓             | 0/3 ×             | 1/3 ✓           | 2/3 ✓               |
| solve_linear_elastic_1D_self_contained::test_uniform_extension_analytical_self_contained                                  | 1/3 ✓         | 3/3 ✓         | 1/3 ✓              | 2/3 ✓            | 2/3 ✓         | 3/3 ✓             | 0/3 ×             | 1/3 ✓           | 2/3 ✓               |
| solve_linear_elastic_frame_3D::test_complex_geometry_and_basic_loading                                                    | 0/3 ×         | 3/3 ✓         | 0/3 ×              | 2/3 ✓            | 2/3 ✓         | 3/3 ✓             | 3/3 ✓             | 1/3 ✓           | 0/3 ×               |
| solve_linear_elastic_frame_3D::test_simple_beam_discretized_axis_111                                                      | 0/3 ×         | 3/3 ✓         | 0/3 ×              | 2/3 ✓            | 2/3 ✓         | 0/3 ×             | 3/3 ✓             | 3/3 ✓           | 3/3 ✓               |
| solve_linear_elastic_frame_3D_self_contained::test_complex_geometry_and_basic_loading                                     | 1/3 ✓         | 3/3 ✓         | 0/3 ×              | 3/3 ✓            | 0/3 ×         | 3/3 ✓             | 3/3 ✓             | 1/3 ✓           | 0/3 ×               |
| solve_linear_elastic_frame_3D_self_contained::test_ill_conditioned_due_to_under_constrained_structure                     | 3/3 ✓         | 3/3 ✓         | 0/3 ×              | 3/3 ✓            | 3/3 ✓         | 3/3 ✓             | 3/3 ✓             | 2/3 ✓           | 3/3 ✓               |
| solve_linear_elastic_frame_3D_self_contained::test_simple_beam_discretized_axis_111                                       | 0/3 ×         | 3/3 ✓         | 0/3 ×              | 2/3 ✓            | 1/3 ✓         | 0/3 ×             | 2/3 ✓             | 1/3 ✓           | 1/3 ✓               |
| tri6_shape_functions_and_derivatives::test_derivative_partition_of_unity_tri6                                             | 3/3 ✓         | 3/3 ✓         | 2/3 ✓              | 1/3 ✓            | 1/3 ✓         | 3/3 ✓             | 3/3 ✓             | 3/3 ✓           | 3/3 ✓               |
| tri6_shape_functions_and_derivatives::test_gradient_completeness_tri6                                                     | 0/3 ×         | 2/3 ✓         | 0/3 ×              | 1/3 ✓            | 0/3 ×         | 3/3 ✓             | 3/3 ✓             | 0/3 ×           | 0/3 ×               |
| tri6_shape_functions_and_derivatives::test_kronecker_delta_at_nodes_tri6                                                  | 3/3 ✓         | 3/3 ✓         | 0/3 ×              | 1/3 ✓            | 0/3 ×         | 3/3 ✓             | 3/3 ✓             | 2/3 ✓           | 0/3 ×               |
| tri6_shape_functions_and_derivatives::test_partition_of_unity_tri6                                                        | 3/3 ✓         | 3/3 ✓         | 2/3 ✓              | 1/3 ✓            | 1/3 ✓         | 3/3 ✓             | 3/3 ✓             | 3/3 ✓           | 3/3 ✓               |
| tri6_shape_functions_and_derivatives::test_tri6_shape_functions_and_derivatives_input_errors                              | 3/3 ✓         | 3/3 ✓         | 1/3 ✓              | 2/3 ✓            | 1/3 ✓         | 3/3 ✓             | 3/3 ✓             | 3/3 ✓           | 3/3 ✓               |
| tri6_shape_functions_and_derivatives::test_value_completeness_tri6                                                        | 2/3 ✓         | 3/3 ✓         | 0/3 ×              | 1/3 ✓            | 0/3 ×         | 3/3 ✓             | 3/3 ✓             | 0/3 ×           | 0/3 ×               |
| triangle_quadrature_2D::test_triangle_quadrature_2D_basics                                                                | 3/3 ✓         | 3/3 ✓         | 2/3 ✓              | 1/3 ✓            | 3/3 ✓         | 3/3 ✓             | 3/3 ✓             | 2/3 ✓           | 3/3 ✓               |
| triangle_quadrature_2D::test_triangle_quadrature_2D_degree_exactness_1pt                                                  | 3/3 ✓         | 3/3 ✓         | 0/3 ×              | 1/3 ✓            | 0/3 ×         | 1/3 ✓             | 2/3 ✓             | 0/3 ×           | 3/3 ✓               |
| triangle_quadrature_2D::test_triangle_quadrature_2D_degree_exactness_3pt                                                  | 1/3 ✓         | 3/3 ✓         | 0/3 ×              | 1/3 ✓            | 0/3 ×         | 1/3 ✓             | 2/3 ✓             | 0/3 ×           | 1/3 ✓               |
| triangle_quadrature_2D::test_triangle_quadrature_2D_degree_exactness_4pt                                                  | 1/3 ✓         | 3/3 ✓         | 0/3 ×              | 1/3 ✓            | 0/3 ×         | 1/3 ✓             | 2/3 ✓             | 0/3 ×           | 3/3 ✓               |
| triangle_quadrature_2D::test_triangle_quadrature_2D_invalid_inputs                                                        | 3/3 ✓         | 3/3 ✓         | 0/3 ×              | 1/3 ✓            | 3/3 ✓         | 3/3 ✓             | 3/3 ✓             | 2/3 ✓           | 3/3 ✓               |
| **Overall Joint@3**                                                                                                       | 41/75 (54.7%) | 56/75 (74.7%) | 25/75 (33.3%)      | 51/75 (68.0%)    | 41/75 (54.7%) | 48/75 (64.0%)     | 50/75 (66.7%)     | 45/75 (60.0%)   | 46/75 (61.3%)       |
