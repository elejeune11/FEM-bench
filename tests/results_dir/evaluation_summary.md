### Function Correctness (✓ = Match)

| Task                                   | claude4   | gemini   | gpt41   | maiu   |
|:---------------------------------------|:----------|:---------|:--------|:-------|
| element_stiffness_linear_elastic_1D    | ✓         | ✓        | ✓       | ×      |
| linear_uniform_mesh_1D                 | ✓         | ✓        | ✓       | ✓      |
| solve_linear_elastic_1D_self_contained | ×         | ×        | ✓       | ✓      |
| Total                                  | 2/3       | 2/3      | 3/3     | 2/3    |

### Joint Test Success Rate (%)

| Task                                   | claude4   | gemini   | gpt41   | maiu   |
|:---------------------------------------|:----------|:---------|:--------|:-------|
| element_stiffness_linear_elastic_1D    | 0.0%      | 100.0%   | 100.0%  | 100.0% |
| linear_uniform_mesh_1D                 | 100.0%    | 100.0%   | 100.0%  | 100.0% |
| solve_linear_elastic_1D_self_contained | 50.0%     | 50.0%    | 100.0%  | 50.0%  |
| Avg Joint Success %                    | 50.0%     | 83.3%    | 100.0%  | 83.3%  |

