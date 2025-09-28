from pathlib import Path
from tools.report_task_seed_summary import build_task_by_task_accuracy_at_k_markdown
from tools.report_joint_per_test import build_joint_per_test_markdown


# Results from three independent seeds (each dir is a separate pipeline run):
# dirs = [
#     "results_temperature1",
#     "results_temperature2",
#     "results_temperature3",
# ]

dirs = [
    "results_no_system_prompt",
    "results_system_prompt",
    "results_system_prompt_v2",
]

# Optional fixed model order to match your paper/figure:
models = ["gpt-4o", "gpt-5", "gemini-1.5-flash", "gemini-2.5-pro",
          "claude-3-5", "claude-sonnet-4", "claude-opus-4.1",
          "deepseek-chat", "deepseek-reasoner"]

md = build_task_by_task_accuracy_at_k_markdown(
    dirs,
    models=models,
    table_title="Task-by-task Accuracy@3 (best of 3 seeds)"
)

# Save next to other artifacts:
# Path("accuracy_at_3_task_table_temp025.md").write_text(md, encoding="utf-8")
Path("accuracy_at_3_task_table_system_prompt.md").write_text(md, encoding="utf-8")


md = build_joint_per_test_markdown(
    dirs,
    models=models,
    table_title="Joint@3 per test (grouped by task)"
)

# Path("accuracy_at_3_joint_per_test_temp025.md").write_text(md, encoding="utf-8")
Path("accuracy_at_3_joint_per_test_system_prompt.md").write_text(md, encoding="utf-8")



aa = 44