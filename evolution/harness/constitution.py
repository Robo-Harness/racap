"""Small, stable priors supplied to every coding iteration."""

from .prompting import read_prompt

CONSTITUTION = read_prompt("constitution.md")

BOUNDARY_RUBRIC = {
    "agent_control_fraction": (
        "Fraction of retries/routing/stopping choices made by runtime agent, target >= 0.8"
    ),
    "api_parameter_coverage": (
        "Fraction of materially distinct skill strategies exposed as arguments, target >= 0.7"
    ),
    "structured_failure_coverage": (
        "Fraction of failed calls returning phase + evidence rather than a bare boolean, target 1.0"
    ),
    "validator_veto_count": "Number of non-safety validators that silently stop an agent action, target 0",
    "task_id_branch_count": "Branches on benchmark task ID/seed/order, required 0",
}
