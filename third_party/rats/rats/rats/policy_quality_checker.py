from __future__ import annotations

from rats.rats.schemas import PolicyDraft, QualityCheckResult


class PolicyQualityChecker:
    FORBIDDEN_PATTERNS = {
        "while True": "unbounded retry loop",
        "find_object_base_rotate": "search helper is forbidden in deterministic RATS path",
        "find_object_torso_rotate": "search helper is forbidden in deterministic RATS path",
        "exec(": "dynamic execution is forbidden",
        "eval(": "dynamic evaluation is forbidden",
        "env.": "use imported primitive functions instead of env.<method>",
        "low_level_env.": "use imported primitive functions instead of low_level_env.<method>",
    }

    def check(self, draft: PolicyDraft) -> QualityCheckResult:
        code = draft.code.strip()
        if not code:
            return QualityCheckResult(approved=False, feedback="policy draft is empty")

        for pattern, reason in self.FORBIDDEN_PATTERNS.items():
            if pattern in code:
                return QualityCheckResult(approved=False, feedback=f"rejected: {reason}")

        if "RESULT" not in code and "return" not in code:
            return QualityCheckResult(
                approved=False,
                feedback="rejected: policy draft must communicate an outcome via RESULT or return",
            )

        return QualityCheckResult(approved=True, feedback="approved")
