"""Multi-turn decider: CaP-X-style "FINISH vs REGENERATE+code" gate.

Runs inside an attempt's turn loop (env not reset between turns). After a
turn's code executes, the decider sees the executed code, console output,
and the resulting frame, then returns either:

  - FINISH      -> stop iterating and hand off to the verifier
  - REGENERATE  -> a new code block to execute on the next turn

This is a single LLM call per turn, much lighter than the full
verifier -> diagnoser -> feedback chain. It only gates within-attempt
turn transitions; attempt boundaries still use the full RATS pipeline so
skill extraction and failure-memory accounting are unaffected.

Mirrors rats/envs/trial.py:_handle_multi_turn_step (the CaP-X reference
implementation), simplified for RATS:
  - Always uses the after-execution frame; no visual differencing or
    video VDM (config-selectable later if needed).
  - Returns a plain dict instead of CaP-X's tuple of five values.
"""

from __future__ import annotations

import logging
from typing import Any

import re

from rats.agents.base_agent import (
    DEFAULT_MAX_TOKENS,
    image_to_data_url,
    query_llm_text,
)


_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE
)

logger = logging.getLogger("rats.multi_turn_decider")


def resolve_public_task_goal(
    task_proposal: dict[str, Any] | None,
    scene_context: dict[str, Any] | None,
) -> str:
    """Resolve the public natural-language goal used for runtime decisions.

    Catalog proposals are not required to duplicate the environment language
    under ``language``.  Falling straight back to an opaque identifier such as
    ``libero_90_task9`` leaves the visual decider unable to distinguish a
    displaced-object recovery from an unrelated task suggested by whatever
    fixture remains visible.  Prefer public instruction strings exposed by the
    proposal or scene context; use the activity id only as a last resort.

    Non-string values are deliberately ignored so symbolic/native predicate
    structures can never enter the controller prompt through this resolver.
    """

    proposal = task_proposal or {}
    scene = scene_context or {}
    candidates = (
        proposal.get("language"),
        proposal.get("goal_description"),
        scene.get("goal_conditions_nl"),
        proposal.get("goal_conditions"),
        scene.get("task_language"),
        proposal.get("activity_name"),
    )
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


_SYSTEM_PROMPT = (
    "You are a helpful assistant that decides whether a multi-step robot "
    "policy execution should stop or continue. You will be shown the code "
    "that was just executed, the console output, and an image of the "
    "current environment state. Respond with EXACTLY ONE of:\n"
    "  - The word 'FINISH' if the task appears complete.\n"
    "  - The word 'REGENERATE' followed immediately by new Python code in "
    "a fenced ```python ... ``` block if you want to keep going from the "
    "current state. The new code will be executed without an environment "
    "reset, so it must build on the state visible in the image."
)


_USER_TEMPLATE = """\
Task goal: {task_goal}

The following code block was just executed:
```python
{executed_code}
```

Console stdout:
```
{stdout}
```

Console stderr:
```
{stderr}
```

If there were no errors during compilation or execution the stderr block \
will be empty. The attached image shows the current state of the \
environment AFTER the code above was executed.

Decide based on the image (visual evidence) and the console output:
  - FINISH if the task goal already appears satisfied.
  - REGENERATE followed by new ```python``` code if more actions are \
needed. The environment WILL NOT be reset before your new code runs, so \
do not redo actions whose effects you can already see in the image \
(e.g. don't re-grasp an object that is already in the gripper).
"""


class MultiTurnDecider:
    """Single-call gate that decides FINISH vs REGENERATE+code."""

    def __init__(
        self,
        *,
        model: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens

    def decide(
        self,
        *,
        executed_code: str,
        stdout: str,
        stderr: str,
        after_frame: Any,
        task_goal: str = "",
    ) -> dict[str, Any]:
        """Return decision dict {action, new_code, raw}."""
        user_prompt = _USER_TEMPLATE.format(
            task_goal=task_goal or "(no explicit goal text available)",
            executed_code=(executed_code or "").strip() or "(empty)",
            stdout=(stdout or "").strip(),
            stderr=(stderr or "").strip(),
        )

        images: list[str] = []
        img_url = image_to_data_url(after_frame) if after_frame is not None else None
        if img_url:
            images.append(img_url)

        kwargs: dict[str, Any] = {
            "images": images or None,
            "max_tokens": self.max_tokens,
        }
        if self.model:
            kwargs["model"] = self.model

        try:
            raw = query_llm_text(_SYSTEM_PROMPT, user_prompt, **kwargs)
        except Exception as exc:
            logger.warning(
                "multi_turn_decider LLM call failed (%s): %s; falling back to FINISH",
                type(exc).__name__,
                str(exc)[:200],
            )
            return {
                "action": "finish",
                "new_code": None,
                "raw": "",
                "error": str(exc),
            }

        return self._parse(raw)

    @staticmethod
    def _parse(content: str) -> dict[str, Any]:
        """Parse the model response. Mirrors CaP-X's _parse_multi_turn_decision."""
        text = (content or "").strip()
        if not text:
            return {"action": "finish", "new_code": None, "raw": ""}

        upper = text.upper()
        if "REGENERATE" in upper:
            # Look for a fenced ```python``` block AFTER the REGENERATE
            # token. We require an actual fence so we never execute the
            # model's prose as code on the next turn.
            idx = upper.find("REGENERATE")
            tail = text[idx + len("REGENERATE"):]
            match = _FENCE_RE.search(tail)
            if match:
                new_code = match.group(1).strip()
                if new_code:
                    return {
                        "action": "regenerate",
                        "new_code": new_code,
                        "raw": text,
                    }
            logger.warning(
                "multi_turn_decider returned REGENERATE without a fenced "
                "code block; falling back to FINISH"
            )
            return {"action": "finish", "new_code": None, "raw": text}

        return {"action": "finish", "new_code": None, "raw": text}
