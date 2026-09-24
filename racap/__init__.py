"""RACaP: reasoning, acting, and coding as reusable robot policies.

Layering, from bottom to top:

``racap.backends``   bind a simulator or robot to the PrimitiveRuntime protocol
``racap.policy_api`` the evolved, stable-signature skills (pickplace, ...)
``racap.agent``      the long-horizon ReAct controllers that compose those skills
"""

from racap.contracts import GroundedTarget, PrimitiveRuntime, SkillResult

__all__ = ["GroundedTarget", "PrimitiveRuntime", "SkillResult"]
