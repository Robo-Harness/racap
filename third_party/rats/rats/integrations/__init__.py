from __future__ import annotations

import os

from .base_api import list_apis, register_api


_REQUESTED_STACKS = {
    item.strip().lower()
    for item in os.environ.get("CAPX_ENV_STACK", "all").split(",")
    if item.strip()
}


def _should_register(stack: str) -> bool:
    return "all" in _REQUESTED_STACKS or stack in _REQUESTED_STACKS


def _safe_registration(label: str, fn) -> None:
    try:
        fn()
    except Exception as exc:  # pragma: no cover - optional integrations
        print(f"{label} not installed, skipping APIs: {exc}")


def _register_franka_control() -> None:
    from .franka.control import FrankaControlApi
    from .franka.control_privileged import FrankaControlPrivilegedApi
    from .franka.control_reduced import FrankaControlApiReduced
    from .franka.control_reduced_skill_library import FrankaControlApiReducedSkillLibrary
    from .franka.control_reduced_exampleless import FrankaControlApiReducedExampleless
    from .franka.spill_wipe import FrankaControlSpillWipeApi
    from .franka.spill_wipe_privileged import FrankaControlSpillWipePrivilegedApi

    register_api("FrankaControlPrivilegedApi", FrankaControlPrivilegedApi)
    register_api("FrankaControlApi", lambda env: FrankaControlApi(env, use_sam3=True))
    register_api("FrankaControlApiReduced", FrankaControlApiReduced)
    register_api("FrankaControlApiReducedBimanual", lambda env: FrankaControlApiReduced(env, bimanual=True))
    register_api(
        "FrankaControlApiReducedExamplelessBimanual",
        lambda env: FrankaControlApiReducedExampleless(env, bimanual=True),
    )
    register_api(
        "FrankaControlApiReducedBimanualHandover",
        lambda env: FrankaControlApiReduced(env, bimanual=True, is_handover=True),
    )
    register_api(
        "FrankaControlApiReducedExamplelessBimanualHandover",
        lambda env: FrankaControlApiReducedExampleless(env, bimanual=True, is_handover=True),
    )
    register_api(
        "FrankaControlApiReducedSpillWipe",
        lambda env: FrankaControlApiReduced(env, tcp_offset=[0.0, 0.0, -0.0158]),
    )
    register_api("FrankaControlApiReducedExampleless", FrankaControlApiReducedExampleless)
    register_api("FrankaControlApiReducedSkillLibrary", FrankaControlApiReducedSkillLibrary)
    register_api(
        "FrankaControlApiReducedSkillLibraryBimanual",
        lambda env: FrankaControlApiReducedSkillLibrary(env, bimanual=True),
    )
    register_api(
        "FrankaControlApiReducedSkillLibrarySpillWipe",
        lambda env: FrankaControlApiReducedSkillLibrary(env, tcp_offset=[0.0, 0.0, -0.0158]),
    )
    register_api(
        "FrankaControlApiReducedSkillLibraryBimanualHandover",
        lambda env: FrankaControlApiReducedSkillLibrary(env, bimanual=True, is_handover=True),
    )
    register_api(
        "FrankaControlSpillWipeApi",
        lambda env: FrankaControlSpillWipeApi(env, tcp_offset=[0.0, 0.0, -0.0158], use_sam3=True),
    )
    register_api(
        "FrankaControlSpillWipeApiReduced",
        lambda env: FrankaControlApiReduced(env, tcp_offset=[0.0, 0.0, -0.0158], is_spill_wipe=True),
    )
    register_api(
        "FrankaControlSpillWipePrivilegedApi",
        lambda env: FrankaControlSpillWipePrivilegedApi(env, tcp_offset=[0.0, 0.0, -0.0158]),
    )
    register_api(
        "FrankaControlSpillWipeApiReducedExampleless",
        lambda env: FrankaControlApiReducedExampleless(env, tcp_offset=[0.0, 0.0, -0.0158], is_spill_wipe=True),
    )
    register_api("FrankaControlMultiPrivilegedApi", lambda env: FrankaControlPrivilegedApi(env, multi_turn=True))
    register_api(
        "FrankaRealReducedSkillLibraryControlApi",
        lambda env: FrankaControlApiReducedSkillLibrary(env, tcp_offset=[0.0, 0.0, -0.157], real=True),
    )
    register_api(
        "FrankaRealControlApi",
        lambda env: FrankaControlApi(env, tcp_offset=[0.0, 0.0, -0.157], real=True),
    )


def _register_franka_handover_and_lift() -> None:
    from .franka.handover_privileged import FrankaHandoverPrivilegedApi
    from .franka.handover import FrankaHandoverApi
    from .franka.handover_reduced import FrankaHandoverApiReduced
    from .franka.handover_reduced_exampleless import FrankaHandoverApiReducedExampleless
    from .franka.two_arm_lift import FrankaTwoArmLiftApi
    from .franka.two_arm_lift_privileged import FrankaTwoArmLiftPrivilegedApi
    from .franka.control_reduced import FrankaControlApiReduced
    from .franka.control_reduced_exampleless import FrankaControlApiReducedExampleless

    register_api("FrankaHandoverPrivilegedApi", FrankaHandoverPrivilegedApi)
    register_api("FrankaHandoverApi", FrankaHandoverApi)
    register_api("FrankaHandoverApiReduced", FrankaHandoverApiReduced)
    register_api("FrankaHandoverApiReducedExampleless", FrankaHandoverApiReducedExampleless)
    register_api("FrankaTwoArmLiftApi", FrankaTwoArmLiftApi)
    register_api("FrankaTwoArmLiftPrivilegedApi", FrankaTwoArmLiftPrivilegedApi)
    register_api("FrankaTwoArmLiftApiReduced", lambda env: FrankaControlApiReduced(env, bimanual=True, use_sam3=False))
    register_api(
        "FrankaTwoArmLiftApiReducedExampleless",
        lambda env: FrankaControlApiReducedExampleless(env, bimanual=True, use_sam3=False),
    )


def _register_nut_assembly() -> None:
    from .franka.nut_assembly_privileged import FrankaControlNutAssemblyPrivilegedApi
    from .franka.nut_assembly_visual import FrankaControlNutAssemblyVisualApi
    from .franka.control_reduced import FrankaControlApiReduced
    from .franka.control_reduced_exampleless import FrankaControlApiReducedExampleless

    register_api("FrankaControlNutAssemblyPrivilegedApi", FrankaControlNutAssemblyPrivilegedApi)
    register_api("FrankaControlNutAssemblyVisualApi", FrankaControlNutAssemblyVisualApi)
    register_api("FrankaControlNutAssemblyApiReduced", lambda env: FrankaControlApiReduced(env, is_peg_assembly=True))
    register_api(
        "FrankaControlNutAssemblyApiReducedExampleless",
        lambda env: FrankaControlApiReducedExampleless(env, is_peg_assembly=True),
    )


def _register_libero_privileged() -> None:
    from .franka.libero_privileged import FrankaLiberoPrivilegedApi

    register_api("FrankaLiberoPrivilegedApi", FrankaLiberoPrivilegedApi)


def _register_libero() -> None:
    _register_libero_privileged()

    from .franka.libero import FrankaLiberoApi
    from .franka.libero_reduced import FrankaLiberoApiReduced
    from .franka.libero_reduced_skill_library import FrankaLiberoApiReducedSkillLibrary

    register_api("FrankaLiberoApi", lambda env: FrankaLiberoApi(env, use_sam3=True))
    register_api(
        "FrankaLiberoGraspGenApi",
        lambda env: FrankaLiberoApi(env, use_sam3=True, grasp_backend="graspgen"),
    )
    register_api("FrankaLiberoApiReduced", FrankaLiberoApiReduced)
    register_api("FrankaLiberoApiReducedSkillLibrary", FrankaLiberoApiReducedSkillLibrary)
    register_api(
        "FrankaLiberoGraspGenApiReduced",
        lambda env: FrankaLiberoApiReduced(env, grasp_backend="graspgen"),
    )
    register_api(
        "FrankaLiberoGraspGenApiReducedSkillLibrary",
        lambda env: FrankaLiberoApiReducedSkillLibrary(env, grasp_backend="graspgen"),
    )
    # Opt-in variant: enables `grasp_with_wrist_closeloop` (RATS-side closed-loop
    # wrist-refined grasp added 2026-04-17, NOT in the CaP-X paper baseline).
    register_api(
        "FrankaLiberoApiReducedSkillLibraryWristCloseloop",
        lambda env: FrankaLiberoApiReducedSkillLibrary(env, enable_wrist_closeloop=True),
    )
    register_api(
        "FrankaLiberoGraspGenApiReducedSkillLibraryWristCloseloop",
        lambda env: FrankaLiberoApiReducedSkillLibrary(
            env,
            enable_wrist_closeloop=True,
            grasp_backend="graspgen",
        ),
    )


def _register_molmospaces() -> None:
    from .franka.molmospaces import FrankaMolmoSpacesApi, FrankaMolmoSpacesPrivilegedApi

    register_api("FrankaMolmoSpacesApi", FrankaMolmoSpacesApi)
    register_api("FrankaMolmoSpacesPrivilegedApi", FrankaMolmoSpacesPrivilegedApi)


def _register_molmospaces_control() -> None:
    from .franka.molmospaces_control import FrankaMolmoSpacesControlApi
    from .franka.molmospaces_reduced import FrankaMolmoSpacesApiReduced
    from .franka.molmospaces_reduced_skill_library import (
        FrankaMolmoSpacesApiReducedSkillLibrary,
    )

    register_api(
        "FrankaMolmoSpacesControlApi",
        lambda env: FrankaMolmoSpacesControlApi(env, use_sam3=True),
    )
    register_api(
        "FrankaMolmoSpacesApiReduced",
        lambda env: FrankaMolmoSpacesApiReduced(env, use_sam3=True),
    )
    register_api(
        "FrankaMolmoSpacesGraspGenApiReduced",
        lambda env: FrankaMolmoSpacesApiReduced(
            env,
            use_sam3=True,
            grasp_backend="graspgen",
        ),
    )
    register_api(
        "FrankaMolmoSpacesApiReducedWithHelpers",
        lambda env: FrankaMolmoSpacesApiReduced(
            env,
            use_sam3=True,
            enable_augmented_helpers=True,
            enable_filter_noise=True,
        ),
    )
    register_api(
        "FrankaMolmoSpacesApiReducedWithArmSpeed",
        lambda env: FrankaMolmoSpacesApiReduced(
            env, use_sam3=True, enable_arm_speed=True,
        ),
    )
    register_api(
        "FrankaMolmoSpacesApiReducedWithHelpersAndArmSpeed",
        lambda env: FrankaMolmoSpacesApiReduced(
            env,
            use_sam3=True,
            enable_augmented_helpers=True,
            enable_arm_speed=True,
            enable_filter_noise=True,
        ),
    )
    register_api(
        "FrankaMolmoSpacesApiReducedSkillLibrary",
        FrankaMolmoSpacesApiReducedSkillLibrary,
    )
    register_api(
        "FrankaMolmoSpacesGraspGenApiReducedSkillLibrary",
        lambda env: FrankaMolmoSpacesApiReducedSkillLibrary(
            env,
            grasp_backend="graspgen",
        ),
    )
    # Explicit MolmoSpaces names for baseline-vs-augmented comparisons.
    # The default skill-library name exposes perception helpers, but keeps
    # arm-speed control, grasp-selection helpers, raw language pointcloud
    # helpers, and wrist closed-loop grasp opt-in so prompt examples do not
    # steer default policies toward hidden APIs or hidden long retries.
    register_api(
        "FrankaMolmoSpacesApiReducedSkillLibraryBaseline",
        lambda env: FrankaMolmoSpacesApiReducedSkillLibrary(
            env,
            enable_augmented_helpers=False,
            enable_arm_speed=False,
            enable_filter_noise=False,
            enable_raw_language_pointcloud_helpers=False,
            enable_grasp_selection_helpers=False,
            enable_wrist_closeloop=False,
        ),
    )
    register_api(
        "FrankaMolmoSpacesApiReducedSkillLibraryWithHelpers",
        lambda env: FrankaMolmoSpacesApiReducedSkillLibrary(
            env,
            enable_augmented_helpers=True,
            enable_arm_speed=False,
            enable_filter_noise=True,
            enable_raw_language_pointcloud_helpers=True,
            enable_grasp_selection_helpers=True,
            enable_wrist_closeloop=False,
        ),
    )
    register_api(
        "FrankaMolmoSpacesApiReducedSkillLibraryWithArmSpeed",
        lambda env: FrankaMolmoSpacesApiReducedSkillLibrary(
            env,
            enable_augmented_helpers=False,
            enable_arm_speed=True,
            enable_filter_noise=False,
            enable_raw_language_pointcloud_helpers=False,
            enable_grasp_selection_helpers=False,
            enable_wrist_closeloop=False,
        ),
    )
    register_api(
        "FrankaMolmoSpacesApiReducedSkillLibraryWithHelpersAndArmSpeed",
        lambda env: FrankaMolmoSpacesApiReducedSkillLibrary(
            env,
            enable_augmented_helpers=True,
            enable_arm_speed=True,
            enable_filter_noise=True,
            enable_raw_language_pointcloud_helpers=True,
            enable_grasp_selection_helpers=True,
            enable_wrist_closeloop=False,
        ),
    )
    register_api(
        "FrankaMolmoSpacesApiReducedSkillLibraryWristCloseloop",
        lambda env: FrankaMolmoSpacesApiReducedSkillLibrary(
            env,
            enable_augmented_helpers=True,
            enable_arm_speed=False,
            enable_filter_noise=True,
            enable_raw_language_pointcloud_helpers=True,
            enable_grasp_selection_helpers=True,
            enable_wrist_closeloop=True,
        ),
    )
    register_api(
        "FrankaMolmoSpacesApiReducedSkillLibraryFull",
        lambda env: FrankaMolmoSpacesApiReducedSkillLibrary(
            env,
            enable_augmented_helpers=True,
            enable_arm_speed=True,
            enable_filter_noise=True,
            enable_raw_language_pointcloud_helpers=True,
            enable_grasp_selection_helpers=True,
            enable_wrist_closeloop=True,
        ),
    )


if _should_register("franka"):
    _safe_registration("Franka control stack", _register_franka_control)
if _should_register("robosuite"):
    _safe_registration("Franka handover/two-arm stack", _register_franka_handover_and_lift)
    _safe_registration("Franka nut assembly stack", _register_nut_assembly)
if _should_register("libero-privileged"):
    _safe_registration("LIBERO privileged stack", _register_libero_privileged)
elif _should_register("libero"):
    _safe_registration("LIBERO stack", _register_libero)
if _should_register("molmospaces"):
    _safe_registration("MolmoSpaces stack", _register_molmospaces)
    _safe_registration("MolmoSpaces control stack", _register_molmospaces_control)

__all__ = ["list_apis", "register_api"]
