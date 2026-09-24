"""Best-effort MolmoSpaces observation publisher for the RATS WebUI Viser pane."""

from __future__ import annotations

import logging
import math
import pathlib
import re
from collections.abc import Iterable
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from rats.utils.depth_utils import depth_color_to_pointcloud

logger = logging.getLogger("rats.molmospaces_viser")


def _as_array(value: Any, *, dtype: Any = np.float64) -> np.ndarray | None:
    if value is None:
        return None
    try:
        return np.asarray(value, dtype=dtype)
    except Exception:
        return None


def _quat_wxyz_from_matrix(rot: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a normalized wxyz quaternion."""
    m = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    quat = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm > 0:
        quat = quat / norm
    return quat


def _pose_from_mat(pose_mat: Any) -> tuple[np.ndarray, np.ndarray] | None:
    mat = _as_array(pose_mat)
    if mat is None or mat.shape != (4, 4):
        return None
    return mat[:3, 3], _quat_wxyz_from_matrix(mat[:3, :3])


def _safe_scene_name(value: Any, *, fallback: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or fallback)).strip("_")
    return safe[:80] or fallback


def _sphere_surface_points(
    center: np.ndarray,
    *,
    radius: float,
    count: int = 384,
) -> np.ndarray:
    """Return deterministic Fibonacci-lattice points on a sphere surface."""
    n = max(32, int(count))
    idx = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * idx / n)
    theta = math.pi * (1.0 + 5.0**0.5) * idx
    unit = np.column_stack(
        [
            np.cos(theta) * np.sin(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(phi),
        ]
    )
    return np.asarray(center, dtype=np.float64).reshape(3) + float(radius) * unit


def _extract_observation(env: Any) -> dict[str, Any] | None:
    candidates = [env, getattr(env, "low_level_env", None)]
    for obj in candidates:
        getter = getattr(obj, "get_observation", None)
        if callable(getter):
            try:
                obs = getter()
            except Exception:
                continue
            if isinstance(obs, dict):
                return obs
    return None


def _current_task_target_position(env: Any) -> tuple[np.ndarray, str] | None:
    """Resolve the current anchored task target position from live inventory."""
    low_level = getattr(env, "low_level_env", env)
    anchor_fn = getattr(low_level, "get_anchored_task_target", None)
    inventory_fn = getattr(low_level, "describe_scene_inventory", None)
    if not callable(inventory_fn):
        return None

    try:
        inventory = dict(inventory_fn() or {})
    except Exception:
        return None
    try:
        anchor = dict(anchor_fn() or {}) if callable(anchor_fn) else {}
    except Exception:
        anchor = {}
    if not anchor and isinstance(inventory.get("anchored_target"), dict):
        anchor = dict(inventory.get("anchored_target") or {})
    if not anchor:
        return None

    items = _flatten_inventory_for_focus(inventory)
    if not items:
        return None
    by_name = {str(item.get("internal_name")): item for item in items}

    for name in _anchored_focus_candidate_names(anchor):
        item = by_name.get(name) or _find_item_by_joint_name(items, name)
        if not item:
            continue
        pos = _as_array(item.get("position"), dtype=np.float64)
        if pos is None or pos.size < 3 or not np.isfinite(pos[:3]).all():
            continue
        label = item.get("category") or name or "task target"
        return pos[:3].astype(np.float64, copy=True), str(label)
    return None


def _flatten_inventory_for_focus(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for kind in ("pickables", "receptacles", "articulations"):
        for entry in inventory.get(kind, []) or []:
            if isinstance(entry, dict) and entry.get("internal_name"):
                items.append(entry)
    return items


def _anchored_focus_candidate_names(anchor: dict[str, Any]) -> list[str]:
    task_type = str(anchor.get("task_type") or "").strip().lower()
    if task_type in {"open", "close"}:
        keys = ("joint_name", "pickup_obj_name", "place_receptacle_name")
    elif task_type == "pick_and_place":
        keys = ("pickup_obj_name", "place_receptacle_name", "joint_name")
    else:
        keys = ("pickup_obj_name", "joint_name", "place_receptacle_name")

    names: list[str] = []
    for key in keys:
        value = anchor.get(key)
        if value is None:
            continue
        name = str(value).strip()
        if name and name not in names:
            names.append(name)
    return names


def _find_item_by_joint_name(
    items: list[dict[str, Any]],
    joint_name: str,
) -> dict[str, Any] | None:
    for item in items:
        for joint in item.get("joints") or []:
            if isinstance(joint, dict) and str(joint.get("name") or "") == joint_name:
                return item
    return None


class MolmoSpacesViserPublisher:
    """Own a Viser server and publish MolmoSpaces observations into it.

    This is deliberately best-effort: visualization must never fail or slow the
    robot loop materially. It publishes camera RGB/depth point clouds, camera
    frusta, robot base/EEF frames, and a compact textual status panel.
    """

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        ports: Iterable[int] = range(8080, 8090),
        max_points_per_camera: int = 9000,
        max_recorded_frames: int = 1200,
    ) -> None:
        self.host = host
        self.ports = list(ports)
        self.max_points_per_camera = int(max_points_per_camera)
        self.max_recorded_frames = int(max_recorded_frames)
        self.server: Any | None = None
        self.port: int | None = None
        self._image_handles: dict[str, Any] = {}
        self._publish_count = 0
        self._last_error: str | None = None
        self._last_camera_clouds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._last_robot_frames: dict[str, np.ndarray] = {}
        self._last_object_points: tuple[np.ndarray, str] | None = None
        self._last_molmo_point: dict[str, Any] | None = None
        self._last_robot_target: dict[str, Any] | None = None
        self._object_point_handles: dict[str, list[Any]] = {}
        self._molmo_point_handles: list[Any] = []
        self._molmo_point_image_handle: Any | None = None
        self._recorded_frames: list[dict[str, Any]] = []
        # MolmoSpaces benchmarks pin the robot anywhere in the scene's world
        # frame (procthor-10k house_0 sits near (6, 7); iTHOR house_17 near
        # (-1, 2)). Viser's default camera looks at the world origin, so the
        # robot/scene can render far off-center. Prefer the current task
        # target as the camera focus, falling back to the robot base until a
        # target can be resolved.
        self._latest_robot_base_xyz: np.ndarray | None = None
        self._latest_task_focus_xyz: np.ndarray | None = None
        self._latest_task_focus_label: str | None = None
        self._camera_focus_xyz: np.ndarray | None = None
        self._connected_clients: dict[int, Any] = {}

    @staticmethod
    def _set_handle_visible(handle: Any, visible: bool) -> None:
        """Best-effort Viser handle visibility toggle across API versions."""
        if handle is None:
            return
        for attr in ("visible", "show"):
            try:
                if hasattr(handle, attr):
                    setattr(handle, attr, bool(visible))
                    return
            except Exception:
                pass

    def _hide_previous_object_pointclouds(self) -> None:
        """Collapse stale target clouds so only the newest one is shown by default."""
        for handles in self._object_point_handles.values():
            for handle in handles:
                self._set_handle_visible(handle, False)

    @staticmethod
    def _molmo_point_overlay(
        image: np.ndarray,
        result: dict[str, Any],
        *,
        title: str,
    ) -> np.ndarray:
        """Return an RGB image with Molmo point-prompt coordinates drawn."""
        rgb = np.asarray(image, dtype=np.uint8)[..., :3].copy()
        pil = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)
        if pil.width >= 12 and pil.height >= 16:
            draw.rectangle([4, 4, min(pil.width - 4, 560), min(pil.height - 1, 28)], fill=(0, 0, 0))
            draw.text((8, 9), title[:95], fill=(255, 255, 255))
        for name, point in (result or {}).items():
            if point is None:
                continue
            try:
                x = int(round(float(point[0])))
                y = int(round(float(point[1])))
            except Exception:
                continue
            if x < 0 or y < 0 or x >= pil.width or y >= pil.height:
                continue
            outer = 13
            inner = 7
            draw.ellipse(
                [x - outer, y - outer, x + outer, y + outer],
                outline=(255, 255, 255),
                width=4,
            )
            draw.ellipse(
                [x - inner, y - inner, x + inner, y + inner],
                fill=(240, 82, 156),
            )
            draw.line([x - 18, y, x + 18, y], fill=(255, 255, 255), width=2)
            draw.line([x, y - 18, x, y + 18], fill=(255, 255, 255), width=2)
            draw.text((x + 16, max(0, y - 16)), str(name)[:50], fill=(255, 255, 255))
        return np.asarray(pil)

    def start(self) -> bool:
        if self.server is not None:
            return True
        try:
            import viser  # type: ignore
        except Exception as exc:
            self._last_error = f"viser import failed: {exc}"
            logger.warning("MolmoSpaces Viser publisher disabled: %s", self._last_error)
            return False

        for port in self.ports:
            try:
                self.server = viser.ViserServer(host=self.host, port=int(port))
                self.port = int(port)
                logger.info("MolmoSpaces Viser publisher running at http://localhost:%d", port)
                self._publish_static_scene()
                self._register_camera_centering()
                return True
            except Exception as exc:
                logger.debug("Could not start MolmoSpaces Viser on %s:%s: %s", self.host, port, exc)
                self.server = None
                self.port = None
        self._last_error = "no free Viser port in configured range"
        logger.warning("MolmoSpaces Viser publisher disabled: %s", self._last_error)
        return False

    def _publish_static_scene(self) -> None:
        if self.server is None:
            return
        try:
            self.server.scene.add_frame(
                "/world",
                position=(0.0, 0.0, 0.0),
                wxyz=(1.0, 0.0, 0.0, 0.0),
                axes_length=0.35,
                axes_radius=0.01,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Camera centering
    # ------------------------------------------------------------------
    def _register_camera_centering(self) -> None:
        """Center each client's camera on the current task target."""
        if self.server is None:
            return

        try:
            @self.server.on_client_connect
            def _on_connect(client: Any) -> None:
                try:
                    cid = int(getattr(client, "client_id", id(client)))
                except Exception:
                    cid = id(client)
                self._connected_clients[cid] = client
                self._apply_camera_focus(client, force=True)

            @self.server.on_client_disconnect
            def _on_disconnect(client: Any) -> None:
                try:
                    cid = int(getattr(client, "client_id", id(client)))
                except Exception:
                    cid = id(client)
                self._connected_clients.pop(cid, None)
        except Exception as exc:
            logger.debug("Could not register Viser client hooks: %s", exc)

        try:
            recenter_button = self.server.gui.add_button("Center camera on target")

            @recenter_button.on_click
            def _on_recenter(event: Any) -> None:
                client = getattr(event, "client", None)
                if client is None:
                    for c in list(self._connected_clients.values()):
                        self._apply_camera_focus(c, force=True)
                else:
                    self._apply_camera_focus(client, force=True)
        except Exception as exc:
            logger.debug("Could not register Viser recenter button: %s", exc)

    def _apply_camera_focus(self, client: Any, *, force: bool = False) -> None:
        """Best-effort camera framing for one Viser client."""
        if client is None:
            return
        focus = self._latest_task_focus_xyz
        if focus is None:
            focus = self._camera_focus_xyz
        if focus is None:
            focus = self._latest_robot_base_xyz
        if focus is None:
            return
        camera = getattr(client, "camera", None)
        if camera is None:
            return
        # Frame a compact workspace around the active task. Important:
        # Viser's CameraHandle.position setter preserves the current camera
        # orientation by translating look_at too, so set position first and
        # look_at second.
        offset = np.array([0.58, -0.58, 0.46], dtype=np.float64)
        eye = (focus + offset).astype(np.float64)
        target = focus.astype(np.float64)
        try:
            camera.position = tuple(float(v) for v in eye)
        except Exception:
            pass
        try:
            camera.look_at = tuple(float(v) for v in target)
        except Exception:
            pass
        try:
            camera.up_direction = (0.0, 0.0, 1.0)
        except Exception:
            pass
        # Mild zoom-in. Defaults are broad enough to make the task cloud and
        # goto_pose sphere look tiny in full-house MolmoSpaces scenes.
        try:
            camera.fov = 0.72
        except Exception:
            pass
        try:
            camera.near = 0.01
        except Exception:
            pass
        try:
            camera.far = 20.0
        except Exception:
            pass
        self._camera_focus_xyz = focus.copy()
        if force:
            label = self._latest_task_focus_label or "robot base"
            logger.debug(
                "Centered Viser camera on %s at (%.3f, %.3f, %.3f).",
                label,
                float(focus[0]),
                float(focus[1]),
                float(focus[2]),
            )

    def _update_robot_camera_fallback(self, base_xyz: np.ndarray) -> None:
        """Track the latest robot base; re-frame clients if it shifts."""
        if base_xyz is None or base_xyz.size < 3:
            return
        new_xyz = np.asarray(base_xyz[:3], dtype=np.float64)
        if not np.isfinite(new_xyz).all():
            return
        prev = self._latest_robot_base_xyz
        self._latest_robot_base_xyz = new_xyz
        if self._latest_task_focus_xyz is not None:
            return
        # Only re-center on the robot fallback when the robot moved
        # meaningfully (env recreation / house switch). 0.25 m guards against
        # per-step jitter so user-driven camera pans aren't yanked back on
        # every publish.
        if prev is None or float(np.linalg.norm(new_xyz - prev)) > 0.25:
            for client in list(self._connected_clients.values()):
                self._apply_camera_focus(client, force=False)

    def _update_task_camera_focus(self, focus_xyz: Any, *, label: str) -> None:
        """Track the active task target and re-frame connected clients."""
        pos = _as_array(focus_xyz, dtype=np.float64)
        if pos is None or pos.size < 3:
            return
        new_xyz = np.asarray(pos[:3], dtype=np.float64)
        if not np.isfinite(new_xyz).all():
            return
        prev = self._latest_task_focus_xyz
        self._latest_task_focus_xyz = new_xyz
        self._latest_task_focus_label = str(label or "task target")
        # Target updates are sparse and semantically meaningful: a house/task
        # switch, a segmented task cloud, or a Molmo point. Re-frame existing
        # clients when the target appears or shifts enough to matter.
        if prev is None or float(np.linalg.norm(new_xyz - prev)) > 0.08:
            for client in list(self._connected_clients.values()):
                self._apply_camera_focus(client, force=False)

    def _update_task_camera_focus_from_env(self, env: Any) -> None:
        target = _current_task_target_position(env)
        if target is None:
            return
        pos, label = target
        self._update_task_camera_focus(pos, label=label)

    def publish_env(self, env: Any, *, reason: str = "update") -> bool:
        if not self.start() or self.server is None:
            return False
        obs = _extract_observation(env)
        if not obs:
            return False
        try:
            self._publish_count += 1
            self._update_task_camera_focus_from_env(env)
            self._publish_robot_frames(obs)
            self._last_robot_frames = self._robot_frames_from_obs(obs)
            camera_clouds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for cam_name in ("agentview", "robot0_eye_in_hand"):
                cloud = self._publish_camera(obs, cam_name)
                if cloud is not None:
                    camera_clouds[cam_name] = cloud
            if camera_clouds:
                self._last_camera_clouds = camera_clouds
            self._record_snapshot(reason=reason)
            # Keep the Viser scene itself as the status surface; avoid adding
            # unbounded GUI markdown elements on frequent API updates.
            return True
        except Exception as exc:
            # Keep logs bounded; publishing happens frequently during API calls.
            msg = f"{type(exc).__name__}: {exc}"
            if msg != self._last_error:
                logger.debug("MolmoSpaces Viser publish failed: %s", msg, exc_info=True)
                self._last_error = msg
        return False

    def publish_object_points(
        self,
        points: Any,
        *,
        label: str = "task_object",
        score: float | None = None,
        max_points: int = 6000,
    ) -> bool:
        """Publish segmented task/object world points as a highlighted cloud."""
        if not self.start() or self.server is None:
            return False
        pts = _as_array(points, dtype=np.float64)
        if pts is None or pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
            return False
        finite = np.isfinite(pts).all(axis=1)
        pts = pts[finite]
        if len(pts) == 0:
            return False
        if len(pts) > max_points:
            idx = np.random.choice(len(pts), int(max_points), replace=False)
            pts = pts[idx]
        safe = _safe_scene_name(label, fallback="task_object")
        self._last_object_points = (pts.copy(), str(label or "task_object"))
        colors = np.tile(np.array([[1.0, 0.12, 0.08]], dtype=np.float32), (len(pts), 1))
        try:
            # A new attempt/object query should not leave every old red target
            # cloud enabled. Keep stale nodes in the Viser tree for inspection,
            # but collapse them by default so the current target is readable.
            self._hide_previous_object_pointclouds()
            cloud_handle = self.server.scene.add_point_cloud(
                f"/task_object_points/{safe}",
                points=pts.astype(np.float32),
                colors=colors,
                point_size=0.012,
                point_shape="circle",
            )
            center = pts.mean(axis=0)
            center_handle = self.server.scene.add_frame(
                f"/task_object_points/{safe}/center",
                position=tuple(map(float, center)),
                wxyz=(1.0, 0.0, 0.0, 0.0),
                axes_length=0.06,
                axes_radius=0.003,
            )
            self._set_handle_visible(cloud_handle, True)
            self._set_handle_visible(center_handle, True)
            self._object_point_handles[safe] = [cloud_handle, center_handle]
            self._update_task_camera_focus(center, label=f"task object: {label}")
            self._record_snapshot(reason=f"object_points:{safe}")
            return True
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            if msg != self._last_error:
                logger.debug("MolmoSpaces object-point publish failed: %s", msg, exc_info=True)
                self._last_error = msg
            return False

    def publish_molmo_point(
        self,
        image: Any,
        result: dict[str, Any],
        *,
        prompt: str,
        camera_name: str | None = None,
        world_point: Any | None = None,
        radius: float = 0.018,
    ) -> bool:
        """Publish Molmo's raw 2D point as a separate WebUI visualization.

        This intentionally visualizes the point prompt itself, before/independent
        of any SAM3 text-prompt mask.  When a caller also supplies a projected
        world point, a small magenta 3D marker is shown in Viser.
        """
        if not self.start() or self.server is None:
            return False
        rgb = _as_array(image, dtype=np.uint8)
        if rgb is None or rgb.ndim != 3 or rgb.shape[-1] < 3:
            return False
        rgb = rgb[..., :3]
        point = (result or {}).get(prompt)
        try:
            valid = bool(point is not None and point[0] is not None and point[1] is not None)
        except Exception:
            valid = False
        if not valid:
            return False

        safe = _safe_scene_name(prompt, fallback="molmo_point")
        title = (
            f"Molmo point prompt: {prompt}"
            + (f" [{camera_name}]" if camera_name else "")
        )
        overlay = self._molmo_point_overlay(rgb, result, title=title)
        published = False
        try:
            if self._molmo_point_image_handle is None:
                self._molmo_point_image_handle = self.server.gui.add_image(
                    overlay,
                    label="Latest Molmo point prompt",
                )
            else:
                self._molmo_point_image_handle.image = overlay
            published = True
        except Exception:
            pass

        world = _as_array(world_point, dtype=np.float64)
        if world is not None and world.size == 3 and np.isfinite(world).all():
            pos = world.reshape(3)
            sphere_points = _sphere_surface_points(pos, radius=float(radius), count=192)
            colors = np.tile(
                np.array([[1.0, 0.05, 0.62]], dtype=np.float32),
                (len(sphere_points), 1),
            )
            try:
                for handle in self._molmo_point_handles:
                    self._set_handle_visible(handle, False)
                self._molmo_point_handles = []
                cloud_handle = self.server.scene.add_point_cloud(
                    "/molmo_points/current_sphere",
                    points=sphere_points.astype(np.float32),
                    colors=colors,
                    point_size=max(0.008, float(radius) * 0.45),
                    point_shape="circle",
                )
                frame_handle = self.server.scene.add_frame(
                    "/molmo_points/current_frame",
                    position=tuple(map(float, pos)),
                    wxyz=(1.0, 0.0, 0.0, 0.0),
                    axes_length=max(0.06, float(radius) * 2.5),
                    axes_radius=0.003,
                )
                label_handle = None
                add_label = getattr(self.server.scene, "add_label", None)
                if callable(add_label):
                    try:
                        label_handle = add_label(
                            "/molmo_points/current_label",
                            text=f"Molmo: {prompt}",
                            position=tuple(map(float, pos + np.array([0.0, 0.0, float(radius) * 1.6]))),
                        )
                    except Exception:
                        label_handle = None
                self._molmo_point_handles = [
                    h for h in (cloud_handle, frame_handle, label_handle) if h is not None
                ]
                for handle in self._molmo_point_handles:
                    self._set_handle_visible(handle, True)
                self._last_molmo_point = {
                    "position": pos.copy(),
                    "label": str(prompt),
                    "camera_name": camera_name,
                    "pixel": [int(round(float(point[0]))), int(round(float(point[1])))],
                    "sphere_points": sphere_points.copy(),
                }
                self._update_task_camera_focus(pos, label=f"Molmo point: {prompt}")
                self._record_snapshot(reason=f"molmo_point:{safe}")
                published = True
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                if msg != self._last_error:
                    logger.debug("MolmoSpaces Molmo-point 3D publish failed: %s", msg, exc_info=True)
                    self._last_error = msg
        return published

    def publish_robot_target(
        self,
        position: Any,
        *,
        label: str = "reach_target",
        quaternion_wxyz: Any | None = None,
        radius: float = 0.015,
    ) -> bool:
        """Publish the current commanded robot reach target as a blue sphere.

        The marker is intentionally a small sphere cloud rather than a single
        point so it remains visible in the Web UI's embedded Viser view among
        dense scene/object point clouds.
        """
        if not self.start() or self.server is None:
            return False
        pos = _as_array(position, dtype=np.float64)
        if pos is None or pos.size != 3:
            return False
        pos = pos.reshape(3)
        if not np.isfinite(pos).all():
            return False
        quat = _as_array(quaternion_wxyz, dtype=np.float64)
        if quat is None or quat.size != 4 or not np.isfinite(quat).all():
            quat_tuple = (1.0, 0.0, 0.0, 0.0)
        else:
            quat_tuple = tuple(map(float, quat.reshape(4)))

        safe = _safe_scene_name(label, fallback="reach_target")
        sphere_points = _sphere_surface_points(pos, radius=float(radius))
        self._last_robot_target = {
            "position": pos.copy(),
            "quaternion_wxyz": np.asarray(quat_tuple, dtype=np.float64),
            "label": str(label or "reach_target"),
            "radius": float(radius),
            "sphere_points": sphere_points.copy(),
        }
        colors = np.tile(np.array([[0.05, 0.35, 1.0]], dtype=np.float32), (len(sphere_points), 1))
        try:
            # Stable "current" path keeps the latest commanded target obvious
            # without accumulating hundreds of markers during interpolation.
            self.server.scene.add_point_cloud(
                "/robot/reach_target/current_sphere",
                points=sphere_points.astype(np.float32),
                colors=colors,
                point_size=max(0.01, float(radius) * 0.35),
                point_shape="circle",
            )
            self.server.scene.add_frame(
                "/robot/reach_target/current_frame",
                position=tuple(map(float, pos)),
                wxyz=quat_tuple,
                axes_length=max(0.08, float(radius) * 2.0),
                axes_radius=0.004,
            )
            # Also publish a labeled path for debugging/scene-tree inspection.
            self.server.scene.add_frame(
                f"/robot/reach_target/{safe}",
                position=tuple(map(float, pos)),
                wxyz=quat_tuple,
                axes_length=0.05,
                axes_radius=0.0025,
            )
            add_label = getattr(self.server.scene, "add_label", None)
            if callable(add_label):
                try:
                    add_label(
                        "/robot/reach_target/current_label",
                        text=str(label),
                        position=tuple(map(float, pos + np.array([0.0, 0.0, float(radius) * 1.4]))),
                    )
                except Exception:
                    # Older/newer Viser versions vary; the blue sphere/frame
                    # are the durable visualization.
                    pass
            self._record_snapshot(reason=f"robot_target:{safe}")
            return True
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            if msg != self._last_error:
                logger.debug("MolmoSpaces robot-target publish failed: %s", msg, exc_info=True)
                self._last_error = msg
            return False

    def _robot_frames_from_obs(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        frames: dict[str, np.ndarray] = {}
        ee = _as_array(obs.get("robot_cartesian_pos"))
        if ee is not None and ee.size >= 7 and np.isfinite(ee[:3]).all():
            frames["end_effector"] = ee[:3].astype(np.float64, copy=True)
        base = _as_array(obs.get("robot_base_pose"))
        if base is not None and base.size >= 7 and np.isfinite(base[:3]).all():
            frames["base"] = base[:3].astype(np.float64, copy=True)
        return frames

    def _record_snapshot(self, *, reason: str) -> None:
        """Store a compact copy of the current Viser-published state."""
        if self.max_recorded_frames <= 0:
            return
        if (
            not self._last_camera_clouds
            and self._last_robot_target is None
            and self._last_object_points is None
            and self._last_molmo_point is None
        ):
            return
        camera_clouds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, (points, colors) in self._last_camera_clouds.items():
            pts = np.asarray(points, dtype=np.float32)
            cols = np.asarray(colors, dtype=np.float32)
            if len(pts) > 2500:
                idx = np.linspace(0, len(pts) - 1, 2500).astype(int)
                pts = pts[idx]
                cols = cols[idx]
            camera_clouds[name] = (pts.copy(), cols.copy())
        object_points = None
        if self._last_object_points is not None:
            pts, label = self._last_object_points
            pts = np.asarray(pts, dtype=np.float32)
            if len(pts) > 2000:
                idx = np.linspace(0, len(pts) - 1, 2000).astype(int)
                pts = pts[idx]
            object_points = (pts.copy(), label)
        robot_target = None
        if self._last_robot_target is not None:
            robot_target = {
                key: (value.copy() if isinstance(value, np.ndarray) else value)
                for key, value in self._last_robot_target.items()
            }
        molmo_point = None
        if self._last_molmo_point is not None:
            molmo_point = {
                key: (value.copy() if isinstance(value, np.ndarray) else value)
                for key, value in self._last_molmo_point.items()
            }
        self._recorded_frames.append(
            {
                "reason": reason,
                "camera_clouds": camera_clouds,
                "robot_frames": {
                    key: value.copy() for key, value in self._last_robot_frames.items()
                },
                "object_points": object_points,
                "molmo_point": molmo_point,
                "robot_target": robot_target,
            }
        )
        if len(self._recorded_frames) > self.max_recorded_frames:
            del self._recorded_frames[: len(self._recorded_frames) - self.max_recorded_frames]

    def recorded_frame_count(self) -> int:
        """Return the number of compact Viser snapshots retained in memory."""
        return len(self._recorded_frames)

    def export_recording(
        self,
        output_dir: str | pathlib.Path,
        *,
        fps: int = 8,
        filename: str = "viser_visualization.mp4",
        subdir: str = "viser_recording",
        frame_start: int | None = None,
        frame_end: int | None = None,
        metadata_extra: dict[str, Any] | None = None,
        export_frame_archive: bool = True,
    ) -> pathlib.Path | None:
        """Render the recorded Viser-published state to an MP4 artifact.

        This is a server-side reconstruction from the same point clouds,
        object points, robot frames, and blue target sphere sent to Viser. It is
        intentionally best-effort and not a literal browser-screen recording.

        ``frame_start``/``frame_end`` slice the retained snapshot list.  RATS
        uses that to persist one Viser-style playback per policy attempt while
        still exporting the whole-run recording at completion.
        """
        total_frames = len(self._recorded_frames)
        if total_frames <= 0:
            return None
        start = 0 if frame_start is None else max(0, int(frame_start))
        end = total_frames if frame_end is None else max(start, min(total_frames, int(frame_end)))
        frames = list(self._recorded_frames[start:end])
        if not frames:
            return None
        out_dir = pathlib.Path(output_dir) / str(subdir)
        out_dir.mkdir(parents=True, exist_ok=True)
        video_path = out_dir / filename
        try:
            import imageio
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            self._last_error = f"viser recording export dependencies unavailable: {exc}"
            logger.warning("MolmoSpaces Viser recording export skipped: %s", exc)
            return None

        all_points: list[np.ndarray] = []
        for frame in frames:
            for points, _colors in frame.get("camera_clouds", {}).values():
                if len(points):
                    all_points.append(np.asarray(points)[:, :3])
            obj = frame.get("object_points")
            if obj is not None and len(obj[0]):
                all_points.append(np.asarray(obj[0])[:, :3])
            target = frame.get("robot_target")
            if target is not None:
                all_points.append(np.asarray(target["sphere_points"])[:, :3])
            molmo = frame.get("molmo_point")
            if molmo is not None:
                all_points.append(np.asarray(molmo["sphere_points"])[:, :3])
        if not all_points:
            return None
        stacked = np.concatenate(all_points, axis=0)
        finite = np.isfinite(stacked).all(axis=1)
        stacked = stacked[finite]
        if len(stacked) == 0:
            return None
        lo = np.percentile(stacked, 1, axis=0)
        hi = np.percentile(stacked, 99, axis=0)
        center = (lo + hi) / 2.0
        span = float(max(np.max(hi - lo), 0.25))
        limits = np.column_stack([center - span * 0.55, center + span * 0.55])

        rendered: list[np.ndarray] = []
        for idx, frame in enumerate(frames):
            fig = plt.figure(figsize=(8, 6), dpi=120)
            ax = fig.add_subplot(111, projection="3d")
            for _cam_name, (points, colors) in frame.get("camera_clouds", {}).items():
                points = np.asarray(points)
                colors = np.clip(np.asarray(colors), 0.0, 1.0)
                if len(points):
                    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=0.7, alpha=0.28)
            obj = frame.get("object_points")
            if obj is not None:
                obj_points, obj_label = obj
                obj_points = np.asarray(obj_points)
                if len(obj_points):
                    ax.scatter(
                        obj_points[:, 0],
                        obj_points[:, 1],
                        obj_points[:, 2],
                        c=np.array([[1.0, 0.12, 0.08]]),
                        s=3.0,
                        alpha=0.9,
                        label=f"object: {obj_label}",
                    )
            target = frame.get("robot_target")
            if target is not None:
                sphere_points = np.asarray(target["sphere_points"])
                if len(sphere_points):
                    ax.scatter(
                        sphere_points[:, 0],
                        sphere_points[:, 1],
                        sphere_points[:, 2],
                        c=np.array([[0.05, 0.35, 1.0]]),
                        s=10.0,
                        alpha=1.0,
                        label=f"target: {target.get('label', 'reach_target')}",
                    )
            molmo = frame.get("molmo_point")
            if molmo is not None:
                sphere_points = np.asarray(molmo["sphere_points"])
                if len(sphere_points):
                    ax.scatter(
                        sphere_points[:, 0],
                        sphere_points[:, 1],
                        sphere_points[:, 2],
                        c=np.array([[1.0, 0.05, 0.62]]),
                        s=9.0,
                        alpha=1.0,
                        label=f"molmo: {molmo.get('label', 'point')}",
                    )
            robot_frames = frame.get("robot_frames", {})
            for name, point in robot_frames.items():
                point = np.asarray(point)
                ax.scatter(point[0], point[1], point[2], c="black", s=24.0, marker="x")
                ax.text(point[0], point[1], point[2], name, fontsize=7)
            ax.set_xlim(limits[0, 0], limits[0, 1])
            ax.set_ylim(limits[1, 0], limits[1, 1])
            ax.set_zlim(limits[2, 0], limits[2, 1])
            ax.set_xlabel("world x")
            ax.set_ylabel("world y")
            ax.set_zlabel("world z")
            ax.view_init(elev=23, azim=-58)
            ax.set_title(f"Viser reconstruction {idx + 1}/{len(frames)} — {frame.get('reason', '')}")
            if ax.get_legend_handles_labels()[0]:
                ax.legend(loc="upper right", fontsize=7)
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            rendered.append(rgba[:, :, :3].copy())
            plt.close(fig)
        imageio.mimsave(str(video_path), rendered, fps=int(fps))
        metadata_path = out_dir / "viser_visualization_metadata.json"
        try:
            import json

            metadata = {
                "video_path": str(video_path),
                "frame_count": len(rendered),
                "fps": int(fps),
                "frame_start": start,
                "frame_end": end,
                "total_recorded_frames_at_export": total_frames,
                "reasons": [str(frame.get("reason", "")) for frame in frames],
                "note": (
                    "Server-side reconstruction from Viser-published "
                    "point clouds/object points/Molmo point markers/"
                    "robot target markers; "
                    "not a literal browser capture."
                ),
            }
            if metadata_extra:
                metadata.update(metadata_extra)
            metadata_path.write_text(
                json.dumps(metadata, indent=2, default=str)
            )
        except Exception:
            pass
        if export_frame_archive:
            try:
                self._write_frame_archive(out_dir, frames, frame_start=start, metadata_extra=metadata_extra)
            except Exception as exc:
                logger.debug("MolmoSpaces Viser frame archive export failed: %s", exc)
        return video_path

    def _write_frame_archive(
        self,
        out_dir: pathlib.Path,
        frames: list[dict[str, Any]],
        *,
        frame_start: int,
        metadata_extra: dict[str, Any] | None = None,
    ) -> None:
        """Persist compact Viser snapshots for future interactive replay tools.

        The MP4 above is immediately viewable.  This archive keeps the same
        reduced point clouds and markers in a deterministic ``npz`` + JSON
        manifest so a future WebUI can reconstruct/scrub the Viser scene
        without rerunning the robot policy.
        """
        import json

        arrays: dict[str, np.ndarray] = {}
        manifest: dict[str, Any] = {
            "schema_version": "rats_molmospaces_viser_frames_v1",
            "frame_start": int(frame_start),
            "frame_count": len(frames),
            "frames": [],
        }
        if metadata_extra:
            manifest["metadata"] = metadata_extra

        def safe_name(value: Any, fallback: str) -> str:
            return _safe_scene_name(value, fallback=fallback).replace(".", "_")

        for local_idx, frame in enumerate(frames):
            prefix = f"f{local_idx:04d}"
            frame_meta: dict[str, Any] = {
                "index": local_idx,
                "global_index": int(frame_start + local_idx),
                "reason": str(frame.get("reason", "")),
                "camera_clouds": {},
                "robot_frames": {},
            }
            for cam_name, (points, colors) in frame.get("camera_clouds", {}).items():
                cam_safe = safe_name(cam_name, fallback="camera")
                points_key = f"{prefix}_camera_{cam_safe}_points"
                colors_key = f"{prefix}_camera_{cam_safe}_colors"
                arrays[points_key] = np.asarray(points, dtype=np.float32)
                arrays[colors_key] = np.asarray(colors, dtype=np.float32)
                frame_meta["camera_clouds"][str(cam_name)] = {
                    "points": points_key,
                    "colors": colors_key,
                    "count": int(len(points)),
                }
            obj = frame.get("object_points")
            if obj is not None:
                obj_points, obj_label = obj
                key = f"{prefix}_object_points"
                arrays[key] = np.asarray(obj_points, dtype=np.float32)
                frame_meta["object_points"] = {
                    "points": key,
                    "label": str(obj_label),
                    "count": int(len(obj_points)),
                }
            target = frame.get("robot_target")
            if target is not None:
                key = f"{prefix}_robot_target_sphere"
                arrays[key] = np.asarray(target["sphere_points"], dtype=np.float32)
                frame_meta["robot_target"] = {
                    "sphere_points": key,
                    "label": str(target.get("label", "reach_target")),
                    "position": np.asarray(target.get("position"), dtype=float).tolist(),
                    "quaternion_wxyz": np.asarray(
                        target.get("quaternion_wxyz", [1.0, 0.0, 0.0, 0.0]),
                        dtype=float,
                    ).tolist(),
                    "radius": float(target.get("radius", 0.015)),
                }
            molmo = frame.get("molmo_point")
            if molmo is not None:
                key = f"{prefix}_molmo_point_sphere"
                arrays[key] = np.asarray(molmo["sphere_points"], dtype=np.float32)
                frame_meta["molmo_point"] = {
                    "sphere_points": key,
                    "label": str(molmo.get("label", "point")),
                    "position": np.asarray(molmo.get("position"), dtype=float).tolist(),
                    "camera_name": molmo.get("camera_name"),
                    "pixel": molmo.get("pixel"),
                }
            for name, point in frame.get("robot_frames", {}).items():
                key = f"{prefix}_robot_frame_{safe_name(name, fallback='frame')}"
                arrays[key] = np.asarray(point, dtype=np.float32)
                frame_meta["robot_frames"][str(name)] = key
            manifest["frames"].append(frame_meta)

        if arrays:
            np.savez_compressed(out_dir / "viser_frames.npz", **arrays)
        manifest["npz_path"] = "viser_frames.npz" if arrays else None
        (out_dir / "viser_frames_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str)
        )

    def _publish_robot_frames(self, obs: dict[str, Any]) -> None:
        if self.server is None:
            return
        ee = _as_array(obs.get("robot_cartesian_pos"))
        if ee is not None and ee.size >= 7:
            self.server.scene.add_frame(
                "/robot/end_effector",
                position=tuple(map(float, ee[:3])),
                wxyz=tuple(map(float, ee[3:7])),
                axes_length=0.12,
                axes_radius=0.004,
            )
        base = _as_array(obs.get("robot_base_pose"))
        if base is not None and base.size >= 7:
            self.server.scene.add_frame(
                "/robot/base",
                position=tuple(map(float, base[:3])),
                wxyz=tuple(map(float, base[3:7])),
                axes_length=0.18,
                axes_radius=0.006,
            )
            self._update_robot_camera_fallback(base[:3])

    def _publish_camera(
        self, obs: dict[str, Any], cam_name: str
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if self.server is None:
            return None
        cam = obs.get(cam_name)
        if not isinstance(cam, dict):
            return None
        images = cam.get("images") if isinstance(cam.get("images"), dict) else {}
        rgb = _as_array(images.get("rgb"), dtype=np.uint8)
        depth = _as_array(images.get("depth"), dtype=np.float64)
        intrinsics = _as_array(cam.get("intrinsics"), dtype=np.float64)
        pose = _pose_from_mat(cam.get("pose_mat"))
        if rgb is not None and rgb.ndim == 3 and rgb.shape[-1] == 3:
            if cam_name not in self._image_handles:
                try:
                    self._image_handles[cam_name] = self.server.gui.add_image(
                        rgb,
                        label=f"{cam_name} RGB",
                    )
                except Exception:
                    self._image_handles[cam_name] = None
            handle = self._image_handles.get(cam_name)
            if handle is not None:
                try:
                    handle.image = rgb
                except Exception:
                    pass
        if pose is not None:
            position, wxyz = pose
            self.server.scene.add_frame(
                f"/cameras/{cam_name}",
                position=tuple(map(float, position)),
                wxyz=tuple(map(float, wxyz)),
                axes_length=0.08,
                axes_radius=0.003,
            )
            if rgb is not None and intrinsics is not None and intrinsics.shape == (3, 3):
                h, w = rgb.shape[:2]
                fy = float(intrinsics[1, 1]) if float(intrinsics[1, 1]) else 1.0
                fov = 2.0 * math.atan(float(h) / (2.0 * fy))
                try:
                    self.server.scene.add_camera_frustum(
                        name=f"/cameras/{cam_name}/frustum",
                        position=tuple(map(float, position)),
                        wxyz=tuple(map(float, wxyz)),
                        fov=float(fov),
                        aspect=float(w) / float(h),
                        scale=0.12,
                        image=rgb,
                    )
                except TypeError:
                    self.server.scene.add_camera_frustum(
                        name=f"/cameras/{cam_name}/frustum",
                        position=tuple(map(float, position)),
                        wxyz=tuple(map(float, wxyz)),
                        fov=float(fov),
                        aspect=float(w) / float(h),
                        scale=0.12,
                    )
        if (
            rgb is not None
            and depth is not None
            and intrinsics is not None
            and intrinsics.shape == (3, 3)
            and pose is not None
        ):
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            if depth.ndim != 2 or depth.shape[:2] != rgb.shape[:2]:
                return None
            # Keep browser payload bounded. Subsample to roughly max_points.
            pixels = max(1, int(depth.shape[0] * depth.shape[1]))
            subsample = max(1, int(math.sqrt(pixels / max(1, self.max_points_per_camera))))
            points_cam, colors = depth_color_to_pointcloud(
                depth,
                rgb,
                intrinsics,
                subsample_factor=subsample,
                depth_clip_range=(0.02, 6.0),
            )
            if len(points_cam) == 0:
                return None
            pos, _ = pose
            mat = _as_array(cam.get("pose_mat"), dtype=np.float64)
            if mat is None or mat.shape != (4, 4):
                return None
            points_h = np.concatenate([points_cam, np.ones((len(points_cam), 1))], axis=1)
            points_world = (mat @ points_h.T).T[:, :3]
            self.server.scene.add_point_cloud(
                f"/pointclouds/{cam_name}",
                points=points_world.astype(np.float32),
                colors=colors.astype(np.float32),
                point_size=0.006 if cam_name == "agentview" else 0.004,
                point_shape="square",
            )
            return points_world.astype(np.float32), colors.astype(np.float32)
        return None

    def _publish_status(self, reason: str) -> None:
        if self.server is None:
            return
        text = f"MolmoSpaces live view — update {self._publish_count} ({reason})"
        try:
            self.server.gui.add_markdown(text)
        except Exception:
            # Older/newer viser versions differ here; status is non-essential.
            pass
