# SPDX-License-Identifier: GPL-3.0-or-later
"""GPU drawing for onion skin overlays.

Design: geometry frame cache.

On every ``frame_change_post`` event the handler evaluates the required
before/after frames. If a frame is already cached, it reuses the geometry.
If not, it evaluates the scene and caches the GPU batches.
Ghost markers (user-pinned persistent overlays) are still supported and
stored in cache.py.
"""

import bpy
import blf
import gpu
from gpu_extras.batch import batch_for_shader
from bpy_extras.view3d_utils import location_3d_to_region_2d
from bpy.app.handlers import persistent

from . import cache

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def screen_animation_playing(context):
    """Check if viewport animation is currently playing."""
    try:
        return context.screen and context.screen.is_animation_playing
    except (AttributeError, ReferenceError):
        return False


# Removed get_mesh_objects, now done inline per object in _build_onion_batches.

def extract_all_meshes_merged(objects, depsgraph, wireframe=False):
    """Merge all objects into a single vertex/index array pair (and normals).

    Returns (verts, normals, indices, prim_type) or (None, None, None, None).
    """
    if HAS_NUMPY:
        return _extract_numpy(objects, depsgraph, wireframe)
    return _extract_simple(objects, depsgraph, wireframe)


def _extract_numpy(objects, depsgraph, wireframe):
    all_verts = []
    all_normals = []
    all_indices = []
    vertex_offset = 0

    for obj in objects:
        try:
            eval_obj = obj.evaluated_get(depsgraph)
            mesh = None
            try:
                mesh = eval_obj.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
            except RuntimeError:
                continue

            if not mesh or len(mesh.vertices) == 0:
                if mesh:
                    eval_obj.to_mesh_clear()
                continue

            try:
                num_verts = len(mesh.vertices)
                mesh.transform(eval_obj.matrix_world)

                verts_co = np.empty(num_verts * 3, dtype=np.float32)
                mesh.vertices.foreach_get('co', verts_co)
                verts_world = verts_co.reshape(-1, 3)

                all_verts.append(verts_world)

                if not wireframe:
                    verts_no = np.empty(num_verts * 3, dtype=np.float32)
                    mesh.vertices.foreach_get('normal', verts_no)
                    normals_world = verts_no.reshape(-1, 3)
                    all_normals.append(normals_world)

                if wireframe:
                    num_edges = len(mesh.edges)
                    if num_edges > 0:
                        edge_data = np.empty(num_edges * 2, dtype=np.int32)
                        mesh.edges.foreach_get('vertices', edge_data)
                        edge_data += vertex_offset
                        all_indices.append(edge_data.reshape(-1, 2))
                else:
                    mesh.calc_loop_triangles()
                    num_tris = len(mesh.loop_triangles)
                    if num_tris > 0:
                        tri_data = np.empty(num_tris * 3, dtype=np.int32)
                        mesh.loop_triangles.foreach_get('vertices', tri_data)
                        tri_data += vertex_offset
                        all_indices.append(tri_data.reshape(-1, 3))

                vertex_offset += num_verts
            finally:
                eval_obj.to_mesh_clear()

        except Exception:
            continue

    if not all_verts or not all_indices:
        return None, None, None, None

    merged_verts = np.vstack(all_verts)
    merged_indices = np.vstack(all_indices)
    merged_normals = np.vstack(all_normals) if all_normals else None
    prim_type = 'LINES' if wireframe else 'TRIS'
    return merged_verts, merged_normals, merged_indices, prim_type


def _extract_simple(objects, depsgraph, wireframe):
    all_verts = []
    all_normals = []
    all_indices = []
    vertex_offset = 0

    for obj in objects:
        try:
            eval_obj = obj.evaluated_get(depsgraph)
            mesh = None
            try:
                mesh = eval_obj.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
            except RuntimeError:
                continue

            if not mesh or len(mesh.vertices) == 0:
                if mesh:
                    eval_obj.to_mesh_clear()
                continue

            try:
                mesh.transform(eval_obj.matrix_world)

                all_verts.extend(v.co[:] for v in mesh.vertices)
                if not wireframe:
                    all_normals.extend(v.normal[:] for v in mesh.vertices)

                if wireframe:
                    if vertex_offset == 0:
                        all_indices.extend(e.vertices[:] for e in mesh.edges)
                    else:
                        all_indices.extend((e.vertices[0] + vertex_offset,
                                            e.vertices[1] + vertex_offset) for e in mesh.edges)
                else:
                    mesh.calc_loop_triangles()
                    if vertex_offset == 0:
                        all_indices.extend(t.vertices[:] for t in mesh.loop_triangles)
                    else:
                        all_indices.extend((t.vertices[0] + vertex_offset,
                                            t.vertices[1] + vertex_offset,
                                            t.vertices[2] + vertex_offset) for t in mesh.loop_triangles)

                vertex_offset += len(mesh.vertices)
            finally:
                eval_obj.to_mesh_clear()

        except Exception:
            continue

    if not all_verts or not all_indices:
        return None, None, None, None

    prim_type = 'LINES' if wireframe else 'TRIS'
    return all_verts, all_normals if all_normals else None, all_indices, prim_type


# ---------------------------------------------------------------------------
# Frame geometry helpers
# ---------------------------------------------------------------------------

def get_frame_range(settings, scene):
    """Return (start, end) for the effective frame range."""
    if settings.use_frame_range:
        return settings.frame_range_start, settings.frame_range_end
    return scene.frame_start, scene.frame_end


def get_keyframe_range(onion_skin_objects):
    """Find the first and last keyframes across all tracked objects (including Armatures)."""
    min_frame = float('inf')
    max_frame = float('-inf')

    for item in onion_skin_objects:
        obj = item.object
        if obj and item.visible:
            if obj.animation_data and obj.animation_data.action:
                fcurve_range = obj.animation_data.action.frame_range
                min_frame = min(min_frame, fcurve_range[0])
                max_frame = max(max_frame, fcurve_range[1])

    if min_frame == float('inf'):
        return None, None
    return min_frame, max_frame


def get_keyframe_range_for_object(obj):
    """Find the first and last keyframes for a single object.

    For armatures this also checks the armature's own action so that
    pose-bone keyframes are respected.
    For meshes this also checks parent/modifier armature actions.
    """
    min_frame = float('inf')
    max_frame = float('-inf')

    def _check_action(id_data):
        nonlocal min_frame, max_frame
        if id_data and getattr(id_data, 'animation_data', None) and id_data.animation_data.action:
            fr = id_data.animation_data.action.frame_range
            min_frame = min(min_frame, fr[0])
            max_frame = max(max_frame, fr[1])

    # Check the object's own action
    _check_action(obj)

    # For armatures with mesh children, also consider child actions
    if obj.type == 'ARMATURE':
        for child in obj.children_recursive:
            _check_action(child)
    elif obj.type == 'MESH':
        if obj.parent and obj.parent.type == 'ARMATURE':
            _check_action(obj.parent)
        for mod in obj.modifiers:
            if mod.type == 'ARMATURE' and mod.object:
                _check_action(mod.object)

    if min_frame == float('inf'):
        return None, None
    return min_frame, max_frame


def get_keyframes_for_object(obj, include_children=True):
    """Find all distinct keyframe frame numbers for an object and optional children."""
    keyframes = set()

    def _extract_from_anim_data(ad):
        if not ad or not ad.action:
            return
        action = ad.action
        # Blender < 4.4 legacy action
        if hasattr(action, 'fcurves') and action.fcurves:
            for fc in action.fcurves:
                for kp in fc.keyframe_points:
                    keyframes.add(int(round(kp.co[0])))
        # Blender 4.4+ / 5.x layered action
        if hasattr(action, 'layers'):
            target_slot_handle = getattr(ad, 'action_slot_handle', None)
            for layer in action.layers:
                for strip in layer.strips:
                    if hasattr(strip, 'channelbags'):
                        for bag in strip.channelbags:
                            if target_slot_handle is not None and hasattr(bag, 'slot_handle'):
                                if bag.slot_handle != target_slot_handle:
                                    continue
                            if hasattr(bag, 'fcurves'):
                                for fc in bag.fcurves:
                                    for kp in fc.keyframe_points:
                                        keyframes.add(int(round(kp.co[0])))

    def _collect_from_id(id_data):
        if not id_data:
            return
        if hasattr(id_data, 'animation_data'):
            _extract_from_anim_data(id_data.animation_data)
        if hasattr(id_data, 'shape_keys') and id_data.shape_keys:
            if hasattr(id_data.shape_keys, 'animation_data'):
                _extract_from_anim_data(id_data.shape_keys.animation_data)

    _collect_from_id(obj)
    if hasattr(obj, 'data'):
        _collect_from_id(obj.data)

    if include_children and obj.type == 'ARMATURE':
        for child in obj.children_recursive:
            _collect_from_id(child)
            if hasattr(child, 'data'):
                _collect_from_id(child.data)
    elif obj.type == 'MESH':
        if obj.parent and obj.parent.type == 'ARMATURE':
            _collect_from_id(obj.parent)
        for mod in obj.modifiers:
            if mod.type == 'ARMATURE' and mod.object:
                _collect_from_id(mod.object)

    return sorted(keyframes)


def get_needed_frames(current, settings, scene, onion_skin_objects=None, obj=None):
    """Return list of (frame, direction, t, dist) for the current position.

    When *obj* is provided the keyframe range and keyframes are computed for that
    single object so that keyframe modes work independently per object.
    Falls back to the legacy all-objects behaviour when only
    *onion_skin_objects* is given.
    """
    range_start, range_end = get_frame_range(settings, scene)

    if settings.use_frame_range and (current < range_start or current > range_end):
        return []

    # Retrieve all keyframes if keyframe filtering or limit_to_keyframes is needed
    kf_mode = getattr(settings, 'keyframe_mode', 'OFF')
    range_mode = getattr(settings, 'keyframe_range_mode', 'COUNT')
    obj_keyframes = None

    if kf_mode != 'OFF' or settings.limit_to_keyframes:
        if obj is not None:
            obj_keyframes = get_keyframes_for_object(obj, include_children=settings.include_children)
        elif onion_skin_objects:
            all_keys = set()
            for item in onion_skin_objects:
                if item.object and item.visible:
                    all_keys.update(get_keyframes_for_object(item.object, include_children=item.object.onion_skin.include_children))
            obj_keyframes = sorted(all_keys)

    if settings.limit_to_keyframes:
        if obj_keyframes:
            range_start = max(range_start, obj_keyframes[0])
            range_end = min(range_end, obj_keyframes[-1])
        else:
            # Fallback to keyframe range search
            if obj is not None:
                kf_start, kf_end = get_keyframe_range_for_object(obj)
            elif onion_skin_objects:
                kf_start, kf_end = get_keyframe_range(onion_skin_objects)
            else:
                kf_start, kf_end = None, None
            if kf_start is not None and kf_end is not None:
                range_start = max(range_start, int(kf_start))
                range_end = min(range_end, int(kf_end))

    frames = []
    step = max(1, settings.frame_step)

    # -------------------------------------------------------------------------
    # Mode: OFF (Regular Frame Step)
    # -------------------------------------------------------------------------
    if kf_mode == 'OFF':
        if settings.show_before:
            max_before = settings.frames_before
            inv_max = 1.0 / max_before if max_before > 0 else 0
            for i in range(step, max_before + 1, step):
                f = current - i
                if range_start <= f <= range_end:
                    frames.append((f, 'before', i * inv_max, i))

        if settings.show_after:
            max_after = settings.frames_after
            inv_max = 1.0 / max_after if max_after > 0 else 0
            for i in range(step, max_after + 1, step):
                f = current + i
                if range_start <= f <= range_end:
                    frames.append((f, 'after', i * inv_max, i))

        return frames

    # -------------------------------------------------------------------------
    # Modes: KEYFRAMES_ONLY and KEYFRAMES_AND_STEP
    # -------------------------------------------------------------------------
    keyframes_list = obj_keyframes if obj_keyframes is not None else []
    before_keys = [k for k in keyframes_list if range_start <= k < current]
    after_keys = [k for k in keyframes_list if current < k <= range_end]

    # Process BEFORE
    if settings.show_before and settings.frames_before > 0:
        val_before = settings.frames_before
        if kf_mode == 'KEYFRAMES_ONLY':
            if range_mode == 'COUNT':
                chosen = before_keys[-val_before:] if before_keys else []
                total = len(chosen)
                for rank, k in enumerate(reversed(chosen), 1):
                    dist = current - k
                    t = rank / total if total > 0 else 1.0
                    frames.append((k, 'before', t, dist))
            else:  # FRAMES span
                chosen = [k for k in before_keys if current - k <= val_before]
                for k in reversed(chosen):
                    dist = current - k
                    t = dist / val_before
                    frames.append((k, 'before', t, dist))

        elif kf_mode == 'KEYFRAMES_AND_STEP':
            if range_mode == 'COUNT':
                chosen_keys = before_keys[-val_before:] if before_keys else []
                if chosen_keys:
                    min_boundary = min(chosen_keys)
                else:
                    min_boundary = max(range_start, current - val_before * step)
                max_span = max(current - min_boundary, 1)
            else:  # FRAMES span
                chosen_keys = [k for k in before_keys if current - k <= val_before]
                min_boundary = max(range_start, current - val_before)
                max_span = max(val_before, 1)

            pts = set(chosen_keys)
            for f in range(current - step, min_boundary - 1, -step):
                if range_start <= f <= range_end:
                    pts.add(f)

            for f in sorted(pts, reverse=True):
                dist = current - f
                t = min(1.0, dist / max_span)
                frames.append((f, 'before', t, dist))

    # Process AFTER
    if settings.show_after and settings.frames_after > 0:
        val_after = settings.frames_after
        if kf_mode == 'KEYFRAMES_ONLY':
            if range_mode == 'COUNT':
                chosen = after_keys[:val_after] if after_keys else []
                total = len(chosen)
                for rank, k in enumerate(chosen, 1):
                    dist = k - current
                    t = rank / total if total > 0 else 1.0
                    frames.append((k, 'after', t, dist))
            else:  # FRAMES span
                chosen = [k for k in after_keys if k - current <= val_after]
                for k in chosen:
                    dist = k - current
                    t = dist / val_after
                    frames.append((k, 'after', t, dist))

        elif kf_mode == 'KEYFRAMES_AND_STEP':
            if range_mode == 'COUNT':
                chosen_keys = after_keys[:val_after] if after_keys else []
                if chosen_keys:
                    max_boundary = max(chosen_keys)
                else:
                    max_boundary = min(range_end, current + val_after * step)
                max_span = max(max_boundary - current, 1)
            else:  # FRAMES span
                chosen_keys = [k for k in after_keys if k - current <= val_after]
                max_boundary = min(range_end, current + val_after)
                max_span = max(val_after, 1)

            pts = set(chosen_keys)
            for f in range(current + step, max_boundary + 1, step):
                if range_start <= f <= range_end:
                    pts.add(f)

            for f in sorted(pts):
                dist = f - current
                t = min(1.0, dist / max_span)
                frames.append((f, 'after', t, dist))

    return frames


def calculate_alpha(base_alpha, t, falloff, curve):
    """Calculate ghost alpha based on distance and falloff settings."""
    if curve == 'linear':
        factor = 1.0 - t * falloff
    elif curve == 'exponential':
        factor = (1.0 - t) ** (1.0 + falloff * 2.0)
    else:  # smooth
        ts = t * falloff
        if ts >= 1.0:
            factor = 0.1
        elif ts <= 0.0:
            factor = 1.0
        else:
            factor = 1.0 - ts * ts * (3.0 - 2.0 * ts)

    return base_alpha * max(0.1, factor)


def _safe_redraw(context=None):
    """Safely trigger viewport redraw across all windows."""
    try:
        wm_list = []
        if context and getattr(context, "window_manager", None):
            wm_list.append(context.window_manager)
        for wm in bpy.data.window_managers:
            if wm not in wm_list:
                wm_list.append(wm)
        for wm in wm_list:
            for window in wm.windows:
                screen = window.screen
                if screen:
                    for area in screen.areas:
                        if area.type == 'VIEW_3D':
                            area.tag_redraw()
    except (AttributeError, ReferenceError, Exception):
        pass


# ---------------------------------------------------------------------------
# Animation State Tracking
# ---------------------------------------------------------------------------

_tracked_action_signatures = {}  # {object_name: signature}


def _get_anim_data_signature(id_data):
    """Extract a signature tuple for the animation data of a single ID block."""
    if not id_data or not getattr(id_data, 'animation_data', None):
        return None
    ad = id_data.animation_data
    act = ad.action
    act_name = act.name if act else None

    # In Blender 4.4+ / 5.x, handle Action Slots
    slot_handle = getattr(ad, 'action_slot_handle', None)
    slot_id = None
    if hasattr(ad, 'action_slot') and ad.action_slot:
        slot = ad.action_slot
        slot_id = getattr(slot, 'identifier', None) or getattr(slot, 'name_display', None) or getattr(slot, 'handle', None)

    tweak = getattr(ad, 'use_tweak_mode', False)
    return (act_name, slot_handle, slot_id, tweak)


def get_object_anim_signature(obj):
    """Return a comprehensive hashable signature of all animation data driving an object."""
    if not obj:
        return None
    sigs = []

    # 1. Object's own animation_data
    sigs.append(('self', _get_anim_data_signature(obj)))

    # 2. Object's data animation_data (e.g. armature or mesh data)
    if hasattr(obj, 'data') and obj.data:
        sigs.append(('data', _get_anim_data_signature(obj.data)))
        # Shape keys animation_data
        if hasattr(obj.data, 'shape_keys') and obj.data.shape_keys:
            sigs.append(('shape_keys', _get_anim_data_signature(obj.data.shape_keys)))

    # 3. If Armature, include children animation data
    if obj.type == 'ARMATURE':
        for child in obj.children_recursive:
            sigs.append(('child_' + child.name, _get_anim_data_signature(child)))
            if hasattr(child, 'data') and hasattr(child.data, 'shape_keys') and child.data.shape_keys:
                sigs.append(('child_sk_' + child.name, _get_anim_data_signature(child.data.shape_keys)))

    # 4. If Mesh, include parent Armature or Armature Modifier's animation data
    elif obj.type == 'MESH':
        if obj.parent and obj.parent.type == 'ARMATURE':
            sigs.append(('parent_arm', _get_anim_data_signature(obj.parent)))
        for mod in obj.modifiers:
            if mod.type == 'ARMATURE' and mod.object:
                sigs.append(('mod_arm_' + mod.object.name, _get_anim_data_signature(mod.object)))

    return tuple(sigs)


def check_action_changes(scene):
    """Check if any tracked object's animation signature has changed.

    Returns True if any object's action/slot/animation state changed.
    If changed, it invalidates the frame cache and onion batches.
    """
    global _tracked_action_signatures
    any_changed = False
    current_names = set()

    for item in scene.onion_skin_objects:
        obj = item.object
        if not obj:
            continue
        current_names.add(obj.name)
        new_sig = get_object_anim_signature(obj)
        old_sig = _tracked_action_signatures.get(obj.name)

        if old_sig is not None and old_sig != new_sig:
            cache.clear_frame_cache(obj.name)
            any_changed = True

        _tracked_action_signatures[obj.name] = new_sig

    removed = [k for k in _tracked_action_signatures if k not in current_names]
    for k in removed:
        del _tracked_action_signatures[k]

    if any_changed:
        cache.clear_onion_batches()

    return any_changed


# ---------------------------------------------------------------------------
# Batch building (called from frame_change handler — NOT from draw callback)
# ---------------------------------------------------------------------------

_is_building = False

def _extract_object_batches(obj, dg, settings):
    """Extract GPU batches for a tracked object (and its children if armature) from depsgraph."""
    objects_to_extract = []
    _geo_types = {'MESH', 'CURVE', 'SURFACE', 'FONT'}
    if obj.type in _geo_types:
        objects_to_extract.append(obj)
    elif obj.type == 'ARMATURE' and settings.include_children:
        def add_children_recursive(parent):
            for child in parent.children:
                if child.type in _geo_types:
                    objects_to_extract.append(child)
                if child.children:
                    add_children_recursive(child)
        add_children_recursive(obj)

    if not objects_to_extract:
        return None

    batches_list = []
    for ext_obj in objects_to_extract:
        verts, normals, indices, prim_type = extract_all_meshes_merged(
            [ext_obj], dg, settings.use_wireframe
        )
        if verts is not None and indices is not None:
            if settings.use_fresnel and not settings.use_wireframe and normals is not None:
                current_shader = cache.get_fresnel_shader()
                batch = batch_for_shader(current_shader, prim_type, {"pos": verts, "normal": normals}, indices=indices)
            else:
                current_shader = cache.get_shader()
                batch = batch_for_shader(current_shader, prim_type, {"pos": verts}, indices=indices)
            batches_list.append((ext_obj.name, batch))

    return batches_list if batches_list else None


def _build_onion_batches(context, current, depsgraph):
    """Evaluate all ghost frames and build GPU batches.

    Only ghost frames (frame != current) call scene.frame_set().
    The current frame is extracted directly from the incoming depsgraph.
    """
    global _is_building
    if _is_building:
        return

    scene = context.scene
    global_settings = scene.onion_skin_settings

    if not global_settings.enabled or not scene.onion_skin_objects:
        cache.clear_onion_batches()
        return

    new_batches = []
    original_frame = scene.frame_current
    _is_building = True

    obj_needed_frames = {}
    frames_to_eval = set()

    for item in scene.onion_skin_objects:
        obj = item.object
        if not obj or not item.visible:
            continue
        settings = obj.onion_skin
        needed = get_needed_frames(current, settings, scene, obj=obj)
        obj_needed_frames[obj.name] = needed

        for f, _, _, _ in needed:
            if f != current and not cache.has_frame(obj.name, f):
                frames_to_eval.add(f)

    frames_to_eval = list(frames_to_eval)

    try:
        for frame in frames_to_eval:
            scene.frame_set(frame)
            dg = context.evaluated_depsgraph_get()

            for item in scene.onion_skin_objects:
                obj = item.object
                if not obj or not item.visible:
                    continue

                if not cache.has_frame(obj.name, frame):
                    batches = _extract_object_batches(obj, dg, obj.onion_skin)
                    cache.put_frame(obj.name, frame, batches)
    finally:
        if frames_to_eval:
            scene.frame_set(original_frame)
        _is_building = False

    # For current frame: extract directly from depsgraph without calling scene.frame_set()
    for item in scene.onion_skin_objects:
        obj = item.object
        if not obj or not item.visible:
            continue
        if not cache.has_frame(obj.name, current):
            batches = _extract_object_batches(obj, depsgraph, obj.onion_skin)
            cache.put_frame(obj.name, current, batches)

    for item in scene.onion_skin_objects:
        obj = item.object
        if not obj or not item.visible:
            continue
        needed = obj_needed_frames.get(obj.name, [])
        for frame, direction, t, dist in needed:
            batch = cache.get_frame(obj.name, frame)
            if batch is not None:
                new_batches.append((obj.name, frame, direction, t, dist, batch))

    cache.set_onion_batches(new_batches)

    # Collect mask batches for current frame
    mask_batches = {}
    for item in scene.onion_skin_objects:
        obj = item.object
        if not obj or not item.visible:
            continue
        cur_batch = cache.get_frame(obj.name, current)
        if cur_batch:
            mask_batches[obj.name] = cur_batch
    cache.set_mask_batches(mask_batches)


# ---------------------------------------------------------------------------
# Draw callbacks  (must be fast — no scene changes allowed here)
# ---------------------------------------------------------------------------

def draw_onion_skins():
    """Main draw callback for all onion skin overlays."""
    context = bpy.context

    if not context.area or context.area.type != 'VIEW_3D':
        return

    scene = context.scene
    if not hasattr(scene, 'onion_skin_settings'):
        return

    global_settings = scene.onion_skin_settings
    if not global_settings.enabled or not scene.onion_skin_objects:
        return

    if any(item.object is None for item in scene.onion_skin_objects):
        return

    onion_batches = cache.get_onion_batches()
    ghost_markers = cache.get_ghost_markers()

    if not onion_batches and not ghost_markers:
        return

    # Write near-plane depth for current frame objects to prevent ghost overlap
    mask_batches = cache.get_mask_batches()
    any_on_top = any(
        (bpy.data.objects.get(name) and bpy.data.objects.get(name).onion_skin.keep_current_on_top)
        for name in mask_batches
    )
    if any_on_top:
        mask_shader = cache.get_mask_shader()
        if mask_shader:
            mv_matrix = gpu.matrix.get_model_view_matrix()
            mvp_matrix = gpu.matrix.get_projection_matrix() @ mv_matrix
            mask_shader.bind()
            mask_shader.uniform_float("ModelViewProjectionMatrix", mvp_matrix)

            gpu.state.color_mask_set(False, False, False, False)
            gpu.state.depth_mask_set(True)
            gpu.state.depth_test_set('ALWAYS')

            for obj_name, cur_batch in mask_batches.items():
                obj = bpy.data.objects.get(obj_name)
                if not obj or not obj.visible_get() or not obj.onion_skin.keep_current_on_top:
                    continue
                if isinstance(cur_batch, list):
                    for child_name, child_batch in cur_batch:
                        child_obj = bpy.data.objects.get(child_name)
                        if not child_obj or not child_obj.visible_get():
                            continue
                        child_batch.draw(mask_shader)
                elif cur_batch:
                    cur_batch.draw(mask_shader)

            gpu.state.color_mask_set(True, True, True, True)

    gpu.state.blend_set('ALPHA')
    gpu.state.depth_mask_set(False)

    # Onion skin frames (before/after)
    if onion_batches:
        # Pre-compute view matrices and camera vector once for Fresnel shader
        mv_matrix = gpu.matrix.get_model_view_matrix()
        mvp_matrix = gpu.matrix.get_projection_matrix() @ mv_matrix

        rv3d = context.space_data.region_3d
        is_ortho = 0.0 if rv3d.is_perspective else 1.0
        vm = rv3d.view_matrix
        if is_ortho > 0.5:
            # Orthographic: view forward direction is row 2 of view_matrix
            view_vec = (vm[2][0], vm[2][1], vm[2][2], 1.0)
        else:
            # Perspective: camera world position from inverted view_matrix translation column
            inv = vm.inverted()
            view_vec = (inv[0][3], inv[1][3], inv[2][3], 0.0)

        # Draw farthest frames first (painter's order)
        for obj_name, frame, direction, t, dist, batch in sorted(onion_batches, key=lambda x: -x[4]):
            obj = bpy.data.objects.get(obj_name)
            if not obj or not obj.visible_get(): continue
            settings = obj.onion_skin
            
            if settings.use_xray:
                # Clear background depth if X-Ray is on, then re-apply current object depth mask if keep_current_on_top
                if settings.keep_current_on_top:
                    fb = gpu.state.active_framebuffer_get()
                    fb.clear(depth=1.0)
                    mask_shader = cache.get_mask_shader()
                    if mask_shader:
                        mask_shader.bind()
                        mask_shader.uniform_float("ModelViewProjectionMatrix", mvp_matrix)
                        gpu.state.color_mask_set(False, False, False, False)
                        gpu.state.depth_mask_set(True)
                        gpu.state.depth_test_set('ALWAYS')
                        cur_batch = mask_batches.get(obj_name)
                        if isinstance(cur_batch, list):
                            for child_name, child_batch in cur_batch:
                                child_obj = bpy.data.objects.get(child_name)
                                if child_obj and child_obj.visible_get():
                                    child_batch.draw(mask_shader)
                        elif cur_batch:
                            cur_batch.draw(mask_shader)
                        gpu.state.color_mask_set(True, True, True, True)
                        gpu.state.depth_mask_set(False)
                    gpu.state.depth_test_set('LESS')
                else:
                    gpu.state.depth_test_set('NONE')
            else:
                gpu.state.depth_test_set('LESS' if settings.keep_current_on_top else 'LESS_EQUAL')
            if settings.use_wireframe:
                gpu.state.line_width_set(1.5)
            else:
                gpu.state.line_width_set(1.0)
                
            color_before = tuple(settings.color_before)
            color_after = tuple(settings.color_after)
            falloff = settings.opacity_falloff
            curve = settings.falloff_curve

            base = color_before if direction == 'before' else color_after
            alpha = calculate_alpha(base[3], t, falloff, curve)
            
            if settings.use_fresnel and not settings.use_wireframe:
                current_shader = cache.get_fresnel_shader()
                current_shader.bind()
                current_shader.uniform_float("ModelViewProjectionMatrix", mvp_matrix)
                current_shader.uniform_float("view_vector", view_vec)
                current_shader.uniform_float("color", (base[0], base[1], base[2], alpha))
                current_shader.uniform_float("fresnel_params", (settings.fresnel_power, settings.fresnel_fade, 0.0, 0.0))
            else:
                current_shader = cache.get_shader()
                current_shader.bind()
                current_shader.uniform_float("color", (base[0], base[1], base[2], alpha))
                
            if isinstance(batch, list):
                for child_name, child_batch in batch:
                    child_obj = bpy.data.objects.get(child_name)
                    if not child_obj or not child_obj.visible_get(): continue
                    child_batch.draw(current_shader)
            else:
                batch.draw(current_shader)

    # Ghost markers (persistent pinned frames)
    if ghost_markers and global_settings.show_ghosts:
        shader = cache.get_shader()
        shader.bind()
        gc = tuple(global_settings.ghost_color)
        shader.uniform_float("color", gc)
        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.state.line_width_set(1.0)
        for i in range(len(ghost_markers)):
            ghost_batch = cache.get_ghost_batch(i)
            if ghost_batch:
                ghost_batch.draw(shader)

    gpu.state.blend_set('NONE')
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(True)
    gpu.state.line_width_set(1.0)


def draw_ghost_labels():
    """Draw frame number labels on ghost overlays (POST_PIXEL)."""
    context = bpy.context

    if not context.area or context.area.type != 'VIEW_3D':
        return

    scene = context.scene
    if not hasattr(scene, 'onion_skin_settings'):
        return

    settings = scene.onion_skin_settings
    if not settings.enabled or not settings.show_ghost_labels or not settings.show_ghosts or not scene.onion_skin_objects:
        return

    ghost_markers = cache.get_ghost_markers()
    if not ghost_markers:
        return

    region = context.region
    rv3d = context.space_data.region_3d

    font_id = 0
    blf.size(font_id, 13)
    blf.color(font_id, 1.0, 1.0, 1.0, 1.0)

    for marker in ghost_markers:
        frame_num, verts = marker[0], marker[1]
        n = len(verts)
        if n == 0:
            continue

        cx = cy = cz = 0.0
        for v in verts:
            cx += v[0]
            cy += v[1]
            cz += v[2]
        cx /= n
        cy /= n
        cz /= n

        pos_2d = location_3d_to_region_2d(region, rv3d, (cx, cy, cz))
        if pos_2d:
            blf.position(font_id, pos_2d.x + 5, pos_2d.y + 5, 0)
            blf.draw(font_id, str(frame_num))


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _cleanup_handlers():
    """Remove all our handlers by name (prevents duplicates on reload)."""
    global _tracked_action_signatures
    _tracked_action_signatures.clear()
    our_names = {'_on_frame_change', '_on_depsgraph_update', '_on_undo_redo'}
    for handler_list in (bpy.app.handlers.frame_change_post,
                         bpy.app.handlers.depsgraph_update_post,
                         bpy.app.handlers.undo_post,
                         bpy.app.handlers.redo_post):
        for fn in [h for h in handler_list if getattr(h, '__name__', None) in our_names]:
            try:
                handler_list.remove(fn)
            except ValueError:
                pass


@persistent
def _on_frame_change(scene, depsgraph):
    """Rebuild ghost batches whenever the frame changes."""
    global _is_building
    if _is_building:
        return

    if not hasattr(scene, 'onion_skin_settings'):
        return
    settings = scene.onion_skin_settings
    if not settings.enabled or not scene.onion_skin_objects:
        cache.clear_onion_batches()
        return
    if any(item.object is None for item in scene.onion_skin_objects):
        return

    # Invalidate stale frame cache if action changed (including when switching actions during playback)
    check_action_changes(scene)

    context = bpy.context
    _build_onion_batches(context, scene.frame_current, depsgraph)


import time
_build_time = 0

def _do_build():
    global _build_time
    # Debounce: Wait 0.02s after the last update before building
    if time.time() - _build_time < 0.02:
        return 0.01

    context = bpy.context
    if not context or not getattr(context, 'scene', None):
        return None

    # Do not interrupt playback
    if getattr(context.screen, 'is_animation_playing', False):
        return None

    scene = context.scene
    depsgraph = context.evaluated_depsgraph_get()
    _build_onion_batches(context, scene.frame_current, depsgraph)
    _safe_redraw(context)
    return None

def _trigger_build(immediate=False):
    global _build_time
    _build_time = time.time()
    if immediate:
        _build_time = 0
    if not bpy.app.timers.is_registered(_do_build):
        interval = 0.001 if immediate else 0.02
        bpy.app.timers.register(_do_build, first_interval=interval)


@persistent
def _on_depsgraph_update(scene, depsgraph):
    """Rebuild ghost batches when tracked objects are edited (e.g. pose change, action switch).

    Also handles stale (deleted) object cleanup.
    """
    global _is_building
    if _is_building:
        return

    if not hasattr(scene, 'onion_skin_settings'):
        return

    # Stale object cleanup
    stale = [i for i, item in enumerate(scene.onion_skin_objects) if item.object is None]
    if stale:
        for i in reversed(stale):
            scene.onion_skin_objects.remove(i)
        cache.clear_frame_cache()
        cache.clear_onion_batches()
        cache.clear_ghost_markers()
        scene.onion_skin_settings.marker_count = 0
        if scene.onion_skin_active_index >= len(scene.onion_skin_objects):
            scene.onion_skin_active_index = max(0, len(scene.onion_skin_objects) - 1)
        _safe_redraw(bpy.context)
        return

    global_settings = scene.onion_skin_settings
    if not global_settings.enabled or not scene.onion_skin_objects:
        return

    if getattr(bpy.context.screen, 'is_animation_playing', False):
        return

    # 1. Check if any tracked object's animation action or slot changed
    action_changed = check_action_changes(scene)

    needs_update = action_changed
    if not needs_update:
        is_transforming = any(u.is_updated_transform for u in depsgraph.updates)
        for update in depsgraph.updates:
            # Direct geometry data-block update (Edit mode mesh edits, curve changes, etc.)
            if isinstance(update.id, (bpy.types.Mesh, bpy.types.Curve, bpy.types.SurfaceCurve, bpy.types.TextCurve)):
                needs_update = True
                cache.clear_frame_cache()
                cache.clear_onion_batches()
                break
            # Object modifier or shape key updates in Object Mode (not active transforms)
            elif isinstance(update.id, bpy.types.Object) and update.id.type in {'MESH', 'ARMATURE', 'CURVE', 'SURFACE', 'FONT'}:
                if update.is_updated_geometry and not update.is_updated_transform and bpy.context.mode == 'OBJECT':
                    needs_update = True
                    cache.clear_frame_cache()
                    cache.clear_onion_batches()
                    break
            # Action update (e.g. keyframe edited in DopeSheet/GraphEditor when not actively transforming)
            elif isinstance(update.id, bpy.types.Action):
                if not is_transforming and bpy.context.mode not in {'POSE', 'PAINT_WEIGHT', 'SCULPT'}:
                    needs_update = True
                    cache.clear_frame_cache()
                    cache.clear_onion_batches()
                    break

    if needs_update:
        _trigger_build(immediate=action_changed)
        if action_changed:
            _safe_redraw(bpy.context)


@persistent
def _on_undo_redo(scene):
    """Rebuild after undo/redo."""
    _tracked_action_signatures.clear()
    cache.clear_frame_cache()
    cache.clear_onion_batches()
    cache.clear_ghost_markers()
    _trigger_build(immediate=True)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_draw_handler = None
_label_handler = None


def register():
    global _draw_handler, _label_handler
    _cleanup_handlers()

    _draw_handler = bpy.types.SpaceView3D.draw_handler_add(
        draw_onion_skins, (), 'WINDOW', 'POST_VIEW'
    )
    _label_handler = bpy.types.SpaceView3D.draw_handler_add(
        draw_ghost_labels, (), 'WINDOW', 'POST_PIXEL'
    )
    bpy.app.handlers.frame_change_post.append(_on_frame_change)
    bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update)
    bpy.app.handlers.undo_post.append(_on_undo_redo)
    bpy.app.handlers.redo_post.append(_on_undo_redo)


def unregister():
    global _draw_handler, _label_handler
    if _draw_handler:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handler, 'WINDOW')
        _draw_handler = None
    if _label_handler:
        bpy.types.SpaceView3D.draw_handler_remove(_label_handler, 'WINDOW')
        _label_handler = None

    _cleanup_handlers()
