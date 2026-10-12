# SPDX-License-Identifier: GPL-3.0-or-later
"""GPU batch storage for onion skin overlays.

Provides a frame cache to avoid evaluating the scene multiple times.
Geometry is cached by frame and cleared when tracked objects are edited.
Ghost markers (user-pinned frames) persist across frame changes.
"""

import gpu
from gpu_extras.batch import batch_for_shader

_shader = None
_fresnel_shader = None
_mask_shader = None
_mask_batches = {}  # {object_name: [(child_name, gpu_batch), ...]}

# Current ghost frame batches — rebuilt on every frame change.
# List of (object_name, frame, direction, t, dist, gpu_batch)
_onion_batches = []

import collections

# Frame cache to avoid re-evaluating the scene for already evaluated frames
# Key: (object_name, frame)
_frame_cache = collections.OrderedDict()
MAX_CACHE_SIZE = 500

def get_frame(object_name, frame):
    key = (object_name, frame)
    if key in _frame_cache:
        _frame_cache.move_to_end(key)
        return _frame_cache[key]
    return None

def has_frame(object_name, frame):
    return (object_name, frame) in _frame_cache

def put_frame(object_name, frame, batch):
    key = (object_name, frame)
    if key in _frame_cache:
        _frame_cache.move_to_end(key)
    _frame_cache[key] = batch
    if len(_frame_cache) > MAX_CACHE_SIZE:
        _frame_cache.popitem(last=False)

def clear_frame_cache(object_name=None):
    if object_name is None:
        _frame_cache.clear()
    else:
        keys_to_delete = [k for k in _frame_cache.keys() if k[0] == object_name]
        for k in keys_to_delete:
            del _frame_cache[k]

# Ghost marker cache — persists until user removes them.
# List of (frame, verts, indices, prim_type)
_ghost_markers = []
_ghost_batches = {}   # {index: gpu_batch}


# ---------------------------------------------------------------------------
# Shader
# ---------------------------------------------------------------------------

def get_shader():
    """Get or create the GPU shader."""
    global _shader
    if _shader is None:
        try:
            _shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        except ValueError:
            try:
                _shader = gpu.shader.from_builtin('3D_UNIFORM_COLOR')
            except ValueError:
                _shader = gpu.shader.from_builtin('FLAT_COLOR')
    return _shader


def get_mask_shader():
    """Get or create the GPU shader for depth masking the current frame."""
    global _mask_shader
    if _mask_shader is None:
        shader_info = gpu.types.GPUShaderCreateInfo()
        shader_info.vertex_in(0, 'VEC3', 'pos')
        shader_info.push_constant('MAT4', 'ModelViewProjectionMatrix')
        shader_info.fragment_out(0, 'VEC4', 'fragColor')
        shader_info.vertex_source("""
            void main() {
                vec4 clip = ModelViewProjectionMatrix * vec4(pos, 1.0);
                gl_Position = vec4(clip.xy, -clip.w, clip.w);
            }
        """)
        shader_info.fragment_source("""
            void main() {
                fragColor = vec4(0.0);
            }
        """)
        _mask_shader = gpu.shader.create_from_info(shader_info)
    return _mask_shader


def set_mask_batches(batches):
    global _mask_batches
    _mask_batches = batches


def get_mask_batches():
    return _mask_batches


def clear_mask_batches():
    global _mask_batches
    _mask_batches = {}


def get_fresnel_shader():
    """Get or create the GPU shader for Fresnel effect.

    Calculates Fresnel directly in world space using the camera origin / view direction,
    requiring only ModelViewProjectionMatrix (64B) + view_vector (16B) + color (16B)
    + fresnel_params (16B) = 112 bytes total. This strictly stays under the 128-byte
    Vulkan Push Constant hardware minimum limit, completely eliminating driver warnings.
    """
    global _fresnel_shader
    if _fresnel_shader is None:
        vert_out = gpu.types.GPUStageInterfaceInfo("fresnel_interface")
        vert_out.smooth('FLOAT', "v_fresnel")

        shader_info = gpu.types.GPUShaderCreateInfo()
        shader_info.vertex_in(0, 'VEC3', "pos")
        shader_info.vertex_in(1, 'VEC3', "normal")
        shader_info.vertex_out(vert_out)

        # Uniform push constants: 64 + 16 + 16 + 16 = 112 bytes (<= 128 bytes limit)
        shader_info.push_constant('MAT4', "ModelViewProjectionMatrix")
        shader_info.push_constant('VEC4', "view_vector")   # xyz: camera world pos (persp) or forward dir (ortho), w: is_ortho
        shader_info.push_constant('VEC4', "color")
        shader_info.push_constant('VEC4', "fresnel_params") # x: power, y: fade

        shader_info.fragment_out(0, 'VEC4', "fragColor")

        shader_info.vertex_source('''
            void main() {
                gl_Position = ModelViewProjectionMatrix * vec4(pos, 1.0);
                vec3 view_dir;
                if (view_vector.w > 0.5) {
                    view_dir = normalize(view_vector.xyz);
                } else {
                    view_dir = normalize(view_vector.xyz - pos);
                }
                vec3 world_normal = normalize(normal);
                float f = 1.0 - abs(dot(world_normal, view_dir));
                v_fresnel = pow(f, fresnel_params.x);
            }
        ''')

        shader_info.fragment_source('''
            void main() {
                fragColor = vec4(color.rgb, color.a * mix(1.0, v_fresnel, fresnel_params.y));
            }
        ''')

        _fresnel_shader = gpu.shader.create_from_info(shader_info)
    return _fresnel_shader


# ---------------------------------------------------------------------------
# Live onion batches (current frame set — no persistence across frames)
# ---------------------------------------------------------------------------

def set_onion_batches(batches):
    """Replace the current set of onion-skin GPU batches.

    ``batches`` is a list of tuples:
        (object_name, frame, direction, t, dist, gpu_batch)
    """
    global _onion_batches
    _onion_batches = batches


def get_onion_batches():
    """Return the current set of onion-skin GPU batches."""
    return _onion_batches


def clear_onion_batches():
    """Discard all current onion-skin GPU batches."""
    global _onion_batches
    _onion_batches = []


# ---------------------------------------------------------------------------
# Ghost markers (user-pinned persistent frames)
# ---------------------------------------------------------------------------

def add_ghost_marker(frame, verts, indices, prim_type, obj_names=None):
    """Add a persistent ghost marker.

    *obj_names* is an optional list of object names that contributed to this
    ghost so that selective removal is possible.
    """
    _ghost_markers.append((frame, verts, indices, prim_type, obj_names or []))


def clear_ghost_markers():
    """Clear all ghost markers."""
    _ghost_markers.clear()
    _ghost_batches.clear()


def remove_ghost_markers_for(frame, obj_names):
    """Remove ghost markers that match *frame* AND whose stored object names
    overlap with *obj_names*.  Returns the number of markers removed."""
    obj_set = set(obj_names)
    to_remove = []
    for i, marker in enumerate(_ghost_markers):
        m_frame = marker[0]
        m_obj_names = marker[4] if len(marker) > 4 else []
        if m_frame == frame and (not m_obj_names or obj_set & set(m_obj_names)):
            to_remove.append(i)

    for i in reversed(to_remove):
        _ghost_markers.pop(i)
    # Invalidate all cached batches since indices shifted
    _ghost_batches.clear()
    return len(to_remove)


def get_ghost_markers():
    """Return the list of ghost markers."""
    return _ghost_markers


def get_ghost_batch(index):
    """Get GPU batch for ghost marker at index, building lazily."""
    if index in _ghost_batches:
        return _ghost_batches[index]

    if index < 0 or index >= len(_ghost_markers):
        return None

    marker = _ghost_markers[index]
    # Support both (frame, verts, indices, prim_type) and
    # (frame, verts, indices, prim_type, obj_names) tuples.
    verts = marker[1]
    indices = marker[2]
    prim_type = marker[3]
    batch = batch_for_shader(get_shader(), prim_type, {"pos": verts}, indices=indices)
    _ghost_batches[index] = batch
    return batch


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def cleanup():
    global _shader, _fresnel_shader, _mask_shader
    _shader = None
    _fresnel_shader = None
    _mask_shader = None
    clear_onion_batches()
    clear_mask_batches()
    clear_ghost_markers()
    clear_frame_cache()

