# SPDX-License-Identifier: GPL-3.0-or-later
"""Property definitions for onion skin addon."""

import bpy
from bpy.types import PropertyGroup
from bpy.props import (
    BoolProperty, IntProperty, FloatProperty,
    FloatVectorProperty, PointerProperty, EnumProperty
)


def _safe_redraw(context):
    """Safely trigger viewport redraw across all windows."""
    try:
        if hasattr(context, "window_manager") and context.window_manager:
            for window in context.window_manager.windows:
                screen = window.screen
                if screen:
                    for area in screen.areas:
                        if area.type == 'VIEW_3D':
                            area.tag_redraw()
        elif hasattr(context, "screen") and context.screen:
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
    except (AttributeError, ReferenceError):
        pass


def _update_redraw(self, context):
    """Trigger a viewport redraw and rebuild onion skin batches."""
    from . import drawing
    try:
        scene = context.scene
        if hasattr(scene, 'onion_skin_settings') and scene.onion_skin_settings.enabled:
            depsgraph = context.evaluated_depsgraph_get()
            drawing._build_onion_batches(context, scene.frame_current, depsgraph)
    except Exception:
        pass
    _safe_redraw(context)


def _just_redraw(self, context):
    """Trigger a viewport redraw without rebuilding batches (for visual properties)."""
    _safe_redraw(context)


def _update_geometry_settings(self, context):
    """Clear frame cache and rebuild."""
    from . import cache
    cache.clear_frame_cache()
    _update_redraw(self, context)


class ONION_ObjectItem(PropertyGroup):
    """Object reference in the onion skin list."""

    object: PointerProperty(
        name="Object",
        type=bpy.types.Object,
        description="Object to display onion skin for",
        update=_update_geometry_settings
    )

    visible: BoolProperty(
        name="Visible",
        description="Show onion skin for this object",
        default=True,
        update=_update_geometry_settings
    )


class ONION_SceneSettings(PropertyGroup):
    """Global settings for the onion skin system."""

    enabled: BoolProperty(
        name="Enable",
        description="Toggle onion skin display",
        default=True,
        update=_update_redraw
    )

    pick_object: PointerProperty(
        name="Pick Object",
        type=bpy.types.Object,
        description="Click eyedropper to pick an object"
    )

    # Ghost frames
    marker_count: IntProperty(
        name="Marker Count",
        description="Number of active ghost frames",
        default=0,
        min=0,
    )

    ghost_color: FloatVectorProperty(
        name="Ghost Color",
        description="Color for pinned ghost overlays",
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(0.2, 0.8, 0.3, 0.5),
    )

    show_ghosts: BoolProperty(
        name="Show Ghosts",
        description="Toggle visibility of all ghost overlays without removing them",
        default=True,
        update=_just_redraw
    )

    show_ghost_labels: BoolProperty(
        name="Frame Labels",
        description="Show frame numbers on ghost overlays",
        default=True,
        update=_update_redraw
    )


class ONION_ObjectSettings(PropertyGroup):
    """Per-object settings for the onion skin system."""

    show_before: BoolProperty(
        name="Show Before",
        description="Show frames before current",
        default=True,
        update=_update_redraw
    )

    show_after: BoolProperty(
        name="Show After",
        description="Show frames after current",
        default=True,
        update=_update_redraw
    )

    frames_before: IntProperty(
        name="Frames Before",
        description="Number of ghost frames before current",
        default=3,
        min=1,
        soft_max=50,
        update=_update_redraw
    )

    frames_after: IntProperty(
        name="Frames After",
        description="Number of ghost frames after current",
        default=3,
        min=1,
        soft_max=50,
        update=_update_redraw
    )

    frame_step: IntProperty(
        name="Frame Step",
        description="Interval between ghost frames",
        default=2,
        min=1,
        soft_max=10,
        update=_update_redraw
    )

    color_before: FloatVectorProperty(
        name="Before Color",
        description="Color for frames before current",
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(0.2, 0.5, 1.0, 0.5),
        update=_just_redraw
    )

    color_after: FloatVectorProperty(
        name="After Color",
        description="Color for frames after current",
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(1.0, 0.3, 0.2, 0.5),
        update=_just_redraw
    )

    opacity_falloff: FloatProperty(
        name="Opacity Falloff",
        description="How quickly ghosts fade with distance",
        default=0.6,
        min=0.0,
        soft_max=1.0,
        subtype='FACTOR',
        update=_just_redraw
    )

    falloff_curve: EnumProperty(
        name="Falloff Curve",
        description="Shape of the opacity falloff",
        items=[
            ('linear', "Linear", "Straight line falloff"),
            ('smooth', "Smooth", "Natural S-curve falloff"),
            ('exponential', "Exponential", "Fast fade at distance"),
        ],
        default='smooth',
        update=_just_redraw
    )

    use_xray: BoolProperty(
        name="X-Ray",
        description="Display ghosts through other geometry",
        default=False,
        update=_just_redraw
    )

    keep_current_on_top: BoolProperty(
        name="Current On Top",
        description="Always display the current frame object in front of onion skins",
        default=True,
        update=_just_redraw
    )

    use_wireframe: BoolProperty(
        name="Wireframe",
        description="Draw ghosts as wireframe",
        default=False,
        update=_update_geometry_settings
    )

    use_fresnel: BoolProperty(
        name="Fresnel",
        description="Use a Fresnel-based shader for a glowing edge effect",
        default=False,
        update=_update_geometry_settings
    )

    fresnel_power: FloatProperty(
        name="Fresnel Exponent",
        description="Controls how quickly the Fresnel effect fades towards the center",
        default=2.0,
        min=0.1,
        max=10.0,
        update=_just_redraw
    )

    fresnel_fade: FloatProperty(
        name="Fresnel Fade",
        description="Controls the strength of the Fresnel fade-out at the center (0.0 = solid color, 1.0 = pure outline)",
        default=1.0,
        min=0.0,
        max=1.0,
        update=_just_redraw
    )

    include_children: BoolProperty(
        name="Armature Children",
        description="Include mesh children of armatures",
        default=True,
        update=_update_geometry_settings
    )

    limit_to_keyframes: BoolProperty(
        name="Limit to Keyframes",
        description="Hide onion skins before the first and after the last keyframe",
        default=True,
        update=_update_redraw
    )

    use_frame_range: BoolProperty(
        name="Limit to Range",
        description="Only show onion skin within specified frame range",
        default=False,
        update=_update_redraw
    )

    frame_range_start: IntProperty(
        name="Start",
        description="Start frame for onion skin display",
        default=1,
        min=0,
        update=_update_redraw
    )

    frame_range_end: IntProperty(
        name="End",
        description="End frame for onion skin display",
        default=250,
        min=0,
        update=_update_redraw
    )


classes = (
    ONION_ObjectItem,
    ONION_SceneSettings,
    ONION_ObjectSettings,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.onion_skin_settings = PointerProperty(type=ONION_SceneSettings)
    bpy.types.Object.onion_skin = PointerProperty(type=ONION_ObjectSettings)
    bpy.types.Scene.onion_skin_objects = bpy.props.CollectionProperty(type=ONION_ObjectItem)
    bpy.types.Scene.onion_skin_active_index = bpy.props.IntProperty(name="Active Object Index")


def unregister():
    del bpy.types.Scene.onion_skin_active_index
    del bpy.types.Scene.onion_skin_objects
    del bpy.types.Object.onion_skin
    del bpy.types.Scene.onion_skin_settings

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
