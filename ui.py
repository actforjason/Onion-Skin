# SPDX-License-Identifier: GPL-3.0-or-later
"""UI Panels for onion skin addon."""

import bpy
from bpy.types import Panel, UIList
from bl_operators.presets import AddPresetBase

class ONION_MT_presets(bpy.types.Menu):
    bl_label = "Onion Skin Presets"
    bl_idname = "ONION_MT_presets"
    preset_subdir = "onion_skin"
    preset_operator = "script.execute_preset"
    draw = bpy.types.Menu.draw_preset

class ONION_OT_preset_add(AddPresetBase, bpy.types.Operator):
    bl_idname = "onion_skin.preset_add"
    bl_label = "Add Onion Skin Preset"
    preset_menu = "ONION_MT_presets"

    preset_defines = [
        "scene = bpy.context.scene",
        "idx = scene.onion_skin_active_index",
        "obj = scene.onion_skin_objects[idx].object if len(scene.onion_skin_objects) > 0 else None",
        "settings = obj.onion_skin if obj else None"
    ]
    preset_values = [
        "settings.show_before",
        "settings.show_after",
        "settings.frames_before",
        "settings.frames_after",
        "settings.frame_step",
        "settings.keyframe_mode",
        "settings.keyframe_range_mode",
        "settings.color_before",
        "settings.color_after",
        "settings.opacity_falloff",
        "settings.falloff_curve",
        "settings.use_xray",
        "settings.keep_current_on_top",
        "settings.use_wireframe",
        "settings.use_fresnel",
        "settings.fresnel_power",
        "settings.fresnel_fade",
        "settings.include_children",
        "settings.limit_to_keyframes",
        "settings.use_frame_range",
        "settings.frame_range_start",
        "settings.frame_range_end",
    ]
    preset_subdir = "onion_skin"


class ONION_UL_objects(UIList):
    """UIList for onion skin objects"""
    bl_idname = "ONION_UL_objects"

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            if item.object:
                _type_icons = {
                    'ARMATURE': 'ARMATURE_DATA',
                    'CURVE': 'CURVE_DATA',
                    'SURFACE': 'SURFACE_DATA',
                    'FONT': 'FONT_DATA',
                }
                obj_icon = _type_icons.get(item.object.type, 'MESH_DATA')
                row.label(text=item.object.name, icon=obj_icon)
            else:
                row.label(text="[Missing]", icon='ERROR')
            vis_icon = 'HIDE_OFF' if item.visible else 'HIDE_ON'
            row.prop(item, "visible", text="", icon=vis_icon, emboss=False)
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            if item.object:
                obj_icon = 'ARMATURE_DATA' if item.object.type == 'ARMATURE' else 'MESH_DATA'
                layout.label(text="", icon=obj_icon)
            else:
                layout.label(text="", icon='ERROR')


class ONION_PT_main(Panel):
    """Main panel for onion skin settings"""
    bl_label = "B Onion Skin"
    bl_idname = "ONION_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Onion Skin'

    def draw_header(self, context):
        settings = context.scene.onion_skin_settings
        self.layout.prop(settings, "enabled", text="")

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        settings = scene.onion_skin_settings

        # Object list
        row = layout.row()
        row.template_list(
            "ONION_UL_objects", "",
            scene, "onion_skin_objects",
            scene, "onion_skin_active_index",
            rows=3
        )
        col = row.column(align=True)
        col.operator("onion_skin.add_object", text="", icon='ADD')
        col.operator("onion_skin.remove_selected", text="", icon='REMOVE')
        col.operator("onion_skin.clear_all", text="", icon='TRASH')

        # Ghost frames
        if len(scene.onion_skin_objects) > 0:
            layout.separator()
            row = layout.row(align=True)
            row.operator('onion_skin.add_marker', text="Add Ghost", icon='GHOST_ENABLED')
            if settings.marker_count > 0:
                row.operator('onion_skin.remove_markers', text="", icon='X')
            row.prop(settings, "ghost_color", text="")

            if settings.marker_count > 0:
                row = layout.row(align=True)
                hide_icon = 'HIDE_ON' if not settings.show_ghosts else 'HIDE_OFF'
                row.operator('onion_skin.hide_all_ghosts', text="Hide All", icon=hide_icon, depress=not settings.show_ghosts)
                row.operator('onion_skin.remove_all_ghosts', text="Remove All", icon='TRASH')

            layout.prop(settings, "show_ghost_labels")

        if scene.onion_skin_objects:
            layout.separator()
            row = layout.row(align=True)
            row.menu("ONION_MT_presets", text=ONION_MT_presets.bl_label)
            row.operator("onion_skin.preset_add", text="", icon='ADD')
            row.operator("onion_skin.preset_add", text="", icon='REMOVE').remove_name = True


class ONION_PT_frames(Panel):
    """Frame range settings"""
    bl_label = "Frames"
    bl_idname = "ONION_PT_frames"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Onion Skin'
    bl_parent_id = "ONION_PT_main"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        if not scene.onion_skin_objects:
            layout.label(text="No object selected")
            return

        idx = scene.onion_skin_active_index
        if idx < 0 or idx >= len(scene.onion_skin_objects):
            return

        obj = scene.onion_skin_objects[idx].object
        if not obj:
            return

        settings = obj.onion_skin

        # Keyframe filter mode
        layout.prop(settings, "keyframe_mode", text="Keyframes")

        if settings.keyframe_mode != 'OFF':
            layout.prop(settings, "keyframe_range_mode", text="Mode")

        col = layout.column(align=True)

        before_label = "Before (Keys)" if (settings.keyframe_mode != 'OFF' and settings.keyframe_range_mode == 'COUNT') else "Before"
        after_label = "After (Keys)" if (settings.keyframe_mode != 'OFF' and settings.keyframe_range_mode == 'COUNT') else "After"

        row = col.row(align=True)
        row.prop(settings, "show_before", text="", icon='TRIA_LEFT')
        row.prop(settings, "frames_before", text=before_label)

        row = col.row(align=True)
        row.prop(settings, "show_after", text="", icon='TRIA_RIGHT')
        row.prop(settings, "frames_after", text=after_label)

        step_row = col.row(align=True)
        step_row.prop(settings, "frame_step")
        if settings.keyframe_mode == 'KEYFRAMES_ONLY':
            step_row.enabled = False

        layout.separator()
        layout.prop(settings, "limit_to_keyframes")
        layout.prop(settings, "use_frame_range")
        if settings.use_frame_range:
            row = layout.row(align=True)
            row.prop(settings, "frame_range_start", text="Start")
            row.prop(settings, "frame_range_end", text="End")


class ONION_PT_appearance(Panel):
    """Appearance settings"""
    bl_label = "Appearance"
    bl_idname = "ONION_PT_appearance"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Onion Skin'
    bl_parent_id = "ONION_PT_main"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        if not scene.onion_skin_objects:
            layout.label(text="No object selected")
            return

        idx = scene.onion_skin_active_index
        if idx < 0 or idx >= len(scene.onion_skin_objects):
            return

        obj = scene.onion_skin_objects[idx].object
        if not obj:
            return

        settings = obj.onion_skin

        row = layout.row()
        col = row.column()
        col.label(text="Before")
        col.prop(settings, "color_before", text="")
        col = row.column()
        col.label(text="After")
        col.prop(settings, "color_after", text="")

        layout.separator()

        layout.prop(settings, "opacity_falloff", text="Fade")
        layout.prop(settings, "falloff_curve", text="")

        layout.separator()

        row = layout.row(align=True)
        row.prop(settings, "use_xray", toggle=True)
        row.prop(settings, "use_wireframe", toggle=True)
        row.prop(settings, "use_fresnel", toggle=True)
        layout.prop(settings, "keep_current_on_top")

        if settings.use_fresnel and not settings.use_wireframe:
            layout.prop(settings, "fresnel_fade", text="Fresnel Fade")
            layout.prop(settings, "fresnel_power", text="Fresnel Exponent")


classes = (
    ONION_UL_objects,
    ONION_MT_presets,
    ONION_OT_preset_add,
    ONION_PT_main,
    ONION_PT_frames,
    ONION_PT_appearance,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
