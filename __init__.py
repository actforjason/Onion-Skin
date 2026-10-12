# SPDX-License-Identifier: GPL-3.0-or-later
"""B Onion Skin — GPU onion skinning for 3D animation."""

bl_info = {
    "name": "B Onion Skin",
    "author": "Dinesh007",
    "version": (1, 1, 0),
    "blender": (5, 2, 0),
    "location": "View3D > Sidebar > Onion Skin",
    "description": "GPU onion skinning for 3D animation",
    "category": "Animation",
}

from . import cache
from . import properties
from . import operators
from . import ui
from . import drawing


def register():
    properties.register()
    operators.register()
    ui.register()
    drawing.register()


def unregister():
    drawing.unregister()
    ui.unregister()
    operators.unregister()
    properties.unregister()
    cache.cleanup()


if __name__ == "__main__":
    register()
