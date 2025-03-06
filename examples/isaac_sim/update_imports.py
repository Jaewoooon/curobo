#!/usr/bin/env python3
#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

"""
Script to update import statements in Isaac Sim example files to support both 
Isaac Sim 4.5.0 and 4.0.0 versions.
"""

import os
import re
import glob
from typing import Dict, List, Optional, Tuple

# Define the mapping of old imports to new imports
IMPORT_MAPPING = {
    "from omni.isaac.kit import SimulationApp": {
        "try_block": """try:
    # Isaac Sim 4.5.0
    from isaacsim import SimulationApp
except ImportError:
    # Isaac Sim 4.0.0
    from omni.isaac.kit import SimulationApp"""
    },
    "from omni.isaac.core import World": {
        "try_block": """try:
    # Isaac Sim 4.5.0 imports
    from isaacsim.core.api.world import World
except ImportError:
    # Isaac Sim 4.0.0 imports
    from omni.isaac.core import World"""
    },
    "from omni.isaac.core.materials import OmniPBR": {
        "try_block": """try:
    # Isaac Sim 4.5.0 imports
    from isaacsim.core.materials import OmniPBR
except ImportError:
    try:
        # Direct import to avoid DeformableMaterialView error
        from omni.isaac.core.materials.pbr_material import OmniPBR
    except ImportError:
        # Fallback to full import (may cause errors with DeformableMaterialView)
        from omni.isaac.core.materials import OmniPBR"""
    },
    "from omni.isaac.core.objects import cuboid": {
        "try_block": """try:
    # Isaac Sim 4.5.0 imports
    from isaacsim.core.prims.objects import cuboid
except ImportError:
    # Isaac Sim 4.0.0 imports
    from omni.isaac.core.objects import cuboid"""
    },
    "from omni.isaac.core.objects import sphere": {
        "try_block": """try:
    # Isaac Sim 4.5.0 imports
    from isaacsim.core.prims.objects import sphere
except ImportError:
    # Isaac Sim 4.0.0 imports
    from omni.isaac.core.objects import sphere"""
    },
    "from omni.isaac.core.robots import Robot": {
        "try_block": """try:
    # Isaac Sim 4.5.0 imports
    from isaacsim.core.robots import Robot
except ImportError:
    # Isaac Sim 4.0.0 imports
    from omni.isaac.core.robots import Robot"""
    },
    "from omni.isaac.core.utils.extensions import enable_extension": {
        "try_block": """try:
    # Isaac Sim 4.5.0
    from isaacsim.core.utils.extensions import enable_extension
except ImportError:
    # Isaac Sim 4.0.0
    from omni.isaac.core.utils.extensions import enable_extension"""
    },
    "from omni.isaac.core.utils.stage import add_reference_to_stage": {
        "try_block": """try:
    # Isaac Sim 4.5.0
    from isaacsim.core.utils.stage import add_reference_to_stage
except ImportError:
    # Isaac Sim 4.0.0
    from omni.isaac.core.utils.stage import add_reference_to_stage"""
    },
    "from omni.isaac.core.utils.nucleus import get_assets_root_path": {
        "try_block": """try:
    # Isaac Sim 4.5.0
    from isaacsim.core.utils.nucleus import get_assets_root_path
except ImportError:
    # Isaac Sim 4.0.0
    from omni.isaac.core.utils.nucleus import get_assets_root_path"""
    },
    "from omni.isaac.core.utils.prims import get_prim_at_path": {
        "try_block": """try:
    # Isaac Sim 4.5.0
    from isaacsim.core.utils.prims import get_prim_at_path
except ImportError:
    # Isaac Sim 4.0.0
    from omni.isaac.core.utils.prims import get_prim_at_path"""
    },
    "from omni.isaac.core.articulations import ArticulationView": {
        "try_block": """try:
    # Isaac Sim 4.5.0
    from isaacsim.core.articulations import ArticulationView
except ImportError:
    # Isaac Sim 4.0.0
    from omni.isaac.core.articulations import ArticulationView"""
    },
    "from omni.isaac.core.prims import RigidPrimView": {
        "try_block": """try:
    # Isaac Sim 4.5.0
    from isaacsim.core.prims import RigidPrimView
except ImportError:
    # Isaac Sim 4.0.0
    from omni.isaac.core.prims import RigidPrimView"""
    },
    "from omni.isaac.core.prims import XFormPrimView": {
        "try_block": """try:
    # Isaac Sim 4.5.0
    from isaacsim.core.prims import XFormPrimView
except ImportError:
    # Isaac Sim 4.0.0
    from omni.isaac.core.prims import XFormPrimView"""
    },
    "from omni.isaac.dynamic_control import _dynamic_control": {
        "try_block": """try:
    # Isaac Sim 4.5.0
    from isaacsim.dynamic_control import _dynamic_control
except ImportError:
    # Isaac Sim 4.0.0
    from omni.isaac.dynamic_control import _dynamic_control"""
    },
    "from omni.importer.urdf import _urdf": {
        "try_block": """try:
    # Isaac Sim 4.5.0
    from isaacsim.importer.urdf import _urdf
except ImportError:
    # Isaac Sim 4.0.0
    from omni.importer.urdf import _urdf"""
    },
}

# Custom OmniPBR implementation to replace problematic imports
CUSTOM_OMNIPBR_CODE = """
from pxr import UsdPhysics, Sdf, UsdShade, Gf

# Custom OmniPBR implementation to avoid DeformableMaterialView import issue
class OmniPBR:
    """A simplified implementation of OmniPBR that doesn't rely on problematic imports"""
    
    def __init__(self, prim_path, color=None):
        """Initialize the OmniPBR material"""
        from omni.usd import get_context
        stage = get_context().get_stage()
        
        # Create the material prim
        self._prim_path = prim_path
        self._prim = stage.DefinePrim(prim_path, "Material")
        
        # Create the shader
        shader_path = f"{prim_path}/Shader"
        self._shader = UsdShade.Shader.Define(stage, shader_path)
        self._shader.CreateIdAttr("OmniPBR")
        
        # Create the material output
        material = UsdShade.Material.Define(stage, prim_path)
        material.CreateSurfaceOutput().ConnectToSource(self._shader.ConnectableAPI(), "surface")
        
        # Set color if provided
        if color is not None:
            self.set_color(color)
    
    def set_color(self, color):
        """Set the diffuse color of the material"""
        if isinstance(color, (list, np.ndarray)):
            if len(color) == 3:
                color = Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
            elif len(color) == 4:
                color = Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
        
        # Set the diffuse color
        self._shader.CreateInput("diffuse_color", Sdf.ValueTypeNames.Color3f).Set(color)
        
        # Set metallic and roughness for a plastic-like material
        self._shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        self._shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
"""

# Add a version flag to detect which Isaac Sim version is being used
VERSION_FLAG = """
# Flag to detect Isaac Sim version
ISAAC_SIM_45 = False
try:
    import isaacsim
    ISAAC_SIM_45 = True
except ImportError:
    ISAAC_SIM_45 = False
"""

def add_isaac_sim_version_flag(content: str) -> str:
    """Add a flag to detect Isaac Sim version."""
    if "ISAAC_SIM_45" not in content:
        # Find the first import statement
        import_match = re.search(r"^import|^from", content, re.MULTILINE)
        if import_match:
            # Insert the version flag before the first import
            pos = import_match.start()
            content = content[:pos] + VERSION_FLAG + content[pos:]
        else:
            # If no import is found, add it at the beginning after any comments or docstrings
            content = VERSION_FLAG + content
    
    return content

def update_import_statement(content: str, old_import: str, new_import: str) -> str:
    """Update a specific import statement to use try-except pattern."""
    # Check if the old import exists in the content
    if f"from {old_import}" in content or f"import {old_import}" in content:
        # Handle special case for OmniPBR
        if "OmniPBR" in old_import or "OmniPBR" in new_import:
            # Instead of try-except, we'll use our custom OmniPBR implementation
            # Remove any existing OmniPBR imports
            content = re.sub(
                r"from (?:omni\.isaac\.core\.materials|isaacsim\.core\.materials)(?:\.pbr_material)? import OmniPBR.*?\n",
                "",
                content,
                flags=re.MULTILINE,
            )
            
            # Check if our custom OmniPBR is already in the content
            if "class OmniPBR:" not in content:
                # Find a good place to insert our custom implementation
                # After imports but before the first class or function
                match = re.search(r"^(?:import|from).*?\n\n", content, re.MULTILINE | re.DOTALL)
                if match:
                    pos = match.end()
                    content = content[:pos] + CUSTOM_OMNIPBR_CODE + content[pos:]
                else:
                    # If no suitable position is found, add it after the first import
                    match = re.search(r"^(?:import|from).*?\n", content, re.MULTILINE)
                    if match:
                        pos = match.end()
                        content = content[:pos] + "\n" + CUSTOM_OMNIPBR_CODE + content[pos:]
            
            return content
        
        # For other imports, create a try-except pattern
        from_import_pattern = rf"from {old_import} import (.*)"
        import_pattern = rf"import {old_import}(.*)"
        
        # Handle 'from X import Y' pattern
        from_match = re.search(from_import_pattern, content)
        if from_match:
            imports = from_match.group(1).strip()
            replacement = f"""try:
    # Isaac Sim 4.5.0 imports
    from {new_import} import {imports}
    ISAAC_SIM_45 = True
except ImportError:
    # Isaac Sim 4.0.0 imports
    from {old_import} import {imports}
    ISAAC_SIM_45 = False
"""
            content = re.sub(from_import_pattern, replacement, content)
        
        # Handle 'import X' pattern
        import_match = re.search(import_pattern, content)
        if import_match:
            suffix = import_match.group(1).strip()
            replacement = f"""try:
    # Isaac Sim 4.5.0 imports
    import {new_import}{suffix}
    ISAAC_SIM_45 = True
except ImportError:
    # Isaac Sim 4.0.0 imports
    import {old_import}{suffix}
    ISAAC_SIM_45 = False
"""
            content = re.sub(import_pattern, replacement, content)
    
    return content

def update_file(file_path: str) -> None:
    """Update a single file with new import statements."""
    print(f"Processing {file_path}...")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if file has already been updated
    if "# Isaac Sim 4.5.0" in content:
        print(f"  File already updated, skipping: {file_path}")
        return
    
    # Add version flag if not already present
    if "ISAAC_SIM_45 = " not in content:
        content = add_isaac_sim_version_flag(content)
    
    # Update import statements
    for old_import, new_import_data in IMPORT_MAPPING.items():
        content = content.replace(old_import, new_import_data["try_block"])
    
    # Write updated content back to file
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"  Updated: {file_path}")

def main():
    """Main function to update all Python files in the isaac_sim directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    python_files = glob.glob(os.path.join(script_dir, "*.py"))
    
    # Skip the update script itself
    python_files = [f for f in python_files if os.path.basename(f) != "update_imports.py"]
    
    print(f"Found {len(python_files)} Python files to process")
    
    for file_path in python_files:
        update_file(file_path)
    
    print("Update complete!")

if __name__ == "__main__":
    main()
