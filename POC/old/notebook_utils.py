"""
Utility functions for Jupyter notebooks in this project.
"""

import sys
import os
import importlib

def setup_path():
    """
    Add the project root directory to the Python path.
    This allows importing modules from the src directory.
    """
    # Get the absolute path of the parent directory (project root)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    
    # Add to path if not already there
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"Added {project_root} to Python path")
    else:
        print(f"{project_root} already in Python path")
        
def reload_module(module):
    """
    Reload a module to pick up changes.
    Useful during development when you're editing source files.
    
    Args:
        module: The module object to reload
    
    Returns:
        The reloaded module
    """
    return importlib.reload(module)