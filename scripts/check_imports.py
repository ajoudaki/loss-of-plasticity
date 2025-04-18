#!/usr/bin/env python3
"""
Script to check for missing imports and other potential issues in the project.
This is intended to help maintain code quality and identify potential problems.
"""

import os
import sys
import importlib
import ast
import argparse
from pathlib import Path

def get_imports_from_file(file_path):
    """
    Extract all import statements from a Python file.
    
    Returns:
        list: List of imported module names
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append(name.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module is not None:
                    imports.append(node.module)
        
        return imports
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []

def check_module_availability(module_name):
    """
    Check if a module can be imported.
    
    Returns:
        tuple: (bool, str) - (is_available, error_message)
    """
    try:
        # Handle relative imports
        if module_name.startswith('.'):
            return True, ""
        
        # Split module path to check just the top-level package
        top_level = module_name.split('.')[0]
        importlib.import_module(top_level)
        return True, ""
    except ImportError as e:
        return False, str(e)

def scan_directory_for_python_files(directory):
    """
    Recursively scan directory for Python files.
    
    Returns:
        list: List of Python file paths
    """
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def check_source_directory(directory, skip_system_modules=True):
    """
    Check all Python files in a directory for import issues.
    
    Args:
        directory: Directory to scan
        skip_system_modules: Whether to skip checking system modules
        
    Returns:
        dict: Dictionary of issues by file
    """
    python_files = scan_directory_for_python_files(directory)
    issues = {}
    
    for file_path in python_files:
        file_issues = []
        imports = get_imports_from_file(file_path)
        
        for module_name in imports:
            # Skip relative imports and system modules if requested
            if skip_system_modules and not module_name.startswith('.') and module_name in sys.builtin_module_names:
                continue
                
            is_available, error = check_module_availability(module_name)
            if not is_available:
                file_issues.append(f"Missing import: {module_name} - {error}")
        
        if file_issues:
            issues[file_path] = file_issues
    
    return issues

def main():
    parser = argparse.ArgumentParser(description="Check Python files for import issues")
    parser.add_argument("directory", 
                      help="Directory to scan for Python files", 
                      default="src",
                      nargs='?')
    parser.add_argument("--check-system-modules", 
                      action="store_true",
                      help="Also check system modules")
    
    args = parser.parse_args()
    directory = args.directory
    
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return 1
    
    print(f"Scanning {directory} for Python import issues...")
    issues = check_source_directory(
        directory,
        skip_system_modules=not args.check_system_modules
    )
    
    if not issues:
        print("No import issues found.")
        return 0
    
    print("\nImport issues found:")
    for file_path, file_issues in issues.items():
        rel_path = os.path.relpath(file_path, directory)
        print(f"\n{rel_path}:")
        for issue in file_issues:
            print(f"  - {issue}")
    
    return 1

if __name__ == "__main__":
    sys.exit(main())