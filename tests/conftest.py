"""Pytest configuration file.

This file automatically sets up the Python path so that the monsoonbench
package can be imported during testing, even if it's not installed.
"""

import sys
from pathlib import Path

# Get the project root directory (assuming conftest.py is in tests/)
# Adjust the number of .parent calls based on your structure
tests_dir = Path(__file__).parent
project_root = tests_dir.parent

# Add the src directory to Python path if it exists
src_dir = project_root / "src"
if src_dir.exists():
    sys.path.insert(0, str(src_dir))
    print(f"Added to Python path: {src_dir}")

# Alternative: if monsoonbench is directly in project root
if (project_root / "monsoonbench").exists():
    sys.path.insert(0, str(project_root))
    print(f"Added to Python path: {project_root}")

# Verify import
try:
    import monsoonbench

    print(f"✓ Successfully imported monsoonbench from {monsoonbench.__file__}")
except ImportError as e:
    print(f"✗ Warning: Could not import monsoonbench: {e}")
    print(f"  Python path: {sys.path}")
