"""Shared code for the model generators in this repository.

Scripts in the project folders add the repository root to sys.path so they
can `import printlib` without installing anything.
"""

from .export import Part, add_format_argument, output_path, save
