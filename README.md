# 3D_Print_Python
A repository for scripts and programs that generate STL and 3MF files for import into 3D printer slicer programs and other uses

## Setup

The scripts need three Python packages:

```
pip3 install -r requirements.txt
```

(With Homebrew's Python, add `--user --break-system-packages`, or use a virtual environment.)

Run any script with `--help` to see all of its options.  Every script writes STL by
default; add `--format 3mf` (or give an `--output` name ending in `.3mf`) to get a 3MF
file instead.  3MF keeps parts as separate, named objects, and Bambu Studio and
OrcaSlicer pick up which filament each part prints with.

## Projects

- **Drawer_Organizer** - organizer trays with diagonal or grid compartments and an
  optional stacking lip.
  - `python3 Drawer_Organizer/drawer_Organizer.py --compartments 4`
  - `python3 Drawer_Organizer/drawer_Organizer.py --grid 3x2 --size 240x160`
  - Fill a drawer: `--drawer 520x380` works out how many organizers it takes and how
    big each one is to fit the bed (`--bed`, default 256 mm).  Add a height,
    `--drawer 520x380x70 --levels 2`, to size them to stack two high.
  - Sizes can be in inches: `--drawer 20-1/2x15x3in` (a unit on the end applies to every
    number; `cm` works too).
  - Test print: `--scale 0.25`.
- **Chess_Set_01** - a chess set.  `--sides 2 --format 3mf` gives both sides, with white
  on filament 1 and black on filament 2.
- **4_Sided_Pyramid** - a stepped pyramid.

## printlib

Shared code used by the scripts: shape building blocks and booleans (`printlib/shapes.py`)
and STL/3MF saving (`printlib/export.py`).  Scripts import it straight from the
repository, so there's nothing to install.
