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

Bambu Studio 2.8.2 shows "The 3mf file has invalid config, load geometry data only" when
it opens these files.  That is a bug in that version (it says so for any 3MF without
Bambu printer settings inside); the fix is in the 2.8.3 and 2.8.4 public betas.  The
model and its filament assignments still load correctly, so click OK.

## Projects

- **Drawer_Organizer** - organizer trays with diagonal or grid compartments and an
  optional stacking lip.
  - `python3 Drawer_Organizer/drawer_Organizer.py --compartments 4`
  - `python3 Drawer_Organizer/drawer_Organizer.py --grid 3x2 --size 240x160`
  - Fill a drawer: `--drawer 520x380x70 --format 3mf` makes a set that fills it: one
    main organizer as big as the bed allows (`--bed`, default 256 mm) with the
    compartments you asked for, plus single-compartment bins in the space beside and
    behind it.  A narrow strip gets one bin as long as the bed can print: laid corner
    to corner, a narrow bin can be longer than the bed is wide, and it is already
    turned that way in the file.  Where that long bin leaves a small corner free, the
    bin next to it is made L-shaped to fill it.  The whole set goes in one file; use
    Arrange All in the slicer to
    spread the pieces over plates (pieces that nearly fill the bed may need dragging
    onto a plate by hand).  With STL you get one file per distinct piece.
  - The drawer height is a ceiling: walls go as tall as it allows, less
    `--drawer_headroom` (2 mm), and never taller.  Add `--height` for lower walls,
    `--levels 2` to stack two sets, or `--bin_size 120` for more, smaller bins.
  - Sizes can be in inches: `--drawer 20-1/2x15x3in` (a unit on the end applies to every
    number; `cm` works too).
  - Test print: `--scale 0.25`.
- **Chess_Set_01** - a chess set.  `--sides 2 --format 3mf` gives both sides, with white
  on filament 1 and black on filament 2.
- **4_Sided_Pyramid** - a stepped pyramid.
- **Queen_Anne** - a simplified, multi-color waterline model of Cunard's cruise ship
  MS Queen Anne, laid out from Cunard's published deck plans and photographs.
  - `python3 Queen_Anne/queen_anne.py` writes `queen_anne.3mf`: one object made of parts
    already assigned to six filaments (1 white, 2 black, 3 red, 4 orange for the
    lifeboats, 5 light brown for the decks, 6 blue for the pools).
  - It is as long as fits the bed straight (250 mm on a 256 mm bed, about 1:1290).
    `--diagonal` makes it as long as fits corner to corner (about 318 mm), and
    `--length` sets the length yourself.
  - `--no-lifeboats`, `--no-decks` and `--no-pools` each save a filament.
    `--no-lettering` leaves CUNARD off the hull; on a small model the letters are
    printed larger than true scale so their strokes are printable.

## printlib

Shared code used by the scripts: shape building blocks and booleans (`printlib/shapes.py`),
STL/3MF saving (`printlib/export.py`), sizes with units (`printlib/units.py`) and
fitting pieces on the bed (`printlib/bed.py`).  Scripts import it straight from the
repository, so there's nothing to install.
