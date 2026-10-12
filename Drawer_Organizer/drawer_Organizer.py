# Generate a drawer organizer.  Can have one or more compartments.
#
# Compartments are diagonal by default to maximize length in the center
# compartment, since many 3D printer beds are less than 11 inches (~ 256 mm)
# in size which limits how large a single piece organizer can be.  A straight
# grid of compartments is also available.
#
# Give it your drawer's inside measurements with --drawer and it makes a set
# that fills the drawer: one main organizer as big as the bed allows, plus
# single-compartment bins in the space beside and behind it.
#
# Optionally the organizer gets a stacking lip on the bottom: the lowest part of
# the outer wall is stepped inward so it drops inside the walls of an identical
# organizer underneath.  The step is a 45 degree chamfer rather than a flat ledge
# so it prints without supports, and the floor stays on the build plate.
#
# Requires: numpy, trimesh, manifold3d (pip install -r requirements.txt, from the repository root)

import argparse
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from printlib import Part, add_format_argument, output_path, save  # noqa: E402
from printlib import shapes  # noqa: E402
from printlib.units import dimensions_type, format_mm  # noqa: E402

DEFAULT_SIZE = 250.0
DEFAULT_HEIGHT = 35.0
BED_MARGIN = 3.0          # Keep this far from the edges of the bed
AUTO_ARRANGE_MARGIN = 24.0  # Bambu Studio's arrange wants about this much spare bed
MIN_TAB = 15.0            # Smallest leftover corner worth growing a bin into
MIN_BIN = 30.0            # Leftover space narrower than this is left empty
SET_SPREAD = 1.02         # Spreads a drawer set apart slightly so objects don't touch


@dataclass(frozen=True)
class Piece:
    """One printed piece: where it sits in the drawer and its outline.  x and y
    are the front left corner of its bounding box.  A notch is a rectangle
    (x, y, width, depth) cut out of one corner, which makes the piece L-shaped."""
    label: str
    x: float
    y: float
    width: float
    depth: float
    is_main: bool = False
    notch: tuple = None

    def scaled(self, factor):
        notch = tuple(v * factor for v in self.notch) if self.notch else None
        return replace(self, x=self.x * factor, y=self.y * factor, width=self.width * factor,
                       depth=self.depth * factor, notch=notch)

    def local_notch(self):
        """The notch measured from the center of the piece, as create_organizer wants it."""
        if not self.notch:
            return None
        nx, ny, nw, nd = self.notch
        return (nx - self.x - self.width / 2, ny - self.y - self.depth / 2, nw, nd)

    @property
    def key(self):
        """Pieces with the same key are identical prints."""
        notch = tuple(round(v, 3) for v in self.local_notch()) if self.notch else None
        return (round(self.width, 3), round(self.depth, 3), self.is_main, notch)

    def describe(self):
        text = f"{format_mm(self.width)} x {format_mm(self.depth)}mm"
        if self.notch:
            text += f" with a {format_mm(self.notch[2])} x {format_mm(self.notch[3])}mm corner cut out"
        return text


def parse_grid(text):
    """Parse 'COLUMNSxROWS', e.g. '3x2'."""
    try:
        columns, rows = (int(v) for v in text.lower().replace(' ', '').split('x'))
    except ValueError:
        raise argparse.ArgumentTypeError(f"--grid must be whole numbers like 3x2, not '{text}'") from None
    if columns < 1 or rows < 1:
        raise argparse.ArgumentTypeError("--grid needs at least 1 column and 1 row")
    return columns, rows


def stepped_block(bottom_width, bottom_depth, inset, z_bottom, z_step, z_top):
    """Rectangular solid centered on the Z axis.

    It is bottom_width x bottom_depth from z_bottom up to z_step, flares out by
    inset on every side at 45 degrees, then goes straight up to z_top.  With
    inset == 0 it is just a box.
    """
    top_width, top_depth = bottom_width + 2 * inset, bottom_depth + 2 * inset
    if inset <= 0:
        rings = [(top_width, top_depth, z_bottom), (top_width, top_depth, z_top)]
    else:
        rings = [(bottom_width, bottom_depth, z_bottom), (bottom_width, bottom_depth, z_step),
                 (top_width, top_depth, z_step + inset), (top_width, top_depth, z_top)]
    return shapes.loft_rects(rings)


def outline_rects(width, depth, notch):
    """The outline as rectangles (center_x, center_y, width, depth).  A plain
    piece is one rectangle.  A piece with a corner notch is an L: the strip
    beside the notch plus the strip beyond it, overlapping in the corner."""
    if not notch:
        return [(0.0, 0.0, width, depth)]
    nx, ny, nw, nd = notch
    if ny <= -depth / 2 + 1e-6:             # notch at the front: keep what is behind it
        y0, y1 = ny + nd, depth / 2
    else:
        y0, y1 = -depth / 2, ny
    if nx <= -width / 2 + 1e-6:             # notch on the left: keep what is right of it
        x0, x1 = nx + nw, width / 2
    else:
        x0, x1 = -width / 2, nx
    return [(0.0, (y0 + y1) / 2, width, y1 - y0), ((x0 + x1) / 2, 0.0, x1 - x0, depth)]


def stepped_outline(rects, shrink, inset, z_bottom, z_step, z_top):
    """stepped_block for an outline made of rectangles, each pulled in by shrink
    on every side at the bottom."""
    blocks = []
    for center_x, center_y, width, depth in rects:
        block = stepped_block(width - 2 * shrink, depth - 2 * shrink, inset, z_bottom, z_step, z_top)
        block.apply_translation([center_x, center_y, 0])
        blocks.append(block)
    return blocks[0] if len(blocks) == 1 else shapes.union(blocks)


def create_divider(start, direction, length, extend_start, extend_end, thickness,
                   z_top, floor_thickness, bite_width, bite_depth):
    """Divider wall running from start along direction for length mm.

    The wall is extended past its ends (into the outer walls or a crossing
    divider); anything sticking outside the organizer is trimmed later.
    """
    # Start halfway into the floor: a divider bottom flush with the underside of
    # the floor leaves coplanar faces that the boolean union handles badly.
    z_bottom = floor_thickness / 2
    total_length = length + extend_start + extend_end
    wall = shapes.block([total_length, thickness, z_top - z_bottom],
                        [(length + extend_end - extend_start) / 2, 0, (z_bottom + z_top) / 2])

    if bite_depth > 0 and bite_width > 0:
        # Elliptical scoop out of the top edge so it's easy to reach into the
        # compartments.  An elliptic cylinder gives a smooth curve, unlike a
        # low-poly ellipsoid.
        bite_depth = min(bite_depth, z_top - floor_thickness - 1.0)
        bite = trimesh.creation.cylinder(radius=1.0, height=thickness * 4, sections=128)
        bite.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        bite.apply_scale([bite_width / 2, 1, bite_depth])
        bite.apply_translation([length / 2, 0, z_top])
        wall = shapes.difference(wall, [bite])

    angle = math.atan2(direction[1], direction[0])
    wall.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 0, 1]))
    wall.apply_translation([start[0], start[1], 0])
    return wall


def clip_line_to_rect(point, direction, half_width, half_depth):
    """Return the (start, end) of the line through point along direction inside
    the rectangle [-half_width, half_width] x [-half_depth, half_depth], or None
    if it misses.  direction must have non-zero components."""
    t_lo = max((-half_width - point[0]) / direction[0], (-half_depth - point[1]) / direction[1])
    t_hi = min((half_width - point[0]) / direction[0], (half_depth - point[1]) / direction[1])
    if t_hi - t_lo <= 1e-6:
        return None
    return point + t_lo * direction, point + t_hi * direction


def diagonal_divider_segments(compartments, half_width, half_depth):
    """Dividers parallel to the y = x diagonal, equally spaced across the
    inside of the organizer.  Yields (start, direction, length) with both ends
    on the outer walls."""
    direction = np.array([1.0, 1.0]) / math.sqrt(2)
    normal = np.array([-1.0, 1.0]) / math.sqrt(2)
    total_width = (2 * half_width + 2 * half_depth) / math.sqrt(2)
    compartment_width = total_width / compartments

    for i in range(1, compartments):
        d = (i - compartments / 2) * compartment_width
        segment = clip_line_to_rect(normal * d, direction, half_width, half_depth)
        if segment is None:
            print(f"Warning: Divider {i} doesn't cross the organizer. Skipping.")
            continue
        start, end = segment
        length = float(np.linalg.norm(end - start))
        if length < 1.0:
            print(f"Warning: Divider {i} is only {length:.2f}mm long. Skipping.")
            continue
        yield start, direction, length, True, True


def grid_divider_segments(columns, rows, half_width, half_depth):
    """Straight dividers for a columns x rows grid.  Each divider is split into
    one segment per compartment so every compartment gets its own scoop.
    Yields (start, direction, length, starts_at_wall, ends_at_wall)."""
    xs = np.linspace(-half_width, half_width, columns + 1)
    ys = np.linspace(-half_depth, half_depth, rows + 1)
    for x in xs[1:-1]:                          # Dividers running front to back
        for j in range(rows):
            yield (np.array([x, ys[j]]), np.array([0.0, 1.0]), ys[j + 1] - ys[j],
                   j == 0, j == rows - 1)
    for y in ys[1:-1]:                          # Dividers running left to right
        for i in range(columns):
            yield (np.array([xs[i], y]), np.array([1.0, 0.0]), xs[i + 1] - xs[i],
                   i == 0, i == columns - 1)


def create_organizer(width, depth, height, wall_thickness, divider_thickness, floor_thickness,
                     compartments, grid, stack_lip, lip_height, lip_clearance,
                     bite_width_ratio, bite_depth_ratio, notch=None):
    total_height = floor_thickness + height
    t = wall_thickness
    rects = outline_rects(width, depth, notch)

    # Outer body: the solid envelope of the organizer.  width x depth is the true
    # outside footprint, so walls are inside it rather than centered on its edge.
    if stack_lip:
        inset = t + lip_clearance                  # lip slips inside the walls below
        chamfer_shift = t * (math.sqrt(2) - 1)     # keeps the chamfered wall t thick
        outer = stepped_outline(rects, inset, inset, 0, lip_height, total_height)
        cavity = stepped_outline(rects, inset + t, inset,
                                 floor_thickness, lip_height + chamfer_shift, total_height + 1)
        # The organizer above rests on its chamfer, lip_height + lip_clearance up
        # from its bottom, so it hangs that far down into this one.  Dividers have
        # to stop short of that or the organizer above sits on them instead.
        divider_top = total_height - lip_height - 2 * lip_clearance
    else:
        outer = stepped_outline(rects, 0, 0, 0, 0, total_height)
        cavity = stepped_outline(rects, t, 0, floor_thickness, 0, total_height + 1)
        divider_top = total_height

    components = [shapes.difference(outer, [cavity])]

    half_width, half_depth = width / 2 - t, depth / 2 - t
    if grid:
        segments = grid_divider_segments(grid[0], grid[1], half_width, half_depth)
    elif compartments > 1:
        segments = diagonal_divider_segments(compartments, half_width, half_depth)
    else:
        segments = []

    into_wall = 2 * t + divider_thickness       # Plenty to reach through the wall
    into_divider = divider_thickness / 2 + 0.1  # Just past the middle of a crossing
    for start, direction, length, starts_at_wall, ends_at_wall in segments:
        components.append(create_divider(
            start=start,
            direction=direction,
            length=length,
            extend_start=into_wall if starts_at_wall else into_divider,
            extend_end=into_wall if ends_at_wall else into_divider,
            thickness=divider_thickness,
            z_top=divider_top,
            floor_thickness=floor_thickness,
            bite_width=length * bite_width_ratio,
            bite_depth=(divider_top - floor_thickness) * bite_depth_ratio,
        ))

    organizer = shapes.union(components)
    # Trim divider ends that poke out past the outer walls.
    return shapes.intersection([organizer, outer])


def split_rect(x, y, width, depth, max_width, max_depth):
    """Split a rectangle into the fewest equal pieces no bigger than
    max_width x max_depth.  Returns a list of (x, y, width, depth)."""
    if width <= 0 or depth <= 0:
        return []
    columns = math.ceil(width / max_width - 1e-9)
    rows = math.ceil(depth / max_depth - 1e-9)
    w, d = width / columns, depth / rows
    return [(x + i * w, y + j * d, w, d) for j in range(rows) for i in range(columns)]


def longest_fit(width, bed_width, bed_depth):
    """Longest rectangle of the given width that fits on the bed when it may be
    turned to any angle.  A narrow piece laid corner to corner can be a good
    deal longer than the bed is wide.  Rounded down to 0.1 mm."""
    best = 0.0
    for tenth in range(0, 901):
        angle = math.radians(tenth / 10)
        c, s = math.cos(angle), math.sin(angle)
        if c < 1e-9:
            length = bed_depth if width <= bed_width else 0.0
        elif s < 1e-9:
            length = bed_width if width <= bed_depth else 0.0
        else:
            length = min((bed_width - width * s) / c, (bed_depth - width * c) / s)
        best = max(best, length)
    return math.floor(best * 10 + 1e-6) / 10


def print_rotation(width, depth, bed_width, bed_depth):
    """Degrees to turn a width x depth piece so it fits the bed: 0 if it fits as
    it is, otherwise the angle that fits best (45 for a long piece on a square bed)."""
    if width <= bed_width + 1e-6 and depth <= bed_depth + 1e-6:
        return 0.0

    def overflow(degrees):
        c, s = abs(math.cos(math.radians(degrees))), abs(math.sin(math.radians(degrees)))
        return max((width * c + depth * s) / bed_width, (width * s + depth * c) / bed_depth)
    return min((half / 2 for half in range(0, 360)), key=overflow)


def plan_drawer(usable_width, usable_depth, max_width, max_depth, bin_max):
    """Lay out a drawer: one main organizer as big as the bed allows in the
    front left corner, and single-compartment bins filling the L-shaped space
    beside and behind it.

    A narrow strip beside (or behind) the main organizer gets one bin as long as
    the bed can print, which for a narrow bin means laid diagonally on the bed.
    The bins next to it are shortened to make room.

    Where the long bin reaches past the main organizer it leaves a small corner
    free behind the main organizer; the bin next to it grows into that corner,
    which makes it L-shaped.

    Returns (main, bins, gaps).  main is (x, y, width, depth) measured from the
    front left corner.  Each bin is (x, y, width, depth, notch), where notch is
    None or the (x, y, width, depth) rectangle cut out of one corner.  gaps are
    the (width, depth) of spaces left empty because they are too small to use.
    """
    def plain(rects):
        return [(*rect, None) for rect in rects]

    def plan(total_width, total_depth, bed_width, bed_depth):
        # One orientation: the long bin, if any, goes in the strip to the right.
        main_width, main_depth = min(total_width, bed_width), min(total_depth, bed_depth)
        right, back = total_width - main_width, total_depth - main_depth
        gaps = []
        if 0 < right < MIN_BIN:
            gaps.append((right, total_depth))
            right = 0.0
        full_depth = total_depth          # The strip beside the main organizer can use all of it
        if 0 < back < MIN_BIN:
            gaps.append((main_width, back))
            back = 0.0
        cap_width, cap_depth = min(bed_width, bin_max), min(bed_depth, bin_max)
        layouts = []

        longest = min(longest_fit(right, bed_width, bed_depth), bin_max, full_depth) if right else 0.0
        if longest > main_depth + 1e-6:
            long_bin = (main_width, 0.0, right, longest, None)
            behind_long = full_depth - longest

            # A: bins run the full width behind the main organizer; the ones
            # behind the long bin start where it ends.
            bins, a_gaps = [long_bin], []
            for x, y, w, d in split_rect(0, main_depth, main_width + right, back, cap_width, cap_depth):
                notch = None
                if x + w > main_width + 1e-6 and y < longest:
                    corner_width, corner_depth = main_width - x, longest - y
                    if min(corner_width, corner_depth) >= MIN_TAB and d - corner_depth >= MIN_BIN:
                        # Keep the corner behind the main organizer: an L-shaped bin.
                        notch = (main_width, y, x + w - main_width, corner_depth)
                    else:
                        if corner_width > 1e-6:
                            a_gaps.append((corner_width, corner_depth))
                        y, d = longest, y + d - longest
                if d >= MIN_BIN:
                    bins.append((x, y, w, d, notch))
                elif d > 1e-6:
                    a_gaps.append((w, d))
            if not back and behind_long > 1e-6:
                a_gaps.append((right, behind_long))
            layouts.append((bins, a_gaps))

            # B: the strip beside the main organizer runs the full depth.
            bins, b_gaps = [long_bin], []
            if behind_long >= MIN_BIN:
                bins += plain(split_rect(main_width, longest, right, behind_long, cap_width, cap_depth))
            elif behind_long > 1e-6:
                b_gaps.append((right, behind_long))
            bins += plain(split_rect(0, main_depth, main_width, back, cap_width, cap_depth))
            layouts.append((bins, b_gaps))
        else:
            beside_gap = [(right, full_depth - main_depth - back)] if right and full_depth - main_depth - back > 1e-6 else []
            layouts.append((plain(split_rect(0, main_depth, main_width + right, back, cap_width, cap_depth)
                                  + split_rect(main_width, 0, right, main_depth, cap_width, cap_depth)), beside_gap))
            layouts.append((plain(split_rect(main_width, 0, right, main_depth + back, cap_width, cap_depth)
                                  + split_rect(0, main_depth, main_width, back, cap_width, cap_depth)), beside_gap))
        return [((0.0, 0.0, main_width, main_depth), bins, gaps + extra) for bins, extra in layouts]

    def flipped(rect):
        return rect[1], rect[0], rect[3], rect[2]

    def flipped_bin(bin_):
        return (*flipped(bin_), flipped(bin_[4]) if bin_[4] else None)

    def area(bin_):
        return bin_[2] * bin_[3] - (bin_[4][2] * bin_[4][3] if bin_[4] else 0)

    candidates = plan(usable_width, usable_depth, max_width, max_depth)
    # The same with the strips swapped, in case the narrow one is at the back.
    for main, bins, gaps in plan(usable_depth, usable_width, max_depth, max_width):
        candidates.append((flipped(main), [flipped_bin(b) for b in bins], [(d, w) for w, d in gaps]))

    # Prefer fewer bins; then the layout with the longest bin (long utensils are
    # the hard ones to house); then little wasted space and no oversized bin,
    # with empty space counting three times as much.
    def score(candidate):
        bins, gaps = candidate[1], candidate[2]
        longest = max((max(b[2], b[3]) for b in bins), default=0)
        biggest = max((area(b) for b in bins), default=0)
        wasted = sum(w * d for w, d in gaps)
        return len(bins), -round(longest, 1), round(biggest + 3 * wasted)
    return min(candidates, key=score)


def size_text(width, depth):
    return f"{format_mm(width)} x {format_mm(depth)}mm"


def stacked_height(drawer_height, headroom, levels, lip_height, lip_clearance):
    """Overall height of each organizer so `levels` stacked ones fill the drawer.
    Each organizer above the first adds its height less the lip that drops into
    the one below."""
    available = drawer_height - headroom
    return (available + (levels - 1) * (lip_height + lip_clearance)) / levels


def main():
    parser = argparse.ArgumentParser(description='Generate a drawer organizer with diagonal or grid compartments and an open top.')

    parser.add_argument('--compartments', type=int, default=3, help='Number of diagonal compartments to include (default: 3).')
    parser.add_argument('--grid', type=parse_grid, default=None, metavar='COLUMNSxROWS',
                        help='Use a straight grid of compartments instead of diagonal ones, e.g. 3x2.')
    parser.add_argument('--size', type=dimensions_type((1, 2), '--size'), default=None, metavar='SIZE|WxD',
                        help=f'Outside size: one number for square, or WIDTHxDEPTH, in millimeters or with a unit like 10x8in (default: {DEFAULT_SIZE:g} mm).')
    parser.add_argument('--height', type=float, default=None, help=f'Height of the organizer side walls above the floor in millimeters (default: {DEFAULT_HEIGHT:g} mm, or as tall as the drawer allows when --drawer includes a height).')
    parser.add_argument('--drawer', type=dimensions_type((2, 3), '--drawer'), default=None, metavar='WxD[xH]',
                        help="Inside size of your drawer, e.g. 400x300 or 400x300x60 in millimeters, or in inches: 20-1/2x15x3in (a unit on the end applies to every number; fractions like 15-3/4 are fine). Makes a set that fills it: one main organizer as big as the bed allows, with the compartments you asked for, plus single-compartment bins in the space beside and behind it. With a height, the walls are made as tall as the drawer allows, less --drawer_headroom, and never taller; add --height for lower walls (see also --levels).")
    parser.add_argument('--bin_size', type=dimensions_type((1,), '--bin_size'), default=None, metavar='SIZE',
                        help='Longest side allowed for the bins that fill the rest of a --drawer, e.g. 120 or 5in. Smaller values give more, smaller bins (default: as big as the bed can print, including a narrow bin laid diagonally).')
    parser.add_argument('--bed', type=dimensions_type((1, 2), '--bed'), default=(256.0,), metavar='SIZE|WxD',
                        help='Printer bed size, used with --drawer (default: 256, the Bambu Lab X1/P1/X2D bed).')
    parser.add_argument('--drawer_clearance', type=float, default=1.0, help='Total gap left across the drawer in each direction so the organizers drop in easily, in millimeters (default: 1 mm).')
    parser.add_argument('--drawer_headroom', type=float, default=2.0, help='Gap left between the top of the organizers and the top of the drawer when --drawer includes a height, in millimeters (default: 2 mm).')
    parser.add_argument('--levels', type=int, default=1, help='Number of organizers stacked on top of each other in the drawer, used when --drawer includes a height (default: 1).')
    parser.add_argument('--divider_thickness', type=float, default=1.75, help='Thickness of the dividers in millimeters (default: 1.75 mm).')
    parser.add_argument('--wall_thickness', type=float, default=1.75, help='Thickness of the side walls in millimeters (default: 1.75 mm).')
    parser.add_argument('--floor_thickness', type=float, default=1.25, help='Thickness of the floor in millimeters (default: 1.25 mm).')
    parser.add_argument('--stack_lip', action=argparse.BooleanOptionalAction, default=True, help='Add a lip on the bottom so organizers stack (default: on; --no-stack_lip to disable).')
    parser.add_argument('--lip_height', type=float, default=4.0, help='Height of the stacking lip in millimeters (default: 4 mm).')
    parser.add_argument('--lip_clearance', type=float, default=0.3, help='Gap per side between the lip and the walls of the organizer below, in millimeters (default: 0.3 mm).')
    parser.add_argument('--bite_width', type=float, default=0.7, help='Width of the scoop in the top of each divider, as a fraction of its length (default: 0.7, 0 to disable).')
    parser.add_argument('--bite_depth', type=float, default=0.33, help='Depth of the scoop in the top of each divider, as a fraction of its height (default: 0.33, 0 to disable).')
    parser.add_argument('--scale', type=float, default=1.0, help='Scale the whole organizer for a test print, e.g. 0.25 for quarter size. Thicknesses scale too but never below --min_thickness, and the lip clearance is not scaled so the stacking fit is real (default: 1).')
    parser.add_argument('--min_thickness', type=float, default=0.8, help='Thinnest wall, divider or floor --scale will produce, in millimeters (default: 0.8 mm, two lines with a 0.4 mm nozzle).')
    parser.add_argument('--output', type=str, default=None, help='Output filename; .stl or .3mf (default: named after the settings).')
    add_format_argument(parser)

    args = parser.parse_args()

    def fail(message):
        print(f"Error: {message}")
        sys.exit(1)

    if args.compartments < 1:
        fail("Number of compartments must be at least 1.")
    grid = args.grid
    if min(args.wall_thickness, args.divider_thickness, args.floor_thickness) <= 0:
        fail("Thicknesses must all be positive.")
    if not 0 <= args.bite_depth < 1 or not 0 <= args.bite_width < 1:
        fail("--bite_width and --bite_depth must be between 0 and 1.")
    if args.scale <= 0:
        fail("--scale must be positive.")
    if args.levels < 1:
        fail("--levels must be at least 1.")

    # --- Work out what to make: a list of Pieces ---
    # Without --drawer that is one organizer.  With it, a main organizer plus bins.
    gaps = []
    max_width = max_depth = math.inf
    if args.drawer:
        if args.size:
            fail("Use either --drawer or --size, not both.")
        bed = args.bed * 2 if len(args.bed) == 1 else args.bed
        max_width, max_depth = bed[0] - 2 * BED_MARGIN, bed[1] - 2 * BED_MARGIN
        bin_max = args.bin_size[0] if args.bin_size else math.inf
        if bin_max < MIN_BIN:
            fail(f"--bin_size must be at least {MIN_BIN:g} mm.")
        print("Drawer inside: " + " x ".join(format_mm(v) for v in args.drawer) + " mm")
        main, bins, gaps = plan_drawer(args.drawer[0] - args.drawer_clearance,
                                       args.drawer[1] - args.drawer_clearance,
                                       max_width, max_depth, bin_max)
        pieces = [Piece('main organizer', *main, is_main=True)] + [Piece('bin', *b[:4], notch=b[4]) for b in bins]
        if len(args.drawer) == 3:
            if args.levels > 1 and not args.stack_lip:
                fail("Stacking more than one level needs the stack lip.")
            # The drawer height is a ceiling: walls go as high as it allows unless
            # --height asks for less.  Never taller.
            overall = stacked_height(args.drawer[2], args.drawer_headroom, args.levels,
                                     args.lip_height if args.stack_lip else 0,
                                     args.lip_clearance if args.stack_lip else 0)
            tallest = overall - args.floor_thickness
            if args.height is None:
                args.height = tallest
            elif args.height > tallest + 1e-9:
                fail(f"--height {args.height:g} is too tall for the drawer; the most that fits is {tallest:.1f} mm"
                     + (f" with {args.levels} levels." if args.levels > 1 else "."))
        elif args.levels > 1:
            fail("--levels needs a drawer height, e.g. --drawer 400x300x60.")
    else:
        if args.bin_size:
            fail("--bin_size only applies with --drawer.")
        size = args.size or (DEFAULT_SIZE,)
        width, depth = (size[0], size[0]) if len(size) == 1 else size
        pieces = [Piece('organizer', -width / 2, -depth / 2, width, depth, is_main=True)]
    if args.height is None:
        args.height = DEFAULT_HEIGHT
    if args.height <= 0:
        fail("The organizer has no height left; check --drawer, --levels and --floor_thickness.")

    if args.scale != 1:
        pieces = [piece.scaled(args.scale) for piece in pieces]
        args.height *= args.scale
        for name in ('wall_thickness', 'divider_thickness', 'floor_thickness'):
            value = getattr(args, name)
            setattr(args, name, max(value * args.scale, min(value, args.min_thickness)))
        args.lip_height = max(args.lip_height * args.scale, args.floor_thickness + 0.5)
        print(f"Scaling by {args.scale:g}: size {pieces[0].width:g} x {pieces[0].depth:g}mm, height {args.height:g}mm, "
              f"walls {args.wall_thickness:g}mm, dividers {args.divider_thickness:g}mm, "
              f"floor {args.floor_thickness:g}mm, lip {args.lip_height:g}mm; "
              f"lip clearance stays {args.lip_clearance:g}mm.")

    # Every piece needs room for its walls (and the stacking lip's step) plus a
    # cavity, in each rectangle of its outline.
    wall_room = 2 * args.wall_thickness + (2 * (args.wall_thickness + args.lip_clearance) if args.stack_lip else 0)
    for piece in pieces:
        for _, _, w, d in outline_rects(piece.width, piece.depth, piece.local_notch()):
            if min(w, d) <= max(4 * args.wall_thickness, wall_room + 2):
                fail(f"The {piece.label} ({piece.describe()}) is too small for the wall thickness.")
    if args.stack_lip:
        if args.lip_height <= args.floor_thickness:
            fail("--lip_height must be greater than --floor_thickness.")
        # The lip of the organizer above must not reach down to this one's chamfer.
        chamfer_top = args.lip_height + args.wall_thickness * math.sqrt(2) + args.lip_clearance
        if args.floor_thickness + args.height - args.lip_height - args.lip_clearance <= chamfer_top:
            if args.scale != 1:
                fail(f"At --scale {args.scale:g} the organizer is too short for the stacking lip. Use a larger --scale or --no-stack_lip.")
            fail("Organizer is too short for the stacking lip. Increase the height or reduce --lip_height.")

    layout = f"{grid[0]}x{grid[1]} grid" if grid else f"{args.compartments} diagonal compartment(s)"
    width, depth = pieces[0].width, pieces[0].depth
    print(f"Creating organizer: {size_text(width, depth)}, height={format_mm(args.height)}mm, {layout}")

    # Build each distinct piece once.  Only the main organizer gets dividers.
    meshes = {}
    for piece in pieces:
        if piece.key in meshes:
            continue
        meshes[piece.key] = create_organizer(
            width=piece.width,
            depth=piece.depth,
            height=args.height,
            wall_thickness=args.wall_thickness,
            divider_thickness=args.divider_thickness,
            floor_thickness=args.floor_thickness,
            compartments=args.compartments if piece.is_main else 1,
            grid=grid if piece.is_main else None,
            stack_lip=args.stack_lip,
            lip_height=args.lip_height,
            lip_clearance=args.lip_clearance,
            bite_width_ratio=args.bite_width,
            bite_depth_ratio=args.bite_depth,
            notch=piece.local_notch(),
        )
        if not meshes[piece.key].is_watertight:
            print(f"Warning: The {piece.label} mesh is not watertight; the slicer may need to repair it.")
    organizer = meshes[pieces[0].key]

    # Determine output filename
    stem = f"drawer_organizer_{grid[0]}x{grid[1]}_grid" if grid else f"drawer_organizer_{args.compartments}_compartments"
    if args.drawer:
        stem += f"_for_{args.drawer[0]:.0f}x{args.drawer[1]:.0f}_drawer"
    elif args.size:
        stem += f"_{width:.0f}x{depth:.0f}"
    if args.scale != 1:
        stem += f"_scale_{args.scale:g}"
    output_filename = output_path(args.output, stem, args.format)
    is_set = len(pieces) * args.levels > 1

    if not is_set:
        save(Part(stem.replace('_', ' ').capitalize(), organizer), output_filename)
        written = [output_filename]
    else:
        # A piece longer than the bed is turned so it prints corner to corner.
        def for_printing(piece):
            mesh = meshes[piece.key].copy()
            turn = print_rotation(piece.width, piece.depth, max_width, max_depth)
            if turn:
                shapes.rotate_z(mesh, turn)
            return mesh, turn

        if output_filename.lower().endswith('.3mf'):
            # One file with every piece, laid out the way the set sits in the
            # drawer (spread apart a little so no two objects touch).  Turned
            # pieces would overlap their neighbors there, so they go alongside.
            # Levels sit side by side because everything has to print from z = 0.
            parts, level_start = [], 0.0
            for level in range(args.levels):
                bin_number = 0
                edge = max(piece.x + piece.width for piece in pieces) * SET_SPREAD
                parked_y = 0.0
                for piece in pieces:
                    mesh, turn = for_printing(piece)
                    if turn:
                        size = mesh.extents
                        mesh.apply_translation([level_start + edge + 20 + size[0] / 2, parked_y + size[1] / 2, 0])
                        parked_y += size[1] + 20
                    else:
                        mesh.apply_translation([level_start + (piece.x + piece.width / 2) * SET_SPREAD,
                                                (piece.y + piece.depth / 2) * SET_SPREAD, 0])
                    bin_number += 0 if piece.is_main else 1
                    name = "Main organizer" if piece.is_main else f"Bin {bin_number}"
                    name += f" ({format_mm(piece.width)} x {format_mm(piece.depth)} mm" + (", L-shaped)" if piece.notch else ")")
                    parts.append(Part(name if args.levels == 1 else f"Level {level + 1} {name.lower()}", mesh))
                level_start = max(part.mesh.bounds[1][0] for part in parts) + 20
            save(parts, output_filename)
            written = [output_filename]
        else:
            # STL holds one object per file, so write one file per distinct piece.
            base = output_filename[:-len('.stl')] if output_filename.lower().endswith('.stl') else output_filename
            written, done = [], set()
            for piece in pieces:
                if piece.key in done:
                    continue
                done.add(piece.key)
                name = f"{base}_main.stl" if piece.is_main else (
                    f"{base}_bin_{piece.width:.0f}x{piece.depth:.0f}" + ("_L" if piece.notch else "") + ".stl")
                save(Part('Organizer', for_printing(piece)[0]), name)
                written.append(name)
    for name in written:
        print(f"Organizer exported to '{name}'.")

    extents = organizer.extents

    # Print out the arguments used
    print("\nOrganizer Parameters Used:")
    print(f"  Compartments: {layout}")
    print(f"  Size: {size_text(width, depth)}")
    print(f"  Height: {format_mm(args.height)}mm (overall {extents[2]:.2f}mm including floor)")
    print(f"  Divider Thickness: {args.divider_thickness:g}mm")
    print(f"  Wall Thickness: {args.wall_thickness:g}mm")
    print(f"  Floor Thickness: {args.floor_thickness:g}mm")
    if args.stack_lip:
        print(f"  Stacking Lip: {args.lip_height:g}mm high, {args.lip_clearance:g}mm clearance")
        print(f"  Stacked organizers add {args.floor_thickness + args.height - args.lip_height - args.lip_clearance:.2f}mm each")
    else:
        print("  Stacking Lip: none")
    if args.scale != 1:
        print(f"  Scale: {args.scale:g} (lip clearance not scaled)")
    print(f"  Output File: {', '.join(written)}")
    print(f"  Bounding box: {extents[0]:.2f} x {extents[1]:.2f} x {extents[2]:.2f} mm")

    if args.drawer:
        filled_width = max(piece.x + piece.width for piece in pieces)
        filled_depth = max(piece.y + piece.depth for piece in pieces)
        print(f"\nSet for the {size_text(args.drawer[0], args.drawer[1])} drawer"
              + (f" at scale {args.scale:g}" if args.scale != 1 else "")
              + f", filling {size_text(filled_width, filled_depth)}"
              + (f", {args.levels} levels high" if args.levels > 1 else "") + ":")
        counts = {}
        for piece in pieces:
            counts.setdefault(piece.key, [piece, 0])[1] += args.levels
        for piece, number in sorted(counts.values(), key=lambda entry: (not entry[0].is_main, entry[0].key[:2], bool(entry[0].notch))):
            detail = f", {layout}" if piece.is_main else ", single compartment"
            turn = print_rotation(piece.width, piece.depth, max_width, max_depth)
            if turn:
                detail += f" (longer than the bed: it is turned {turn:g} degrees in the file to print corner to corner)"
            print(f"  {number} x {piece.label} {piece.describe()}{detail}")
        for w, d in gaps:
            if min(w, d) > 0.05:
                print(f"  A {size_text(w * args.scale, d * args.scale)} space is left empty (too small for a bin).")
        if is_set and written[0].lower().endswith('.3mf'):
            print("  The file holds every piece. In the slicer, use Arrange All to spread them over plates.")
            bed_limit = min(max_width, max_depth) - AUTO_ARRANGE_MARGIN
            if any(max(piece.width, piece.depth) > bed_limit for piece in pieces):
                print(f"  Pieces longer than about {bed_limit:.0f}mm nearly fill the bed, and the slicer's auto-arrange "
                      "may leave them off: add a plate and drag each of those onto its own plate.")
        elif is_set:
            print("  STL files hold one object each: print the number of copies listed above, "
                  "or use --format 3mf to get the whole set in one file.")
        if len(args.drawer) == 3 and args.scale == 1:
            stack = extents[2] + (args.levels - 1) * (extents[2] - args.lip_height - args.lip_clearance)
            print(f"  Stack height {stack:.1f}mm in a {format_mm(args.drawer[2])}mm deep drawer.")


if __name__ == "__main__":
    main()
