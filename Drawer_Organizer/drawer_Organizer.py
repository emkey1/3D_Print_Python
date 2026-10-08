# Generate a drawer organizer.  Can have one or more compartments.
#
# Compartments are diagonal by default to maximize length in the center
# compartment, since many 3D printer beds are less than 11 inches (~ 256 mm)
# in size which limits how large a single piece organizer can be.  A straight
# grid of compartments is also available.
#
# Give it your drawer's inside measurements with --drawer and it works out how
# many organizers it takes to fill the drawer, sized so each one fits the bed.
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
                     bite_width_ratio, bite_depth_ratio):
    total_height = floor_thickness + height
    t = wall_thickness

    # Outer body: the solid envelope of the organizer.  width x depth is the true
    # outside footprint, so walls are inside it rather than centered on its edge.
    if stack_lip:
        inset = t + lip_clearance                  # lip slips inside the walls below
        chamfer_shift = t * (math.sqrt(2) - 1)     # keeps the chamfered wall t thick
        outer = stepped_block(width - 2 * inset, depth - 2 * inset, inset,
                              0, lip_height, total_height)
        cavity = stepped_block(width - 2 * inset - 2 * t, depth - 2 * inset - 2 * t, inset,
                               floor_thickness, lip_height + chamfer_shift, total_height + 1)
        # The organizer above rests on its chamfer, lip_height + lip_clearance up
        # from its bottom, so it hangs that far down into this one.  Dividers have
        # to stop short of that or the organizer above sits on them instead.
        divider_top = total_height - lip_height - 2 * lip_clearance
    else:
        outer = stepped_block(width, depth, 0, 0, 0, total_height)
        cavity = stepped_block(width - 2 * t, depth - 2 * t, 0, floor_thickness, 0, total_height + 1)
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


def fit_drawer(drawer_width, drawer_depth, bed_width, bed_depth, slack):
    """Split the drawer floor into the fewest equal organizers that fit the bed.
    Returns (columns, rows, organizer_width, organizer_depth)."""
    usable_width, usable_depth = drawer_width - slack, drawer_depth - slack
    max_width, max_depth = bed_width - 2 * BED_MARGIN, bed_depth - 2 * BED_MARGIN

    best = None
    # On a rectangular bed the organizers may fit better turned sideways.
    for fit_width, fit_depth in ((max_width, max_depth), (max_depth, max_width)):
        columns = math.ceil(usable_width / fit_width)
        rows = math.ceil(usable_depth / fit_depth)
        if best is None or columns * rows < best[0] * best[1]:
            best = (columns, rows, usable_width / columns, usable_depth / rows)
    return best


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
    parser.add_argument('--height', type=float, default=None, help=f'Height of the organizer side walls above the floor in millimeters (default: {DEFAULT_HEIGHT:g} mm).')
    parser.add_argument('--drawer', type=dimensions_type((2, 3), '--drawer'), default=None, metavar='WxD[xH]',
                        help="Inside size of your drawer, e.g. 400x300 or 400x300x60 in millimeters, or in inches: 20-1/2x15x3in (a unit on the end applies to every number; fractions like 15-3/4 are fine). Works out how many organizers fill it and how big each one is. With a height, the organizers are made tall enough to fill it (see --levels).")
    parser.add_argument('--bed', type=dimensions_type((1, 2), '--bed'), default=(256.0,), metavar='SIZE|WxD',
                        help='Printer bed size, used with --drawer (default: 256, the Bambu Lab X1/P1/X2D bed).')
    parser.add_argument('--drawer_clearance', type=float, default=1.0, help='Total gap left across the drawer in each direction so the organizers drop in easily, in millimeters (default: 1 mm).')
    parser.add_argument('--drawer_headroom', type=float, default=2.0, help='Gap left above the organizers when --drawer includes a height, in millimeters (default: 2 mm).')
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

    # --- Work out the organizer's outside dimensions ---
    tiles = None
    if args.drawer:
        if args.size:
            fail("Use either --drawer or --size, not both.")
        bed = args.bed * 2 if len(args.bed) == 1 else args.bed
        print("Drawer inside: " + " x ".join(format_mm(v) for v in args.drawer) + " mm")
        columns, rows, width, depth = fit_drawer(args.drawer[0], args.drawer[1], bed[0], bed[1],
                                                 args.drawer_clearance)
        tiles = (columns, rows)
        if len(args.drawer) == 3:
            if args.height is not None:
                fail("--drawer includes a height, so leave out --height.")
            if args.levels > 1 and not args.stack_lip:
                fail("Stacking more than one level needs the stack lip.")
            overall = stacked_height(args.drawer[2], args.drawer_headroom, args.levels,
                                     args.lip_height if args.stack_lip else 0,
                                     args.lip_clearance if args.stack_lip else 0)
            args.height = overall - args.floor_thickness
        elif args.levels > 1:
            fail("--levels needs a drawer height, e.g. --drawer 400x300x60.")
    else:
        size = args.size or (DEFAULT_SIZE,)
        width, depth = (size[0], size[0]) if len(size) == 1 else size
    if args.height is None:
        args.height = DEFAULT_HEIGHT
    if args.height <= 0:
        fail("The organizer has no height left; check --drawer, --levels and --floor_thickness.")

    if args.scale != 1:
        width *= args.scale
        depth *= args.scale
        args.height *= args.scale
        for name in ('wall_thickness', 'divider_thickness', 'floor_thickness'):
            value = getattr(args, name)
            setattr(args, name, max(value * args.scale, min(value, args.min_thickness)))
        args.lip_height = max(args.lip_height * args.scale, args.floor_thickness + 0.5)
        print(f"Scaling by {args.scale:g}: size {width:g} x {depth:g}mm, height {args.height:g}mm, "
              f"walls {args.wall_thickness:g}mm, dividers {args.divider_thickness:g}mm, "
              f"floor {args.floor_thickness:g}mm, lip {args.lip_height:g}mm; "
              f"lip clearance stays {args.lip_clearance:g}mm.")

    if min(width, depth) <= 4 * args.wall_thickness:
        fail("Size is too small for the wall thickness.")
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
    print(f"Creating organizer: {format_mm(width)}mm x {format_mm(depth)}mm, height={format_mm(args.height)}mm, {layout}")

    organizer = create_organizer(
        width=width,
        depth=depth,
        height=args.height,
        wall_thickness=args.wall_thickness,
        divider_thickness=args.divider_thickness,
        floor_thickness=args.floor_thickness,
        compartments=args.compartments,
        grid=grid,
        stack_lip=args.stack_lip,
        lip_height=args.lip_height,
        lip_clearance=args.lip_clearance,
        bite_width_ratio=args.bite_width,
        bite_depth_ratio=args.bite_depth,
    )

    if not organizer.is_watertight:
        print("Warning: Generated mesh is not watertight; the slicer may need to repair it.")

    # Determine output filename
    stem = f"drawer_organizer_{grid[0]}x{grid[1]}_grid" if grid else f"drawer_organizer_{args.compartments}_compartments"
    if tiles or args.size:
        stem += f"_{width:.0f}x{depth:.0f}"
    if args.scale != 1:
        stem += f"_scale_{args.scale:g}"
    output_filename = output_path(args.output, stem, args.format)

    save(Part(stem.replace('_', ' ').capitalize(), organizer), output_filename)
    print(f"Organizer exported to '{output_filename}'.")

    extents = organizer.extents

    # Print out the arguments used
    print("\nOrganizer Parameters Used:")
    print(f"  Compartments: {layout}")
    print(f"  Size: {format_mm(width)}mm x {format_mm(depth)}mm")
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
    print(f"  Output File: {output_filename}")
    print(f"  Bounding box: {extents[0]:.2f} x {extents[1]:.2f} x {extents[2]:.2f} mm")

    if tiles:
        columns, rows = tiles
        count = columns * rows * args.levels
        print(f"\nTo fill the {format_mm(args.drawer[0])} x {format_mm(args.drawer[1])}mm drawer: print {count} of these "
              f"({columns} across x {rows} deep" + (f", {args.levels} levels high" if args.levels > 1 else "") + ").")
        if len(args.drawer) == 3:
            stack = extents[2] + (args.levels - 1) * (extents[2] - args.lip_height - args.lip_clearance)
            print(f"  Stack height {stack:.1f}mm in a {format_mm(args.drawer[2])}mm deep drawer.")


if __name__ == "__main__":
    main()
