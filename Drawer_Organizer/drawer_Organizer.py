# Generate STL file for drawer organizer.  Can have one or more compartments.
# Compartments are diagonal to maximize length in the center compartment since
# Many 3D printer beds are less than 11 inches (~ 256 mm) in size which limits
# How large a single piece organizer can be.
#
# Optionally the organizer gets a stacking lip on the bottom: the lowest part of
# the outer wall is stepped inward so it drops inside the walls of an identical
# organizer underneath.  The step is a 45 degree chamfer rather than a flat ledge
# so it prints without supports, and the floor stays on the build plate.
#
# Requires: numpy, trimesh, manifold3d (pip install -r requirements.txt)

import numpy as np
import trimesh
import argparse
import math
import sys

BOOLEAN_ENGINE = 'manifold'


def square_points(side, z):
    h = side / 2
    return [[-h, -h, z], [h, -h, z], [h, h, z], [-h, h, z]]


def stepped_block(bottom_side, top_side, z_bottom, z_step, z_top):
    """Square solid centered on the Z axis.

    It is bottom_side wide from z_bottom up to z_step, flares out at 45 degrees
    to top_side, then goes straight up to z_top.  With bottom_side == top_side
    it is just a box.
    """
    if bottom_side >= top_side:
        rings = [(top_side, z_bottom), (top_side, z_top)]
    else:
        rise = (top_side - bottom_side) / 2  # 45 degree chamfer
        rings = [(bottom_side, z_bottom), (bottom_side, z_step),
                 (top_side, z_step + rise), (top_side, z_top)]
    return loft_squares(rings)


def loft_squares(rings):
    """Closed solid through a stack of (side, z) squares, bottom to top."""
    vertices = []
    for side, z in rings:
        vertices += square_points(side, z)

    top = 4 * (len(rings) - 1)
    faces = [[0, 2, 1], [0, 3, 2],                              # bottom cap
             [top, top + 1, top + 2], [top, top + 2, top + 3]]  # top cap
    for r in range(len(rings) - 1):
        lo, hi = 4 * r, 4 * (r + 1)
        for k in range(4):
            k1 = (k + 1) % 4
            faces += [[lo + k, lo + k1, hi + k1], [lo + k, hi + k1, hi + k]]
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def create_divider(start, direction, length, extension, thickness, z_top,
                   floor_thickness, bite_width, bite_depth):
    """Divider wall running from start along direction for length mm.

    The wall is extended by extension mm past both ends so it buries itself in
    the outer walls; anything sticking outside the organizer is trimmed later.
    """
    total_length = length + 2 * extension
    # Start halfway into the floor: a divider bottom flush with the underside of
    # the floor leaves coplanar faces that the boolean union handles badly.
    z_bottom = floor_thickness / 2
    wall = trimesh.creation.box(extents=[total_length, thickness, z_top - z_bottom])
    wall.apply_translation([length / 2, 0, (z_bottom + z_top) / 2])

    if bite_depth > 0 and bite_width > 0:
        # Elliptical scoop out of the top edge so it's easy to reach into the
        # compartments.  An elliptic cylinder gives a smooth curve, unlike a
        # low-poly ellipsoid.
        bite_depth = min(bite_depth, z_top - floor_thickness - 1.0)
        bite = trimesh.creation.cylinder(radius=1.0, height=thickness * 4, sections=128)
        bite.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        bite.apply_scale([bite_width / 2, 1, bite_depth])
        bite.apply_translation([length / 2, 0, z_top])
        wall = trimesh.boolean.difference([wall, bite], engine=BOOLEAN_ENGINE)

    angle = math.atan2(direction[1], direction[0])
    T = trimesh.transformations.concatenate_matrices(
        trimesh.transformations.translation_matrix([start[0], start[1], 0]),
        trimesh.transformations.rotation_matrix(angle, [0, 0, 1]),
    )
    wall.apply_transform(T)
    return wall


def clip_line_to_square(point, direction, half):
    """Return the (start, end) of the line through point along direction inside
    the square [-half, half]^2, or None if it misses.  direction must have
    non-zero components."""
    t_lo = max((-half - point[0]) / direction[0], (-half - point[1]) / direction[1])
    t_hi = min((half - point[0]) / direction[0], (half - point[1]) / direction[1])
    if t_hi - t_lo <= 1e-6:
        return None
    return point + t_lo * direction, point + t_hi * direction


def create_organizer(size, height, wall_thickness, divider_thickness, floor_thickness,
                     compartments, stack_lip, lip_height, lip_clearance,
                     bite_width_ratio, bite_depth_ratio):
    total_height = floor_thickness + height
    t = wall_thickness

    # Outer body: the solid envelope of the organizer.  `size` is the true outside
    # footprint, so walls are inside it rather than centered on its edge.
    if stack_lip:
        inset = t + lip_clearance                  # lip slips inside the walls below
        chamfer_shift = t * (math.sqrt(2) - 1)     # keeps the chamfered wall t thick
        outer = stepped_block(size - 2 * inset, size, 0, lip_height, total_height)
        cavity = stepped_block(size - 2 * inset - 2 * t, size - 2 * t, floor_thickness,
                               lip_height + chamfer_shift, total_height + 1)
        # The organizer above rests on its chamfer, lip_height + lip_clearance up
        # from its bottom, so it hangs that far down into this one.  Dividers have
        # to stop short of that or the organizer above sits on them instead.
        divider_top = total_height - lip_height - 2 * lip_clearance
    else:
        outer = stepped_block(size, size, 0, 0, total_height)
        cavity = stepped_block(size - 2 * t, size - 2 * t, floor_thickness, 0, total_height + 1)
        divider_top = total_height

    shell = trimesh.boolean.difference([outer, cavity], engine=BOOLEAN_ENGINE)
    components = [shell]

    # Create dividers, parallel to the y = x diagonal and equally spaced across
    # the inside of the organizer.
    if compartments > 1:
        inner_half = size / 2 - t
        direction = np.array([1.0, 1.0]) / math.sqrt(2)
        normal = np.array([-1.0, 1.0]) / math.sqrt(2)
        total_width = 2 * inner_half * math.sqrt(2)
        compartment_width = total_width / compartments

        for i in range(1, compartments):
            d = (i - compartments / 2) * compartment_width
            segment = clip_line_to_square(normal * d, direction, inner_half)
            if segment is None:
                print(f"Warning: Divider {i} doesn't cross the organizer. Skipping.")
                continue
            start, end = segment
            length = float(np.linalg.norm(end - start))
            if length < 1.0:
                print(f"Warning: Divider {i} is only {length:.2f}mm long. Skipping.")
                continue

            components.append(create_divider(
                start=start,
                direction=direction,
                length=length,
                extension=2 * t + divider_thickness,
                thickness=divider_thickness,
                z_top=divider_top,
                floor_thickness=floor_thickness,
                bite_width=length * bite_width_ratio,
                bite_depth=(divider_top - floor_thickness) * bite_depth_ratio,
            ))

    organizer = trimesh.boolean.union(components, engine=BOOLEAN_ENGINE)
    # Trim divider ends that poke out past the outer walls.
    organizer = trimesh.boolean.intersection([organizer, outer], engine=BOOLEAN_ENGINE)
    return organizer


def main():
    parser = argparse.ArgumentParser(description='Generate a square kitchen drawer organizer with parallel diagonal compartments and an open top.')

    parser.add_argument('--compartments', type=int, default=3, help='Number of compartments to include (default: 3).')
    parser.add_argument('--size', type=float, default=250.0, help='Outside size of the organizer (width and depth) in millimeters (default: 250 mm).')
    parser.add_argument('--height', type=float, default=35.0, help='Height of the organizer side walls above the floor in millimeters (default: 35 mm).')
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
    parser.add_argument('--output', type=str, default='drawer_organizer.stl', help='Output STL filename (default: drawer_organizer.stl).')

    args = parser.parse_args()

    if args.scale <= 0:
        print("Error: --scale must be positive.")
        sys.exit(1)
    if args.scale != 1:
        args.size *= args.scale
        args.height *= args.scale
        for name in ('wall_thickness', 'divider_thickness', 'floor_thickness'):
            value = getattr(args, name)
            setattr(args, name, max(value * args.scale, min(value, args.min_thickness)))
        args.lip_height = max(args.lip_height * args.scale, args.floor_thickness + 0.5)
        print(f"Scaling by {args.scale:g}: size {args.size:g}mm, height {args.height:g}mm, "
              f"walls {args.wall_thickness:g}mm, dividers {args.divider_thickness:g}mm, "
              f"floor {args.floor_thickness:g}mm, lip {args.lip_height:g}mm; "
              f"lip clearance stays {args.lip_clearance:g}mm.")

    if args.compartments < 1:
        print("Error: Number of compartments must be at least 1.")
        sys.exit(1)
    if min(args.size, args.height, args.wall_thickness, args.divider_thickness, args.floor_thickness) <= 0:
        print("Error: Size, height and thicknesses must all be positive.")
        sys.exit(1)
    if args.size <= 4 * args.wall_thickness:
        print("Error: Size is too small for the wall thickness.")
        sys.exit(1)
    if not 0 <= args.bite_depth < 1 or not 0 <= args.bite_width < 1:
        print("Error: --bite_width and --bite_depth must be between 0 and 1.")
        sys.exit(1)
    if args.stack_lip:
        if args.lip_height <= args.floor_thickness:
            print("Error: --lip_height must be greater than --floor_thickness.")
            sys.exit(1)
        # The lip of the organizer above must not reach down to this one's chamfer.
        chamfer_top = args.lip_height + args.wall_thickness * math.sqrt(2) + args.lip_clearance
        if args.floor_thickness + args.height - args.lip_height - args.lip_clearance <= chamfer_top:
            if args.scale != 1:
                print(f"Error: At --scale {args.scale:g} the organizer is too short for the stacking lip. Use a larger --scale or --no-stack_lip.")
            else:
                print("Error: Organizer is too short for the stacking lip. Increase --height or reduce --lip_height.")
            sys.exit(1)

    print(f"Creating organizer: size={args.size}mm x {args.size}mm, height={args.height}mm")
    print(f"Wall thickness: {args.wall_thickness}mm, Divider thickness: {args.divider_thickness}mm, Floor thickness: {args.floor_thickness}mm")
    print(f"Compartments: {args.compartments}")

    organizer = create_organizer(
        size=args.size,
        height=args.height,
        wall_thickness=args.wall_thickness,
        divider_thickness=args.divider_thickness,
        floor_thickness=args.floor_thickness,
        compartments=args.compartments,
        stack_lip=args.stack_lip,
        lip_height=args.lip_height,
        lip_clearance=args.lip_clearance,
        bite_width_ratio=args.bite_width,
        bite_depth_ratio=args.bite_depth,
    )

    if not organizer.is_watertight:
        print("Warning: Generated mesh is not watertight; the slicer may need to repair it.")

    # Determine output filename
    if args.output == 'drawer_organizer.stl':
        output_filename = f"drawer_organizer_{args.compartments}_compartments.stl"
        if args.scale != 1:
            output_filename = output_filename.replace('.stl', f'_scale_{args.scale:g}.stl')
    else:
        output_filename = args.output

    # Export the model to an STL file
    organizer.export(output_filename)
    print(f"Organizer exported to '{output_filename}'.")

    extents = organizer.extents

    # Print out the arguments used
    print("\nOrganizer Parameters Used:")
    print(f"  Compartments: {args.compartments}")
    print(f"  Size: {args.size}mm x {args.size}mm")
    print(f"  Height: {args.height}mm (overall {extents[2]:.2f}mm including floor)")
    print(f"  Divider Thickness: {args.divider_thickness}mm")
    print(f"  Wall Thickness: {args.wall_thickness}mm")
    print(f"  Floor Thickness: {args.floor_thickness}mm")
    if args.stack_lip:
        print(f"  Stacking Lip: {args.lip_height}mm high, {args.lip_clearance}mm clearance")
        print(f"  Stacked organizers add {args.floor_thickness + args.height - args.lip_height - args.lip_clearance:.2f}mm each")
    else:
        print("  Stacking Lip: none")
    if args.scale != 1:
        print(f"  Scale: {args.scale:g} (lip clearance not scaled)")
    print(f"  Output File: {output_filename}")
    print(f"  Bounding box: {extents[0]:.2f} x {extents[1]:.2f} x {extents[2]:.2f} mm")

if __name__ == "__main__":
    main()
