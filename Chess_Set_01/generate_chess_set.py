# generate_chess_set.py
#
# Generates STL files for a simple chess set.  Each piece is built from
# primitive shapes that are merged with real boolean unions, so every piece is
# a single watertight solid.  Heights are measured from the bottom of the base
# and parts are positioned from each other's dimensions, so changing one size
# doesn't leave a gap or a floating part somewhere else.
#
# Requires: numpy, trimesh, manifold3d

import argparse  # Import argparse for command-line argument parsing
import math      # Import math for grid calculations
import sys
from pathlib import Path

import numpy as np
import trimesh.transformations as tt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from printlib import Part, add_format_argument, output_path, save  # noqa: E402
from printlib.shapes import (block, cone, cylinder, difference, drop_to_bed,  # noqa: E402
                             ellipsoid, frustum, prism_y, ring_of, sphere, union)

BASE_RADIUS = 12
BASE_HEIGHT = 5


def base():
    return cylinder(BASE_RADIUS, 0, BASE_HEIGHT)


# --- Pieces ---------------------------------------------------------------

def pawn():
    body_top = 30
    head_radius = 6

    pawn = union([
        base(),
        cylinder(7.5, BASE_HEIGHT, body_top),
        sphere(head_radius, [0, 0, body_top + 3.5]),
    ])
    return drop_to_bed(pawn, scale=4 / 5)


# A more traditional looking rook, not the default
def rook_alt():
    body_radius = 7.5
    top_outer_radius = 11
    top_inner_radius = 9
    flare_top = 36.4
    top_z = 41.1

    notch_height = 3.3
    notch_depth = 3
    notch_width = 2

    # The body flares out at the same angle as the default rook's cone.
    flare_bottom = flare_top - 15 * (top_outer_radius - body_radius) / top_outer_radius

    rook = union([
        base(),
        cylinder(body_radius, BASE_HEIGHT, flare_bottom),
        frustum(body_radius, top_outer_radius, flare_bottom, flare_top),
        cylinder(top_outer_radius, flare_top, top_z),
    ])

    # Hollow out the top into a cup, then cut the notches through its rim.
    cup = cylinder(top_inner_radius, flare_top, top_z + 1)
    notch = block([notch_width, notch_depth + 1, notch_height + 1],
                  [0, 0, top_z - notch_height + (notch_height + 1) / 2])
    notches = ring_of(notch, 6, top_outer_radius - notch_depth / 2 + 0.5)

    return drop_to_bed(difference(rook, [cup] + notches))


def rook():
    body_radius = 7.5
    top_radius = 11
    top_bottom = 35.5
    top_height = 6
    top_z = top_bottom + top_height

    notch_height = 5
    notch_depth = 5
    notch_width = 2

    # Cone-shaped flare: starts at the body and widens to the top's radius.
    # Same slope as a 15 mm tall cone of radius 11.
    flare_bottom = top_bottom - 15 * (top_radius - body_radius) / top_radius

    rook = union([
        base(),
        cylinder(body_radius, BASE_HEIGHT, flare_bottom),
        frustum(body_radius, top_radius, flare_bottom, top_bottom),
        cylinder(top_radius, top_bottom, top_z),
    ])

    # Notches are open at the top and run out past the outside edge so they
    # cut cleanly through it.
    notch = block([notch_width, notch_depth + 1, notch_height + 1],
                  [0, 0, top_z - notch_height + (notch_height + 1) / 2])
    notches = ring_of(notch, 6, top_radius - notch_depth / 2 + 0.5)

    return drop_to_bed(difference(rook, notches))


def knight():
    knight = union([
        base(),
        ellipsoid([9, 4.5, 18], [0, 0, 18.5]),    # Body
        ellipsoid([5, 2.5, 7.5], [0, 0, 37.5]),   # Head
    ])
    return drop_to_bed(knight)


def bishop():
    body_top = 42.5
    head_radius = 6
    head_center = body_top + 0.5

    bishop = union([
        base(),
        cylinder(8, BASE_HEIGHT, body_top),
        sphere(head_radius, [0, 0, head_center]),
    ])

    # Mitre: a slanted slot cut into one side of the top of the head.  It only
    # goes about halfway down, so the head stays in one piece.
    slot = block([1.5, head_radius * 2 + 2, 8], [0, 0, 0])
    slot.apply_transform(tt.rotation_matrix(np.radians(35), [0, 1, 0]))
    slot.apply_translation([2, 0, head_center + 3.5])

    return drop_to_bed(difference(bishop, [slot]))


def queen():
    body_radius = 8.5
    body_top = 48.5
    crown_height = 15.5
    crown_bottom = body_top - 4
    sphere_radius = 1.5

    # Small balls around the top edge of the body
    balls = ring_of(sphere(sphere_radius, [0, 0, body_top]), 6, 8)

    queen = union([
        base(),
        cylinder(10, 4.5, 6.5),                     # Decorative collar
        cylinder(body_radius, BASE_HEIGHT, body_top),
        cone(5, crown_bottom, crown_height),
    ] + balls)
    return drop_to_bed(queen, scale=1.25)


def king():
    body_top = 48.7
    crown_bottom = 47.4
    crown_top = 50.4
    ball_radius = 5
    ball_center = crown_top + 0.6

    # Cross on top.  The arms are chamfered underneath at 45 degrees so they
    # print without supports.
    bar = 2.5                       # Thickness of the cross bars
    cross_bottom = ball_center + ball_radius - 2
    cross_top = cross_bottom + 9
    arm_top = cross_top - 2
    arm_bottom = arm_top - bar
    arm_reach = 3.5                 # From the center to the end of an arm
    arm_underside = arm_bottom - (arm_reach - bar / 2)

    upright = block([bar, bar, cross_top - cross_bottom],
                    [0, 0, (cross_bottom + cross_top) / 2])
    arms = prism_y([(-arm_reach, arm_bottom), (-bar / 2, arm_underside),
                    (bar / 2, arm_underside), (arm_reach, arm_bottom),
                    (arm_reach, arm_top), (-arm_reach, arm_top)], bar)

    king = union([
        base(),
        cylinder(10, 4.6, 6.6),                     # Decorative collar
        cylinder(9, BASE_HEIGHT, body_top),
        cylinder(6, crown_bottom, crown_top),
        sphere(ball_radius, [0, 0, ball_center]),
        upright,
        arms,
    ])
    return drop_to_bed(king, scale=1.25)


# Define the list of possible piece names and their corresponding functions
piece_names = ['pawn', 'rook', 'rook_alt', 'knight', 'bishop', 'queen', 'king']
piece_functions = {
    'pawn': pawn,
    'rook': rook,
    'rook_alt': rook_alt,
    'knight': knight,
    'bishop': bishop,
    'queen': queen,
    'king': king
}

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Generate a set of chess pieces.')

    # Define command-line arguments for each piece with default=None
    parser.add_argument('--pawn', type=int, default=None, help='Number of pawns to include.')
    parser.add_argument('--rook', type=int, default=None, help='Number of rooks to include.')
    parser.add_argument('--rook_alt', type=int, default=None, help='Number of alternative rooks to include.')
    parser.add_argument('--knight', type=int, default=None, help='Number of knights to include.')
    parser.add_argument('--bishop', type=int, default=None, help='Number of bishops to include.')
    parser.add_argument('--queen', type=int, default=None, help='Number of queens to include.')
    parser.add_argument('--king', type=int, default=None, help='Number of kings to include.')
    parser.add_argument('--sides', type=int, choices=(1, 2), default=1,
                        help='1 for one side, 2 for both. With 2 and --format 3mf, white pieces print '
                             'with filament 1 and black pieces with filament 2 (default: 1).')
    parser.add_argument('--output', type=str, default=None, help='Output filename; .stl or .3mf (default: named after the pieces).')
    add_format_argument(parser)

    args = parser.parse_args()

    # Define default counts if no arguments are provided
    default_counts = {
        'pawn': 8,
        'rook': 2,
        'rook_alt': 0,
        'knight': 2,
        'bishop': 2,
        'queen': 1,
        'king': 1
    }

    # Determine if any pieces are specified via command-line
    specified_pieces = {piece: getattr(args, piece) for piece in piece_names if getattr(args, piece) is not None}

    if specified_pieces:
        print("Pieces specified via command-line arguments. Only these pieces will be included:")
        # Use specified counts
        selected_pieces = {piece: count for piece, count in specified_pieces.items() if count > 0}
    else:
        print("No pieces specified. Using default counts:")
        # Use default counts
        selected_pieces = {piece: count for piece, count in default_counts.items() if count > 0}

    # Display selected pieces and their counts
    for piece, count in selected_pieces.items():
        print(f"  {piece.capitalize()}: {count}")

    # Build the list of pieces based on the selected counts.  Each piece type
    # is only built once and then copied.
    sides = [('White', '#F2F2F2', 1), ('Black', '#202020', 2)][:args.sides]
    spacing_x = 40  # Horizontal spacing between pieces
    spacing_y = 40  # Vertical spacing between pieces

    built = {}
    for piece_name in selected_pieces:
        piece = piece_functions[piece_name]()
        if not piece.is_volume:
            print(f"Warning: {piece_name} is not a watertight solid.")
        print(f"  {piece_name.capitalize()} height: {piece.extents[2]:.1f}mm")
        built[piece_name] = piece

    assembled_pieces = []
    for side, color, filament in sides:
        for piece_name, count in selected_pieces.items():
            for number in range(1, count + 1):
                label = piece_name.replace('_', ' ')
                name = f"{side} {label} {number}" if args.sides == 2 else f"{label.capitalize()} {number}"
                assembled_pieces.append(Part(name, built[piece_name].copy(),
                                             color if args.sides == 2 else None,
                                             filament if args.sides == 2 else None))

    total_pieces = len(assembled_pieces)
    if total_pieces == 0:
        print("No pieces to include based on the specified arguments. Exiting.")
        return

    # Calculate grid size (number of columns and rows)
    columns = math.ceil(math.sqrt(total_pieces))
    rows = math.ceil(total_pieces / columns)

    print(f"Arranging {total_pieces} pieces in a grid of {rows} rows and {columns} columns.")

    # Arrange pieces in a grid.  The pieces don't touch, so even in an STL the
    # slicer sees them as separate parts.
    for idx, part in enumerate(assembled_pieces):
        row = idx // columns
        col = idx % columns
        part.mesh.apply_translation([col * spacing_x, row * spacing_y, 0])

    # Determine output filename
    if len(selected_pieces) == 1:
        # Only one type of piece is included
        stem = f"chess_set_{next(iter(selected_pieces))}"
    else:
        # Multiple types of pieces are included
        stem = "chess_set"
    if args.sides == 2:
        stem += "_both_sides"
    output_filename = output_path(args.output, stem, args.format)

    save(assembled_pieces, output_filename)
    print(f"Chess set exported to '{output_filename}'.")
    if args.sides == 2 and not output_filename.lower().endswith('.3mf'):
        print("Note: STL files can't hold colors; use --format 3mf to get the two sides "
              "assigned to filaments 1 and 2.")


if __name__ == "__main__":
    main()
