# generate_chess_set.py
#
# Generates STL files for a simple chess set.  Each piece is built from
# primitive shapes that are merged with real boolean unions, so every piece is
# a single watertight solid.  Heights are measured from the bottom of the base
# and parts are positioned from each other's dimensions, so changing one size
# doesn't leave a gap or a floating part somewhere else.
#
# Requires: numpy, trimesh, manifold3d

import numpy as np
import trimesh
import trimesh.transformations as tt
import argparse  # Import argparse for command-line argument parsing
import math      # Import math for grid calculations

BOOLEAN_ENGINE = 'manifold'
SECTIONS = 32         # Facets around round parts
SPHERE_DETAIL = 3     # Icosphere subdivisions

BASE_RADIUS = 12
BASE_HEIGHT = 5


# --- Shape helpers --------------------------------------------------------

def lathe(rings, sections=SECTIONS):
    """Closed solid of revolution through a stack of (radius, z) rings, bottom to top."""
    angles = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    circle = np.column_stack([np.cos(angles), np.sin(angles)])
    vertices = [np.column_stack([circle * r, np.full(sections, z)]) for r, z in rings]
    vertices = np.vstack(vertices + [[[0, 0, rings[0][1]], [0, 0, rings[-1][1]]]])
    bottom_center, top_center = len(vertices) - 2, len(vertices) - 1
    top = sections * (len(rings) - 1)

    faces = []
    for k in range(sections):
        k1 = (k + 1) % sections
        faces.append([bottom_center, k1, k])
        faces.append([top_center, top + k, top + k1])
        for r in range(len(rings) - 1):
            lo, hi = sections * r, sections * (r + 1)
            faces += [[lo + k, lo + k1, hi + k1], [lo + k, hi + k1, hi + k]]
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def cylinder(radius, z_bottom, z_top):
    return lathe([(radius, z_bottom), (radius, z_top)])


def frustum(radius_bottom, radius_top, z_bottom, z_top):
    return lathe([(radius_bottom, z_bottom), (radius_top, z_top)])


def cone(radius, z_bottom, height):
    """Cone with its base at z_bottom and its point at z_bottom + height."""
    c = trimesh.creation.cone(radius=radius, height=height, sections=SECTIONS)
    c.apply_translation([0, 0, z_bottom])
    return c


def ellipsoid(radii, center):
    e = trimesh.creation.icosphere(subdivisions=SPHERE_DETAIL, radius=1)
    e.apply_scale(radii)
    e.apply_translation(center)
    return e


def sphere(radius, center):
    return ellipsoid([radius] * 3, center)


def block(extents, center):
    b = trimesh.creation.box(extents=extents)
    b.apply_translation(center)
    return b


def prism_y(points_xz, depth):
    """Prism from a convex polygon in the XZ plane (counterclockwise seen from
    -Y), extruded depth mm, centered on y = 0."""
    n = len(points_xz)
    front = [[x, -depth / 2, z] for x, z in points_xz]
    back = [[x, depth / 2, z] for x, z in points_xz]
    faces = []
    for i in range(1, n - 1):
        faces.append([0, i, i + 1])                 # front (-Y) face
        faces.append([n, n + i + 1, n + i])         # back (+Y) face
    for i in range(n):
        i1 = (i + 1) % n
        faces += [[i, n + i, n + i1], [i, n + i1, i1]]
    return trimesh.Trimesh(vertices=front + back, faces=faces)


def ring_of(mesh, count, radius, phase=0.0):
    """Copies of mesh (built at the origin) spaced evenly around a circle."""
    copies = []
    for angle in np.linspace(0, 2 * np.pi, count, endpoint=False) + phase:
        c = mesh.copy()
        c.apply_translation([0, radius, 0])
        c.apply_transform(tt.rotation_matrix(angle, [0, 0, 1]))
        copies.append(c)
    return copies


def union(parts):
    return trimesh.boolean.union(parts, engine=BOOLEAN_ENGINE)


def difference(solid, cutters):
    return trimesh.boolean.difference([solid] + list(cutters), engine=BOOLEAN_ENGINE)


def finish(piece, scale=1.0):
    """Scale the piece and set it on z = 0."""
    if scale != 1.0:
        piece.apply_scale(scale)
    piece.apply_translation([0, 0, -piece.bounds[0][2]])
    return piece


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
    return finish(pawn, scale=4 / 5)


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

    return finish(difference(rook, [cup] + notches))


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

    return finish(difference(rook, notches))


def knight():
    knight = union([
        base(),
        ellipsoid([9, 4.5, 18], [0, 0, 18.5]),    # Body
        ellipsoid([5, 2.5, 7.5], [0, 0, 37.5]),   # Head
    ])
    return finish(knight)


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

    return finish(difference(bishop, [slot]))


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
    return finish(queen, scale=1.25)


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
    return finish(king, scale=1.25)


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
    assembled_pieces = []
    spacing_x = 40  # Horizontal spacing between pieces
    spacing_y = 40  # Vertical spacing between pieces

    for piece_name, count in selected_pieces.items():
        piece = piece_functions[piece_name]()
        if not piece.is_volume:
            print(f"Warning: {piece_name} is not a watertight solid.")
        print(f"  {piece_name.capitalize()} height: {piece.extents[2]:.1f}mm")
        for _ in range(count):
            assembled_pieces.append((piece_name.capitalize(), piece.copy()))

    total_pieces = len(assembled_pieces)
    if total_pieces == 0:
        print("No pieces to include based on the specified arguments. Exiting.")
        return

    # Calculate grid size (number of columns and rows)
    columns = math.ceil(math.sqrt(total_pieces))
    rows = math.ceil(total_pieces / columns)

    print(f"Arranging {total_pieces} pieces in a grid of {rows} rows and {columns} columns.")

    # Arrange pieces in a grid
    chess_set_meshes = []
    for idx, (piece_name, piece) in enumerate(assembled_pieces):
        row = idx // columns
        col = idx % columns
        piece.apply_translation([col * spacing_x, row * spacing_y, 0])
        chess_set_meshes.append(piece)

    # Combine all pieces into one file.  The pieces don't touch, so the slicer
    # sees them as separate parts.
    chess_set = trimesh.util.concatenate(chess_set_meshes)

    # Determine output filename
    unique_piece_types = set([name for name, _ in assembled_pieces])
    if len(unique_piece_types) == 1:
        # Only one type of piece is included
        single_piece = unique_piece_types.pop().lower()
        output_filename = f"chess_set_{single_piece}.stl"
    else:
        # Multiple types of pieces are included
        output_filename = "chess_set.stl"

    # Export the model to an STL file
    chess_set.export(output_filename)
    print(f"Chess set exported to '{output_filename}'.")


if __name__ == "__main__":
    main()
