# generate_chess_set.py

import numpy as np
import trimesh
from trimesh.creation import cylinder, box, cone
from trimesh.primitives import Sphere
import trimesh.transformations as tt  # Import transformations module
from scipy.spatial.transform import Rotation as R
import argparse  # Import argparse for command-line argument parsing
import math      # Import math for grid calculations

def pawn():
    base_height = 5
    body_height = 25
    head_radius = 6

    # Create base
    base = cylinder(radius=12, height=base_height)

    # Create body
    body = cylinder(radius=7.5, height=body_height)
    body.apply_translation([0, 0, base_height + 10])

    # Create head
    head = Sphere(radius=head_radius).to_mesh()

    # Center head at origin
    head_center_z = (head.bounds[1][2] + head.bounds[0][2]) / 2
    # Move head to correct position
    head.apply_translation([0, 0, base_height + body_height + 1])

    # Combine parts
    pawn = trimesh.util.concatenate([base, body, head])

    scale_factor = 4/5
    scaling_matrix = np.eye(4)  # Create a 4x4 identity matrix
    scaling_matrix[:3, :3] *= scale_factor  # Apply scaling to the top-left 3x3 part
    pawn.apply_transform(scaling_matrix)  # Apply the scaling matrix to the pawn

    # Shift pawn so that min z is 0
    min_z = pawn.bounds[0][2]
    if min_z != 0:
        pawn.apply_translation([0, 0, -min_z])

    return pawn

def rook():
    import trimesh
    import numpy as np
    from trimesh.creation import cylinder, cone
    from trimesh.repair import fill_holes, fix_normals
    from trimesh.transformations import rotation_matrix

    # Dimensions
    base_height = 5
    base_radius = 12

    top_height = 6
    top_outer_radius = 11
    top_inner_radius = 9

    notch_height = 3.3
    notch_depth = 3
    notch_width = 2

    body_height = 31
    body_radius = 7.5

    frustum_height = 15
    frustum_lower_radius = top_outer_radius  # 11
    frustum_upper_radius = body_radius       # 7.5

    base_z0 = 0
    body_z0 = base_height
    frustum_z0 = body_z0 + body_height
    #top_z0 = frustum_z0 + frustum_height
    top_z0 = frustum_z0 - .4

    # Create the base
    base = cylinder(radius=base_radius, height=base_height)
    base.apply_translation([0, 0, base_z0])

    # Create the body
    body = cylinder(radius=body_radius, height=body_height)
    body.apply_translation([0, 0, body_z0 * 3.6])

    # --- Create Top ---
    top = cylinder(radius=top_outer_radius, height=top_height)
    top.apply_translation([0, 0, top_z0])

    # Hollow out the top
    inner_cylinder = cylinder(radius=top_inner_radius, height=top_height + 0.1)
    inner_cylinder.apply_translation([0, 0, top_z0])

    top = top.difference(inner_cylinder)
    # Create a cone that tapers from frustum_upper_radius to a point
    frustum = cone(
        radius=frustum_upper_radius,  # body_radius = 7.5
        height=frustum_height,
        sections=64
    )
    # Invert the cone to expand upward
    frustum.vertices[:, 2] *= -1
    # Scale the cone to create a frustum with specified lower and upper radii
    scale_factor = frustum_lower_radius / frustum_upper_radius  # 11 / 7.5
    frustum.apply_scale([scale_factor, scale_factor, 1])
    # Correct translation of frustum
    frustum.apply_translation([0, 0, frustum_z0 - 2.1])

    # Cap the bottom of the frustum using a thin cylinder
    cap_thickness = 0.01  # Very thin
    cap = cylinder(radius=frustum_lower_radius, height=cap_thickness, sections=64)
    # Position the cap at the bottom of the frustum
    cap.apply_translation([0, 0, frustum_z0 - 2.1 - cap_thickness])

    # Combine the cap with the frustum
    frustum = trimesh.util.concatenate([frustum, cap])

    # --- Repair the Frustum Mesh ---
    # Ensure the frustum is watertight and has correct normals
    frustum.fix_normals()
    frustum.fill_holes()
    frustum.update_faces(frustum.unique_faces())
    frustum.remove_unreferenced_vertices()

    # Check if the frustum is a valid volume
    if not frustum.is_volume:
        print("Frustum is not a valid volume after repair.")
        # Attempt to create a volume by taking the convex hull
        frustum = frustum.convex_hull

    # Hollow out the top part of the frustum
    inner_cylinder = cylinder(radius=top_inner_radius, height=top_height + 0.1)
    # Position the inner cylinder at the top of the frustum
    inner_cylinder.apply_translation([
        0,
        0,
        frustum_z0 - 4.1 + frustum_height - top_height + 0.05
    ])

    # Ensure the inner cylinder is watertight
    inner_cylinder.fix_normals()
    inner_cylinder.fill_holes()

    # Perform the boolean difference
    frustum = frustum.difference(inner_cylinder)

    # --- Create Notches ---
    notch = box(extents=[notch_width, notch_depth, notch_height])
    notch_radius = top_outer_radius - notch_depth / 2
    # Calculate the vertical center of the top
    notch_z = top_z0 + top_height / 2

    num_notches = 6
    angles = np.linspace(0, 2 * np.pi, num_notches, endpoint=False)
    notches = []
    for angle in angles:
        notch_copy = notch.copy()
        # Position the notch at the correct radial distance and align its bottom with notch_z
        notch_copy.apply_translation([0, notch_radius, notch_z - notch_height / 2])
        # Rotate the notch around the z-axis by the current angle
        rotation = rotation_matrix(angle, [0, 0, 1])
        notch_copy.apply_transform(rotation)
        notches.append(notch_copy)

    # Combine all notches into a single mesh
    all_notches = trimesh.util.concatenate(notches)

    # Subtract notches from rook_body
    rook_body = trimesh.util.concatenate([base, body, frustum, top])
    rook = rook_body.difference(all_notches)

    min_z = rook.bounds[0][2]
    if min_z != 0:
        rook.apply_translation([0, 0, -min_z])

    print(f"Rook Body is watertight: {rook_body.is_watertight}")
    print(f"All Notches are watertight: {all_notches.is_watertight}")

    return rook

def rook_alt():
    base_height = 5
    body_height = 33
    top_height = 6
    frustum_height = 15

    # Create base
    base = trimesh.creation.cylinder(radius=12, height=base_height)
    base.apply_translation([0,0,0])

    # Create body
    body = trimesh.creation.cylinder(radius=7.5, height=body_height)
    body.apply_translation([0, 0, base_height + 11])

    # Create frustum (tapered ring)
    frustum_lower_radius = 11  # Matches the body's radius
    frustum_upper_radius = 21   # Matches the top's radius
    frustum = cone(radius=frustum_upper_radius, height=frustum_height, sections=32)
    frustum.vertices[:, 2] *= -1  # Invert the cone to create a frustum expanding upward
    frustum.apply_scale([frustum_lower_radius / frustum_upper_radius, frustum_lower_radius / frustum_upper_radius, 1])
    frustum.apply_translation([0, 0, base_height + body_height - 5 ])
    
    # Create top
    top = trimesh.creation.cylinder(radius=11, height=top_height)
    top.apply_translation([0, 0, base_height + body_height -2 ])

    # Create notches
    notch_height = 5
    notch_depth = 5
    notch_width = 2
    notch = trimesh.creation.box(extents=[notch_width, notch_depth, notch_height])

    # Calculate notch position parameters
    notch_radius = 11  # radius of the top cylinder
    notch_offset = notch_radius - notch_depth / 2  # position along y-axis before rotation
    notch_z = base_height + body_height + (top_height - notch_height) / 2  # center z of the notch

    # Create a list to hold all notches
    notches = []
    num_notches = 6  # number of notches
    angles = np.linspace(0, 2 * np.pi, num_notches, endpoint=False)

    for angle in angles:
        notch_copy = notch.copy()
        # Position the notch at the correct height and radius
        notch_copy.apply_translation([0, notch_offset, notch_z - notch_height / 2])
        # Rotate the notch around the z-axis
        rotation = R.from_rotvec(angle * np.array([0, 0, 1]))
        transform = np.eye(4)
        transform[:3, :3] = rotation.as_matrix()
        notch_copy.apply_transform(transform)
        notches.append(notch_copy)

    # Combine all notches into a single mesh
    all_notches = trimesh.util.concatenate(notches)

    # Combine parts
    rook_body = trimesh.util.concatenate([base, body, frustum, top])
#    rook_body = trimesh.util.concatenate([base, body, top])

    # Subtract the notches from the rook body
    rook = rook_body.difference(all_notches)

    # Shift rook so that min z is 0
    min_z = rook.bounds[0][2]
    if min_z != 0:
        rook.apply_translation([0, 0, -min_z])

    return rook

def knight():
    base_height = 5
    base = cylinder(radius=12, height=base_height)

    # Create body
    body = Sphere(radius=9).to_mesh()
    body.apply_scale([1, 0.5, 2])
    body.apply_translation([0, 0, base_height + 11])

    # Create head
    head = Sphere(radius=5).to_mesh()
    head.apply_scale([1, 0.5, 1.5])
    head.apply_translation([0, 0, base_height + 30])

    # Combine parts
    knight = trimesh.util.concatenate([base, body, head])

    # Shift knight so that min z is 0
    min_z = knight.bounds[0][2]
    if min_z != 0:
        knight.apply_translation([0, 0, -min_z])

    return knight

def bishop():
    base_height = 5
    body_height = 40
    head_radius = 6

    # Create base
    base = cylinder(radius=12, height=base_height)

    # Create body
    body = cylinder(radius=8, height=body_height)
    body.apply_translation([0, 0, base_height * 4 ])

    # Create head
    head = Sphere(radius=head_radius).to_mesh()
    # Position head just above the body
    head.apply_translation([0, 0, base_height + body_height - head_radius + 1.5])

    # Create the slot in the head
    slot = box(extents=[2, 12, head_radius * 2])
    # Position the slot slightly inside the head for a clean cut
    slot.apply_translation([0, 0, base_height + body_height - head_radius + 1.5])

    # Subtract slot from head to create the bishop's cut
    head = head.difference(slot)

    # Combine base, body, and head into one bishop mesh
    bishop = trimesh.util.concatenate([base, body, head])

    # Shift bishop so that min z is 0
    min_z = bishop.bounds[0][2]
    if min_z != 0:
        bishop.apply_translation([0, 0, -min_z])

    return bishop


def bishop2():
    base_height = 5
    body_height = 40
    head_radius = 6

    # Create base
    base = cylinder(radius=12, height=base_height)

    # Create body
    body = cylinder(radius=8, height=body_height)
    body.apply_translation([0, 0, base_height + 10])

    # Create head
    head = Sphere(radius=head_radius).to_mesh()
    #head.apply_translation([0, 0, base_height + body_height + 5])
    head.apply_translation([0, 0, base_height + body_height - (head_radius*2 )])

    # Create the slot in the head
    slot = box(extents=[2, 12, 20])
    slot.apply_translation([0, 0, base_height + body_height + 5])

    # Subtract slot from head
    head = head.difference(slot)

    # Combine parts
    bishop = trimesh.util.concatenate([base, body, head])

    # Shift bishop so that min z is 0
    min_z = bishop.bounds[0][2]
    if min_z != 0:
        bishop.apply_translation([0, 0, -min_z])

    return bishop

def queen():
    import trimesh
    import numpy as np
    from trimesh.creation import cylinder, cone, box
    from trimesh.repair import fill_holes, fix_normals
    from trimesh.transformations import rotation_matrix
    from trimesh.primitives import Sphere  # Ensure Sphere is imported

    # Dimensions
    base_height = 5
    body_height = 46  # Changed from 45 to 35
    crown_height = 15.5
    sphere_radius = 1.5

    # Create base
    base = cylinder(radius=12, height=base_height)
    base.apply_translation([0, 0, 0])

    # Create body
    body = cylinder(radius=8.5, height=body_height)
    body.apply_translation([0, 0, 23])  # Changed translation to align with base

    # --- Create Top ---
    crown = cone(radius=5, height=crown_height, sections=32)
    crown.apply_translation([0, 0, base_height + 37])  # [0, 0, 39]

    # Spheres on crown
    crown_top_z = base_height + body_height + 1  # 5 + 35 + 1 = 41
    spheres = []
    for angle in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        s = Sphere(radius=sphere_radius).to_mesh()
        s.apply_translation([
            8 * np.cos(angle),
            8 * np.sin(angle),
            46.0  # Positioned near the crown
        ])
        spheres.append(s)

    spheres_mesh = trimesh.util.concatenate(spheres)

    # Add a decorative collar between base and body
    collar = cylinder(radius=10, height=2)
    collar.apply_translation([0, 0, 3])  # Changed translation to align with base and body

    # Combine parts
    queen = trimesh.util.concatenate([base, collar, body, crown, spheres_mesh])

    scale_factor = 1.25
    scaling_matrix = np.eye(4)  # Create a 4x4 identity matrix
    scaling_matrix[:3, :3] *= scale_factor  # Apply scaling to the top-left 3x3 part
    queen.apply_transform(scaling_matrix)  # Apply the scaling matrix to the queen

    # Shift queen so that min z is 0
    min_z = queen.bounds[0][2]
    if min_z != 0:
        queen.apply_translation([0, 0, -min_z])

    return queen

def king():
    base_height = 5
    body_height = 48
    cross_height = 4.5

    # Create base
    base = cylinder(radius=12, height=base_height)

    # Create body
    body = cylinder(radius=9, height=body_height)
    #body.apply_translation([0, 0, base_height + 17.2])
    body.apply_translation([0, 0, base_height + 17.2])

    # Add a decorative collar between base and body
    collar = cylinder(radius=10, height=2)
    collar.apply_translation([0, 0, base_height - 1.9])

    # Create a simple crown (small cylinder)
    crown_height = 3
    crown = cylinder(radius=6, height = crown_height)
    crown.apply_translation([0, 0, body_height - crown_height + 1.4])

    # Add a sphere on top of the crown
    crown_radius = 5
    crown_top = Sphere(radius=crown_radius).to_mesh()
    crown_top.apply_translation([0, 0, base_height + body_height + .5 - (crown_height + crown_radius) + 3])

    # Combine parts
#    king = trimesh.util.concatenate([base, base_band, collar, body, crown, cross])
    king = trimesh.util.concatenate([base, collar, body, crown, crown_top])
    # Shift king so that min z is 0

    scale_factor = 1.25
    scaling_matrix = np.eye(4)  # Create a 4x4 identity matrix
    scaling_matrix[:3, :3] *= scale_factor  # Apply scaling to the top-left 3x3 part
    king.apply_transform(scaling_matrix)  # Apply the scaling matrix to the king

    min_z = king.bounds[0][2]
    if min_z != 0:
        king.apply_translation([0, 0, -min_z])

    return king

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
        'pawn': 1,
        'rook': 0,
        'rook_alt': 1,
        'knight': 1,
        'bishop': 1,
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
        selected_pieces = default_counts.copy()

    # Display selected pieces and their counts
    for piece, count in selected_pieces.items():
        print(f"  {piece.capitalize()}: {count}")

    # Build the list of pieces based on the selected counts
    assembled_pieces = []
    spacing_x = 40  # Horizontal spacing between pieces
    spacing_y = 40  # Vertical spacing between pieces

    # Collect all pieces first without translation
    for piece_name, count in selected_pieces.items():
        if count <= 0:
            continue  # Skip pieces with non-positive counts
        for _ in range(count):
            piece = piece_functions[piece_name]()  # Create the piece
            assembled_pieces.append((piece_name.capitalize(), piece))  # Store with name for logging

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
        translation_x = col * spacing_x
        translation_y = row * spacing_y
        piece.apply_translation([translation_x, translation_y, 0])
        chess_set_meshes.append(piece)
        # Print min z
        min_z = piece.bounds[0][2]
        print(f"{piece_name} min z: {min_z}")

    # Combine all pieces into one mesh
    chess_set = trimesh.util.concatenate(chess_set_meshes)

    # Shift chess set so that min z is 0
    min_z = chess_set.bounds[0][2]
    if min_z != 0:
        chess_set.apply_translation([0, 0, -min_z])
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

