# Generate STL file for drawer organizer.  Can have one or more compartments. 
# Compartments are diagonal to maximize length in the center compartment since
# Many 3D printer beds are less than 11 inches (~ 256 mm) in size which limits
# How large a single piece organizer can be.

import numpy as np
import trimesh
import argparse
import math
import sys

def create_wall(start, end, thickness, height, floor_thickness, bite=False):
    start = np.array(start)
    end = np.array(end)
    wall_vector = end - start
    length = np.linalg.norm(wall_vector[:2])

    if length == 0:
        print(f"Warning: Wall with zero length between {start} and {end}. Skipping.")
        return None

    direction = wall_vector / length
    angle = np.arctan2(direction[1], direction[0])

    # Create the wall as a box centered at the origin
    wall = trimesh.creation.box(extents=[length, thickness, height + floor_thickness])
    wall.apply_translation([length / 2, 0, (height - floor_thickness) / 2])
    #wall.apply_translation([length / 2, 0, (height - floor_thickness) / 2 + floor_thickness])

    # Apply rotation and translation to position the wall
    T_rotate = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
    T_translate = trimesh.transformations.translation_matrix(start)
    T = trimesh.transformations.concatenate_matrices(T_translate, T_rotate)
    wall.apply_transform(T)

    if bite:
        # Create the ellipsoid for the bite
        bite_depth = .33 * height

        length = length + .5 

        # Radii for the ellipsoid
        radius_x = length * .35  #
        radius_y = thickness * 3.7  # Need to be sure we totally eliminate the partition wall where the ellipsoid is
        radius_z = bite_depth  # Depth of the bite

        # Create an ellipsoid centered at the origin
        subdivisions = 3  # Adjust for smoother ellipsoid
        ellipsoid = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1)
        ellipsoid.apply_scale([radius_x, radius_y, radius_z])

        # Position the ellipsoid to overlap with the top of the wall
        ellipsoid.apply_translation([length / 2, 0, height + 3 ])
        #ellipsoid.apply_translation([length / 2, 0, height - floor_thickness])

        # Apply the same transformation to the ellipsoid
        ellipsoid.apply_transform(T)

        # Ensure the meshes are valid volumes
        wall.process(validate=True)
        ellipsoid.process(validate=True)

        # Perform the boolean difference without specifying the engine
        wall = wall.difference(ellipsoid)

    return wall

def create_organizer(size, height, wall_thickness, divider_thickness, floor_thickness, compartments):
    half_size = size / 2

    # Create base as a thin box
    base = trimesh.creation.box(extents=[size, size, floor_thickness])
    base.apply_translation([0, 0, floor_thickness / 2])

    # Initialize list of components
    components = [base]

    # Create outer walls
    wall_positions = [
        ([-half_size, -half_size], [ half_size, -half_size]),  # Bottom wall
        ([ half_size, -half_size], [ half_size,  half_size]),  # Right wall
        ([ half_size,  half_size], [-half_size,  half_size]),  # Top wall
        ([-half_size,  half_size], [-half_size, -half_size]),  # Left wall
    ]

    for start_coords, end_coords in wall_positions:
        wall = create_wall(
            start=np.array(start_coords + [floor_thickness]),
            end=np.array(end_coords + [floor_thickness]),
            thickness=wall_thickness,
            height=height,
            floor_thickness=floor_thickness
        )

        if wall:
            components.append(wall)

    # Create dividers
    if compartments > 1:
        sqrt2 = math.sqrt(2)
        total_width = size * sqrt2
        compartment_width = total_width / compartments

        # Calculate distances from the center line
        distances = []
        for i in range(1, compartments):
            d = (i - compartments / 2) * compartment_width
            distances.append(d)

        # Define square boundaries for intersections
        min_coord = -half_size
        max_coord = half_size

        for d in distances:
            # Equation of walls parallel to y = x: y = x + c
            c = d * sqrt2

            # Find intersection points with box boundaries
            endpoints = []

            # Intersection with x = min_coord
            x = min_coord
            y = x + c
            if min_coord <= y <= max_coord:
                endpoints.append([x, y])

            # Intersection with x = max_coord
            x = max_coord
            y = x + c
            if min_coord <= y <= max_coord:
                endpoints.append([x, y])

            # Intersection with y = min_coord
            y = min_coord
            x = y - c
            if min_coord <= x <= max_coord:
                endpoints.append([x, y])

            # Intersection with y = max_coord
            y = max_coord
            x = y - c
            if min_coord <= x <= max_coord:
                endpoints.append([x, y])

            if len(endpoints) >= 2:
                # Take the first two valid intersection points
                start_point = endpoints[0]
                end_point = endpoints[1]

                # Check for zero-length walls
                if start_point == end_point:
                    print(f"Warning: Divider has zero length at {start_point}. Skipping.")
                    continue

                # Create the divider wall
                wall = create_wall(
                    start=np.array(start_point + [floor_thickness]),
                    end=np.array(end_point + [floor_thickness]),
                    thickness=divider_thickness,
                    height=height,
                    floor_thickness=floor_thickness,
                    bite=True  # Enable the oval bite for divider walls
                )
                if wall:
                    components.append(wall)

    # Combine all components using boolean union
    organizer = components[0]
    for component in components[1:]:
        organizer = organizer.union(component)

    return organizer

def main():
    parser = argparse.ArgumentParser(description='Generate a square kitchen drawer organizer with parallel diagonal compartments and an open top.')

    parser.add_argument('--compartments', type=int, default=3, help='Number of compartments to include (default: 3).')
    parser.add_argument('--size', type=float, default=250.0, help='Size of the organizer (width and depth) in millimeters (default: 250 mm).')
    parser.add_argument('--height', type=float, default=35.0, help='Height of the organizer side walls in millimeters (default: 35 mm).')
    parser.add_argument('--divider_thickness', type=float, default=1.75, help='Thickness of the dividers in millimeters (default: 1.75 mm).')
    parser.add_argument('--wall_thickness', type=float, default=1.75, help='Thickness of the side walls in millimeters (default: 1.75 mm).')
    parser.add_argument('--floor_thickness', type=float, default=1.25, help='Thickness of the floor in millimeters (default: 1.25 mm).')
    parser.add_argument('--output', type=str, default='drawer_organizer.stl', help='Output STL filename (default: drawer_organizer.stl).')

    args = parser.parse_args()

    if args.compartments < 1:
        print("Error: Number of compartments must be at least 1.")
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
        compartments=args.compartments
    )

    # Shift organizer so that the base sits on z=0
    min_z = organizer.bounds[0][2]
    if min_z != 0:
        organizer.apply_translation([0, 0, -min_z])

    # Determine output filename
    if args.output == 'drawer_organizer.stl':
        output_filename = f"drawer_organizer_{args.compartments}_compartments.stl"
    else:
        output_filename = args.output

    # Export the model to an STL file
    organizer.export(output_filename)
    print(f"Organizer exported to '{output_filename}'.")

    # Print out the arguments used
    print("\nOrganizer Parameters Used:")
    print(f"  Compartments: {args.compartments}")
    print(f"  Size: {args.size}mm x {args.size}mm")
    print(f"  Height: {args.height}mm")
    print(f"  Divider Thickness: {args.divider_thickness}mm")
    print(f"  Wall Thickness: {args.wall_thickness}mm")
    print(f"  Floor Thickness: {args.floor_thickness}mm")
    print(f"  Output File: {output_filename}")

if __name__ == "__main__":
    main()

