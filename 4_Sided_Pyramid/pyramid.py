# Generates a four sided pyramid with stair step layers.
#
# Each layer is one square slab, a step narrower on every side than the one
# below it.  The whole pyramid is built as a single closed mesh, so it's small,
# watertight and slices cleanly.
#
# Requires: numpy, trimesh

import argparse
import sys
import trimesh


def square_points(side, z):
    h = side / 2
    return [[-h, -h, z], [h, -h, z], [h, h, z], [-h, h, z]]


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


def create_stepped_pyramid(base_size, step_width, step_height, top_size):
    """Centered stepped pyramid sitting on z = 0.

    Each layer is step_height tall and step_width narrower on each side than the
    layer below.  Layers are added until the next one would be narrower than
    top_size.
    """
    rings = []
    side = base_size
    z = 0.0
    while side >= top_size:
        # Vertical side of this layer, then (if another layer follows) the
        # flat ledge stepping in to the next one.
        rings.append((side, z))
        z += step_height
        rings.append((side, z))
        side -= 2 * step_width
    return loft_squares(rings), len(rings) // 2


def main():
    parser = argparse.ArgumentParser(description='Generate a four sided stepped pyramid.')
    parser.add_argument('--base_size', type=float, default=120.0, help='Width of the bottom layer in millimeters (default: 120 mm).')
    parser.add_argument('--step_width', type=float, default=1.05, help='How far each layer steps in on every side, in millimeters (default: 1.05 mm).')
    parser.add_argument('--step_height', type=float, default=1.05, help='Height of each layer in millimeters (default: 1.05 mm).')
    parser.add_argument('--top_size', type=float, default=2.1, help='Smallest allowed width for the top layer in millimeters (default: 2.1 mm).')
    parser.add_argument('--output', type=str, default='pyramid_blocks.stl', help='Output STL filename (default: pyramid_blocks.stl).')
    args = parser.parse_args()

    if min(args.base_size, args.step_width, args.step_height, args.top_size) <= 0:
        print("Error: All sizes must be positive.")
        sys.exit(1)
    if args.top_size > args.base_size:
        print("Error: --top_size can't be larger than --base_size.")
        sys.exit(1)

    pyramid, layers = create_stepped_pyramid(args.base_size, args.step_width,
                                             args.step_height, args.top_size)
    pyramid.export(args.output)

    extents = pyramid.extents
    print(f"Stepped pyramid with {layers} layers saved as '{args.output}'.")
    print(f"  Size: {extents[0]:.2f} x {extents[1]:.2f} x {extents[2]:.2f} mm")


if __name__ == "__main__":
    main()
