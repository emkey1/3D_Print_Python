# Generates a four sided pyramid with stair step layers.
#
# Each layer is one square slab, a step narrower on every side than the one
# below it.  The whole pyramid is built as a single closed mesh, so it's small,
# watertight and slices cleanly.
#
# Requires: numpy, trimesh, manifold3d

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from printlib import Part, add_format_argument, output_path, save  # noqa: E402
from printlib.shapes import loft_squares  # noqa: E402


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
    parser.add_argument('--output', type=str, default=None, help='Output filename; .stl or .3mf (default: pyramid_blocks.stl or .3mf).')
    add_format_argument(parser)
    args = parser.parse_args()

    if min(args.base_size, args.step_width, args.step_height, args.top_size) <= 0:
        print("Error: All sizes must be positive.")
        sys.exit(1)
    if args.top_size > args.base_size:
        print("Error: --top_size can't be larger than --base_size.")
        sys.exit(1)

    pyramid, layers = create_stepped_pyramid(args.base_size, args.step_width,
                                             args.step_height, args.top_size)
    output_filename = output_path(args.output, 'pyramid_blocks', args.format)
    save(Part('Stepped pyramid', pyramid), output_filename)

    extents = pyramid.extents
    print(f"Stepped pyramid with {layers} layers saved as '{output_filename}'.")
    print(f"  Size: {extents[0]:.2f} x {extents[1]:.2f} x {extents[2]:.2f} mm")


if __name__ == "__main__":
    main()
