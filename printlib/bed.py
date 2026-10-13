"""Fitting pieces on the print bed."""

import math


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
