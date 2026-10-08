"""Building blocks for the model generators.

Everything here returns trimesh.Trimesh solids in millimeters with Z up.
Booleans use the manifold3d engine, which is fast and always returns closed,
watertight meshes.
"""

import numpy as np
import trimesh
import trimesh.transformations as tt

BOOLEAN_ENGINE = 'manifold'
SECTIONS = 32         # Default facets around round parts
SPHERE_DETAIL = 3     # Default icosphere subdivisions


# --- Lofts ----------------------------------------------------------------

def rect_points(width, depth, z):
    """Corners of a width x depth rectangle centered on the Z axis, counterclockwise."""
    w, d = width / 2, depth / 2
    return [[-w, -d, z], [w, -d, z], [w, d, z], [-w, d, z]]


def _loft(outlines):
    """Closed solid through a stack of same-length outlines (lists of XYZ points,
    counterclockwise seen from above), bottom to top.  Outlines must be convex
    for the end caps to be correct."""
    n = len(outlines[0])
    vertices = [p for outline in outlines for p in outline]
    top = n * (len(outlines) - 1)

    faces = []
    for i in range(1, n - 1):
        faces.append([0, i + 1, i])                     # bottom cap (faces down)
        faces.append([top, top + i, top + i + 1])       # top cap (faces up)
    for r in range(len(outlines) - 1):
        lo, hi = n * r, n * (r + 1)
        for k in range(n):
            k1 = (k + 1) % n
            faces += [[lo + k, lo + k1, hi + k1], [lo + k, hi + k1, hi + k]]
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def loft_rects(rings):
    """Closed solid through a stack of (width, depth, z) rectangles, bottom to top.

    Two rings at the same z with different sizes make a flat ledge, so this
    builds stepped and chamfered shapes as a single mesh with no booleans.
    """
    return _loft([rect_points(w, d, z) for w, d, z in rings])


def loft_squares(rings):
    """Like loft_rects, for a stack of (side, z) squares."""
    return loft_rects([(side, side, z) for side, z in rings])


def lathe(rings, sections=SECTIONS):
    """Closed solid of revolution through a stack of (radius, z) rings, bottom to top.
    Radii must be greater than zero; use cone() for a point."""
    angles = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    circle = np.column_stack([np.cos(angles), np.sin(angles)])
    outlines = [np.column_stack([circle * r, np.full(sections, z)]).tolist() for r, z in rings]
    return _loft(outlines)


# --- Primitives -----------------------------------------------------------

def block(extents, center=(0, 0, 0)):
    """Box with the given XYZ extents, centered on center."""
    b = trimesh.creation.box(extents=extents)
    b.apply_translation(center)
    return b


def cylinder(radius, z_bottom, z_top, sections=SECTIONS):
    return lathe([(radius, z_bottom), (radius, z_top)], sections)


def frustum(radius_bottom, radius_top, z_bottom, z_top, sections=SECTIONS):
    return lathe([(radius_bottom, z_bottom), (radius_top, z_top)], sections)


def cone(radius, z_bottom, height, sections=SECTIONS):
    """Cone with its base at z_bottom and its point at z_bottom + height."""
    c = trimesh.creation.cone(radius=radius, height=height, sections=sections)
    c.apply_translation([0, 0, z_bottom])
    return c


def ellipsoid(radii, center, detail=SPHERE_DETAIL):
    e = trimesh.creation.icosphere(subdivisions=detail, radius=1)
    e.apply_scale(radii)
    e.apply_translation(center)
    return e


def sphere(radius, center, detail=SPHERE_DETAIL):
    return ellipsoid([radius] * 3, center, detail)


def prism_y(points_xz, depth):
    """Prism from a convex polygon in the XZ plane (counterclockwise seen from
    -Y), extruded depth mm and centered on y = 0."""
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


# --- Placement ------------------------------------------------------------

def rotate_z(mesh, degrees):
    mesh.apply_transform(tt.rotation_matrix(np.radians(degrees), [0, 0, 1]))
    return mesh


def ring_of(mesh, count, radius, phase_degrees=0.0):
    """Copies of mesh (built at the origin) spaced evenly around a circle,
    starting on the +Y axis."""
    copies = []
    for angle in np.linspace(0, 360, count, endpoint=False) + phase_degrees:
        c = mesh.copy()
        c.apply_translation([0, radius, 0])
        copies.append(rotate_z(c, angle))
    return copies


def drop_to_bed(mesh, scale=1.0):
    """Optionally scale the mesh, then move it so it sits on z = 0."""
    if scale != 1.0:
        mesh.apply_scale(scale)
    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])
    return mesh


# --- Booleans -------------------------------------------------------------

def union(parts):
    return trimesh.boolean.union(list(parts), engine=BOOLEAN_ENGINE)


def difference(solid, cutters):
    return trimesh.boolean.difference([solid] + list(cutters), engine=BOOLEAN_ENGINE)


def intersection(parts):
    return trimesh.boolean.intersection(list(parts), engine=BOOLEAN_ENGINE)
