"""Saving models as STL or 3MF.

A model is a list of Part objects.  STL can only hold one unnamed mesh, so all
parts are merged into it.  3MF keeps each part as its own named object, with
an optional display color and filament slot, so a slicer opens it with the
parts already separated and (in Bambu Studio / OrcaSlicer) already assigned
to filaments.

The 3MF writer only needs the Python standard library.
"""

import os
import zipfile
from dataclasses import dataclass
from xml.sax.saxutils import quoteattr

import numpy as np
import trimesh

FORMATS = ('stl', '3mf')


@dataclass
class Part:
    name: str
    mesh: trimesh.Trimesh
    color: str = None      # '#RRGGBB'
    filament: int = None   # 1-based filament / AMS slot number


def add_format_argument(parser, default='stl'):
    parser.add_argument('--format', choices=FORMATS, default=default,
                        help=f'Output file format when --output is not given (default: {default}). '
                             '3mf keeps parts as separate, named objects with filament assignments.')


def output_path(output, default_stem, file_format):
    """The output filename: output if given (its extension wins), otherwise
    default_stem with the extension for file_format."""
    if output:
        return output
    return f'{default_stem}.{file_format}'


def save(parts, path):
    """Save a Part, a mesh, or a list of either.  The format comes from the
    file extension."""
    if not isinstance(parts, (list, tuple)):
        parts = [parts]
    parts = [p if isinstance(p, Part) else Part(f'Part {i + 1}', p) for i, p in enumerate(parts)]

    extension = os.path.splitext(path)[1].lower()
    if extension == '.stl':
        trimesh.util.concatenate([p.mesh for p in parts]).export(path)
    elif extension == '.3mf':
        write_3mf(parts, path)
    else:
        raise ValueError(f"Don't know how to save '{path}'; use a .stl or .3mf extension.")


# --- 3MF ------------------------------------------------------------------

_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
 <Default Extension="config" ContentType="text/xml"/>
</Types>
"""

_RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>
"""


def _mesh_xml(mesh):
    vertices = '\n'.join(f'<vertex x="{x:.5f}" y="{y:.5f}" z="{z:.5f}"/>' for x, y, z in mesh.vertices)
    triangles = '\n'.join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in mesh.faces)
    return f'<mesh>\n<vertices>\n{vertices}\n</vertices>\n<triangles>\n{triangles}\n</triangles>\n</mesh>'


def write_3mf(parts, path):
    colors = []
    for p in parts:
        if p.color and p.color not in colors:
            colors.append(p.color)

    resources = []
    if colors:
        bases = ''.join(f'<base name={quoteattr(c)} displaycolor="{c.upper()}FF"/>' for c in colors)
        resources.append(f'<basematerials id="1">{bases}</basematerials>')

    items = []
    settings = []
    for index, part in enumerate(parts):
        object_id = index + 2   # id 1 is the material list
        material = (f' pid="1" pindex="{colors.index(part.color)}"' if part.color else '')
        resources.append(f'<object id="{object_id}" type="model" name={quoteattr(part.name)}{material}>\n'
                         f'{_mesh_xml(part.mesh)}\n</object>')
        items.append(f'<item objectid="{object_id}"/>')
        settings.append(_object_settings(object_id, part))

    model = ('<?xml version="1.0" encoding="UTF-8"?>\n'
             '<model unit="millimeter" xml:lang="en-US" '
             'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">\n'
             '<resources>\n' + '\n'.join(resources) + '\n</resources>\n'
             '<build>\n' + '\n'.join(items) + '\n</build>\n</model>\n')

    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', _CONTENT_TYPES)
        z.writestr('_rels/.rels', _RELS)
        z.writestr('3D/3dmodel.model', model)
        z.writestr('Metadata/model_settings.config',
                   '<?xml version="1.0" encoding="UTF-8"?>\n<config>\n' + '\n'.join(settings) + '\n</config>\n')


def _object_settings(object_id, part):
    """Per-object settings read by Bambu Studio and OrcaSlicer: the object name
    and which filament it prints with."""
    lines = [f'<object id="{object_id}">',
             f'  <metadata key="name" value={quoteattr(part.name)}/>']
    if part.filament:
        lines.append(f'  <metadata key="extruder" value="{part.filament}"/>')
    lines.append('</object>')
    return '\n'.join(lines)
