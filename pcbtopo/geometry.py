"""Board geometries.

Conventions shared by every geometry:

* the region where copper may be placed is the material ``"design"``;
* each pad (an equipotential contact: a connector or a component lead) is a
  separate material ``<name>`` whose boundary is named ``"pad_<name>"``;
* regions with a prescribed density use other material names, e.g.
  ``"keepout"`` (no copper allowed).

The plate with two electrodes and the resistor board reproduce the test cases
of the original study; the units are arbitrary (board side = 2).
"""
from __future__ import annotations

from netgen.geom2d import CSG2d, Rectangle
from ngsolve import Mesh


def _pad(name, pmin, pmax):
    return Rectangle(pmin=pmin, pmax=pmax, bc=f"pad_{name}").Mat(name)


def electrode_plate(keepout: tuple | None = None, maxh: float = 0.05) -> Mesh:
    """Square plate [-1, 1]^2 between two full-width electrodes ``top`` and ``bottom``
    (strips of height 0.1). Optionally, a rectangular keep-out zone
    ``((xmin, ymin), (xmax, ymax))`` where no copper may be placed.
    """
    geo = CSG2d()
    plate = Rectangle(pmin=(-1, -1), pmax=(1, 1))
    top = _pad("top", (-1, 1), (1, 1.1))
    bottom = _pad("bottom", (-1, -1.1), (1, -1))
    if keepout is not None:
        hole = Rectangle(pmin=keepout[0], pmax=keepout[1]).Mat("keepout")
        geo.Add((plate - hole).Mat("design"))
        geo.Add(hole)
    else:
        geo.Add(plate.Mat("design"))
    geo.Add(top)
    geo.Add(bottom)
    return Mesh(geo.GenerateMesh(maxh=maxh))


RESISTOR_PADS = {
    "P": ((-0.8, -1.0), (-0.5, -0.8)),        # supply connector, bottom left
    "N": ((0.5, 0.8), (0.8, 1.0)),            # return connector, top right
    "Rp": ((-0.2, -0.1), (-0.1, 0.1)),        # resistor lead
    "Rn": ((0.1, -0.1), (0.2, 0.1)),          # resistor lead
}


def resistor_board(maxh: float = 0.04, pads: dict | None = None) -> Mesh:
    """Square board [-1, 1]^2 with a supply connector P, a return connector N and
    the two leads Rp and Rn of a surface-mounted resistor in the middle."""
    pads = pads or RESISTOR_PADS
    geo = CSG2d()
    board = Rectangle(pmin=(-1, -1), pmax=(1, 1))
    shapes = [_pad(name, *box) for name, box in pads.items()]
    design = board
    for s in shapes:
        design = design - s
    geo.Add(design.Mat("design"))
    for s in shapes:
        geo.Add(s)
    return Mesh(geo.GenerateMesh(maxh=maxh))


def two_plate_board(thickness: float = 0.1, copper: float = 0.005, maxh: float = 0.1):
    """3D board: a dielectric core of size 1 x 1 x ``thickness`` between two copper
    planes ``plus`` (top) and ``minus`` (bottom). Outer faces of the planes are
    named ``top`` and ``bottom``; every other outer face is ``outer``."""
    from netgen.csg import CSGeometry, OrthoBrick, Pnt

    core = OrthoBrick(Pnt(0, 0, 0), Pnt(1, 1, thickness)).bc("outer")
    plus = OrthoBrick(Pnt(0, 0, thickness), Pnt(1, 1, thickness + copper)).bc("outer")
    minus = OrthoBrick(Pnt(0, 0, -copper), Pnt(1, 1, 0)).bc("outer")
    geo = CSGeometry()
    geo.Add(core.mat("core"))
    geo.Add(plus.mat("plus"))
    geo.Add(minus.mat("minus"))
    mesh = Mesh(geo.GenerateMesh(maxh=maxh))
    _name_plate_faces(mesh, thickness + copper, -copper)
    return Mesh(mesh.ngmesh)


def _name_plate_faces(mesh, z_top, z_bottom, tol=1e-9):
    """Rename the boundary faces lying in the planes z = z_top / z = z_bottom."""
    ng = mesh.ngmesh
    pts = ng.Points()
    for el in ng.Elements2D():
        zs = [pts[v][2] for v in el.vertices]
        if all(abs(z - z_top) < tol for z in zs):
            ng.SetBCName(el.index - 1, "top")
        elif all(abs(z - z_bottom) < tol for z in zs):
            ng.SetBCName(el.index - 1, "bottom")
