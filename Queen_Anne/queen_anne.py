# Generates a multi-color waterline model of Cunard's cruise ship MS Queen Anne.
#
# This is a simplified, recognisable model, not a scale replica.  The layout
# comes from Cunard's published deck plans and their side profile of the ship
# (which deck is how high, how long each deck is, where the funnel, the
# Pavilion, the deckhouses, the lifeboats and the open decks are), her overall
# dimensions (322.5 m long, 35.6 m beam), and photographs for the look: dark
# hull with a red waterline stripe, white superstructure with dark glazed
# bands, orange lifeboats, light brown decks, and the red Cunard funnel with
# its black top and bands.
#
# The model is one object made of parts, one or more per color, so a 3MF opens
# in the slicer with each part already assigned to a filament:
#   1 white   superstructure, mast, domes
#   2 black   hull, windows, funnel top and bands
#   3 red     waterline stripe, funnel
#   4 orange  lifeboats
#   5 light brown  decking on the open decks
#   6 blue    pools
# Most layers use only one or two of them.
#
# All of the ship's geometry below is in meters, measured from the stern (x),
# the centerline (y) and the waterline (z).  It is scaled to millimeters at the
# end.
#
# Requires: numpy, trimesh, manifold3d (pip install -r requirements.txt, from the repository root)

import argparse
import math
import sys
from pathlib import Path

import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from printlib import Part, add_format_argument, output_path, save  # noqa: E402
from printlib import shapes  # noqa: E402
from printlib.bed import longest_fit, print_rotation  # noqa: E402
from printlib.units import dimensions_type, format_mm  # noqa: E402

LENGTH = 322.5            # Overall length
BEAM = 35.6               # Width
HALF_BEAM = BEAM / 2
BED_MARGIN = 3.0          # Millimeters to keep clear at the edges of the bed

# Heights above the waterline
BOOT_TOP = 1.2            # Top of the red waterline stripe
DECK_1 = 5.9              # Floor of deck 1
DECK_HEIGHT = 2.84
HULL_TOP = DECK_1 + 2 * DECK_HEIGHT     # Top of the dark hull: the floor of deck 3, the promenade deck
RIM_HEIGHT = 0.7          # Each deck's floor edge, which stands proud of the balconies below it
RIB_DEPTH = 0.5           # How far the balcony fronts sit back from the deck edges
FORECASTLE_LENGTH = 42.0  # Raised dark hull at the bow, up to the observation deck on deck 5
FORECASTLE_HEIGHT = 2 * DECK_HEIGHT
STERN_LENGTH = 27.0       # ... and at the stern
STERN_HEIGHT = 4.2
PLANK = 0.5               # Thickness of the decking laid into the open decks
DECK_MARGIN = 0.9         # Width of the edge left around the decking
LETTER_HEIGHT = 3.0       # True height of the name on the hull
MIN_LETTER_MM = 4.2       # ... but never printed smaller than this, or the strokes are too fine to print

# The company name on the hull, as a 5 x 7 block font.
FONT = {
    'C': (" ### ", "#   #", "#    ", "#    ", "#    ", "#   #", " ### "),
    'U': ("#   #", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "),
    'N': ("#   #", "##  #", "# # #", "#  ##", "#   #", "#   #", "#   #"),
    'A': (" ### ", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"),
    'R': ("#### ", "#   #", "#   #", "#### ", "# #  ", "#  # ", "#   #"),
    'D': ("#### ", "#   #", "#   #", "#   #", "#   #", "#   #", "#### "),
}

WHITE, BLACK, RED, ORANGE, TAN, BLUE = 'white', 'black', 'red', 'orange', 'light brown', 'blue'
FILAMENTS = {WHITE: (1, '#F5F5F5'), BLACK: (2, '#1E1E1E'), RED: (3, '#C8102E'), ORANGE: (4, '#F26A1B'),
             TAN: (5, '#C8A46E'), BLUE: (6, '#2F86D6')}


def deck_z(deck):
    """Height of a deck's floor above the waterline.  Queen Anne has no deck 13:
    deck 14 sits on deck 12, and deck_z(15) is the roof of deck 14."""
    return DECK_1 + ((deck if deck < 13 else deck - 1) - 1) * DECK_HEIGHT


def hull_half_breadth(x, x_stern, x_bow, run, transom, entrance, sharpness, fullness):
    """Half the hull's width at x.  The stern widens from a flat transom over
    `run` meters, the middle is full width, and the bow narrows to a point from
    `entrance` forward."""
    if x <= x_stern + run:
        return HALF_BEAM * (transom + (1 - transom) * math.sin(math.pi / 2 * (x - x_stern) / run))
    if x >= entrance:
        v = min(1.0, (x - entrance) / (x_bow - entrance))
        return HALF_BEAM * max(0.0, 1 - v ** sharpness) ** fullness
    return HALF_BEAM


def deck_edge(x):
    """Half the hull's width at promenade deck level."""
    return hull_half_breadth(x, 0.0, LENGTH, 30.0, 0.72, LENGTH - 76.0, 2.0, 0.75)


def waterline_edge(x):
    """Half the hull's width at the waterline: finer at the bow and shorter at
    both ends than the deck above, which gives the bow its rake and flare."""
    return hull_half_breadth(x, 3.0, LENGTH - 7.0, 36.0, 0.66, LENGTH - 88.0, 1.8, 1.0)


def outline(half_width, x_aft, x_fwd, stations=36):
    """Plan-view outline between x_aft and x_fwd for a half-width function,
    counterclockwise seen from above.  Stations bunch up toward both ends,
    where the shape curves most.  half_width must not have any hollows, so the
    outline is convex."""
    xs = [x_aft + (x_fwd - x_aft) * (1 - math.cos(math.pi * i / (stations - 1))) / 2 for i in range(stations)]
    widths = [max(0.0, half_width(x)) for x in xs]
    starboard = [(x, -w) for x, w in zip(xs, widths)]
    port = [(x, w) for x, w in zip(xs, widths) if w > 1e-9]
    return starboard + port[::-1]


def blend(lower, upper, t):
    return [(xa + (xb - xa) * t, ya + (yb - ya) * t) for (xa, ya), (xb, yb) in zip(lower, upper)]


def tier_outline(x_aft, x_fwd, half_width, front_round=0.0, inset=0.0):
    """Outline of a deckhouse: straight sides, square stern end, and a front
    rounded over `front_round` meters.  It never overhangs the hull.  inset
    pulls every edge in by that much."""
    x_aft, x_fwd, half_width = x_aft + inset, x_fwd - inset, half_width - inset
    start = x_fwd - front_round

    def width(x):
        w = min(half_width, deck_edge(x) - 0.6 - inset)
        if front_round and x > start:
            w = min(w, half_width * math.sqrt(max(0.0, 1 - ((x - start) / front_round) ** 2)))
        return w
    return outline(width, x_aft, x_fwd)


def slab(points, z_bottom, z_top):
    return shapes.loft_outlines([(points, z_bottom), (points, z_top)])


def oval(center_x, half_length, half_width, points=32):
    return [(center_x + half_length * math.cos(2 * math.pi * i / points),
             half_width * math.sin(2 * math.pi * i / points)) for i in range(points)]


class Ship:
    """Collects the ship's pieces by color."""

    def __init__(self, deck_lines=True):
        self.pieces = {WHITE: [], BLACK: [], RED: [], ORANGE: []}
        self.deck_lines = deck_lines

    def add(self, color, mesh):
        self.pieces[color].append(mesh)

    def deck(self, deck, x_aft, x_fwd, half_width, front_round=0.0, glazed=False):
        """One deck of a deckhouse.  Its floor edge is a white rim on top; below
        it the balcony fronts sit back slightly, which gives the sides their
        horizontal lines.  A glazed deck's fronts are dark instead.  The rim
        flares out at 45 degrees underneath, so nothing overhangs."""
        z = deck_z(deck)
        rim_bottom = z + DECK_HEIGHT - RIM_HEIGHT
        full = tier_outline(x_aft, x_fwd, half_width, front_round)
        if glazed or self.deck_lines:
            body = tier_outline(x_aft, x_fwd, half_width, front_round, inset=RIB_DEPTH)
            self.add(BLACK if glazed else WHITE, slab(body, z, rim_bottom - RIB_DEPTH))
            self.add(WHITE, shapes.loft_outlines([(body, rim_bottom - RIB_DEPTH), (full, rim_bottom),
                                                  (full, z + DECK_HEIGHT)]))
        else:
            self.add(WHITE, slab(full, z, z + DECK_HEIGHT))


def build_ship(deck_lines=True, lifeboats=True):
    """The ship's hull, the raised ends of the hull, its other pieces by color,
    and its open decks as (outline, height) pairs.

    Deck numbers are the ship's own.  Positions along the hull are measured off
    Cunard's deck plans, in meters from the stern, so they are good to a couple
    of meters."""
    ship = Ship(deck_lines)
    open_decks = []
    details = {BLUE: [], WHITE: [], BLACK: []}      # Small things set into the surface, by color
    wide = HALF_BEAM - 0.6

    # Each detail is a (shape, cutter) pair.  The cutter is the shape carried on
    # past the surface: cutting with the shape itself, flush with the surface,
    # can leave a film of the old surface over it.
    def set_in(color, points, deck_height):
        """Something flush with a deck: a pool, or a marking."""
        details[color].append((slab(points, deck_height - PLANK, deck_height),
                               slab(points, deck_height - PLANK, deck_height + 1.0)))

    def at(mesh, x, y=0.0):
        mesh.apply_translation([x, y, 0])
        return mesh

    def house(x_aft, x_fwd, half_width, deck, top_deck, front_round=0.0):
        """A plain white deckhouse from the floor of one deck to the floor of another."""
        ship.add(WHITE, slab(tier_outline(x_aft, x_fwd, half_width, front_round), deck_z(deck), deck_z(top_deck)))

    def open_deck(x_aft, x_fwd, deck):
        """The floor of this deck is open to the sky between these positions.
        Whatever stands on it is cut out of its decking later."""
        open_decks.append((tier_outline(x_aft, x_fwd, wide, inset=DECK_MARGIN), deck_z(deck)))

    # --- Hull: red stripe at the waterline, dark above, flaring out to the deck ---
    waterline = outline(waterline_edge, 3.0, LENGTH - 7.0)
    deck_plan = outline(deck_edge, 0.0, LENGTH)
    boot_top = blend(waterline, deck_plan, BOOT_TOP / HULL_TOP)
    ship.add(RED, shapes.loft_outlines([(waterline, 0.0), (boot_top, BOOT_TOP)]))
    hull = shapes.loft_outlines([(boot_top, BOOT_TOP), (deck_plan, HULL_TOP)])
    # At both ends the dark hull carries on above the promenade deck, and the
    # white tiers rise from behind it.
    raised_ends = [
        slab(outline(deck_edge, LENGTH - FORECASTLE_LENGTH, LENGTH), HULL_TOP, HULL_TOP + FORECASTLE_HEIGHT),
        slab(outline(deck_edge, 0.0, STERN_LENGTH), HULL_TOP, HULL_TOP + STERN_HEIGHT),
    ]
    hull = shapes.union([hull] + raised_ends)

    # The top of the forecastle is the open observation deck at the bow, planked
    # as far forward as there is room.
    def planked_edge(x):
        return deck_edge(x) - DECK_MARGIN
    low, high = LENGTH - FORECASTLE_LENGTH, LENGTH
    for _ in range(40):                                 # Where the decking narrows to a meter wide
        middle = (low + high) / 2
        low, high = (middle, high) if planked_edge(middle) > 0.5 else (low, middle)
    open_decks.append((outline(planked_edge, LENGTH - FORECASTLE_LENGTH, low), HULL_TOP + FORECASTLE_HEIGHT))
    # The helicopter landing circle on it.
    bow_deck = HULL_TOP + FORECASTLE_HEIGHT
    hole = slab(oval(LENGTH - 19.0, 4.7, 4.7), bow_deck - PLANK - 1.0, bow_deck + 2.0)
    details[WHITE].append((shapes.difference(slab(oval(LENGTH - 19.0, 6.0, 6.0), bow_deck - PLANK, bow_deck), [hole]),
                           shapes.difference(slab(oval(LENGTH - 19.0, 6.0, 6.0), bow_deck - PLANK, bow_deck + 1.0), [hole])))

    # --- Decks 3 to 8: the long block of cabins, upright at the stern and
    # stepping back a little with each deck at the bow ---
    stern = 3.5
    ship.deck(3, stern, LENGTH - 40.0, wide)
    # Deck 4 is set in where the lifeboats hang: three aft and eight forward of
    # amidships.  The boats stand on the deck below and reach the deck above, so
    # they carry its overhang when printing.
    aft_boats = [(65.3, 4.3), (80.9, 5.5), (96.5, 4.3)]
    forward_boats = [(150.0 + 10.83 * i, 4.3) for i in range(8)]
    boat_zones = [(59.0, 102.5, aft_boats), (142.0, 231.7, forward_boats)] if lifeboats else []
    edges = [stern] + [x for zone in boat_zones for x in zone[:2]] + [LENGTH - 40.0]
    for i, (x_aft, x_fwd) in enumerate(zip(edges, edges[1:])):
        ship.deck(4, x_aft, x_fwd, 14.6 if i % 2 else wide)
    ship.deck(5, stern, LENGTH - 28.0, wide, front_round=12)
    ship.deck(6, stern, LENGTH - 29.5, wide, front_round=12)
    ship.deck(7, stern, LENGTH - 31.0, wide, front_round=12)

    # Deck 8: the bridge, with its dark windows, at the front.
    bridge_front = LENGTH - 34.5
    ship.deck(8, stern, bridge_front - 12.0, wide)
    ship.deck(8, bridge_front - 12.0, bridge_front, wide, front_round=10, glazed=True)

    # --- Deck 9: an open pool deck at the stern with a wing down each side,
    # then a long run of windows, then cabins forward ---
    cabins = 208.0                      # Where the forward cabins start on decks 9 to 11
    pool_deck = 38.0
    open_deck(stern, pool_deck + 1.0, 9)
    # The Panorama pool, with a whirlpool on each side of its aft end.
    set_in(BLUE, [(18.4, -2.25), (26.3, -2.25), (26.3, 2.25), (18.4, 2.25)], deck_z(9))
    for side in (-1, 1):
        set_in(BLUE, [(15.7, side * 3.5 - 1.0), (17.7, side * 3.5 - 1.0), (17.7, side * 3.5 + 1.0), (15.7, side * 3.5 + 1.0)], deck_z(9))
    wings = slab(tier_outline(11.5, pool_deck, wide), deck_z(9), deck_z(10))
    court = shapes.block([pool_deck, 21.0, DECK_HEIGHT + 2.0], [pool_deck / 2 + 2.0, 0, deck_z(9) + DECK_HEIGHT / 2])
    ship.add(WHITE, shapes.difference(wings, [court]))
    ship.deck(9, pool_deck, cabins, wide, glazed=True)
    ship.deck(9, cabins, LENGTH - 35.5, wide, front_round=12)

    # --- Deck 10: restaurants aft and amidships with open side decks between
    # them around the funnel's casing, then cabins forward ---
    open_deck(11.5, 100.0, 10)
    ship.deck(10, pool_deck, 58.5, wide, glazed=True)
    ship.deck(10, 58.5, 99.5, 11.0)
    ship.deck(10, 99.5, cabins, wide, glazed=True)
    ship.deck(10, cabins, LENGTH - 37.0, wide, front_round=12)

    # --- Deck 11: cabins forward; aft of them it is open deck, with the
    # deckhouses, the funnel and the Pavilion's roof standing on it ---
    ship.deck(11, cabins, LENGTH - 38.5, wide, front_round=12)
    open_deck(pool_deck, cabins, 11)
    # Whirlpools at the sides of the Grills Terrace.
    for side in (-1, 1):
        set_in(BLUE, [(114.0, side * 15.1 - 0.9), (118.0, side * 15.1 - 0.9), (118.0, side * 15.1 + 0.9), (114.0, side * 15.1 + 0.9)], deck_z(11))

    # A deckhouse carrying the two satellite domes.  The domes have straight
    # sides up to their widest point, so there is no underside to print in mid-air.
    house(49.0, 68.0, 6.5, 11, 14)
    for x, y in ((53.0, 2.5), (62.0, -2.5)):
        ship.add(WHITE, at(shapes.cylinder(2.6, deck_z(14) - 0.2, deck_z(14) + 3.4, sections=16), x, y))
        ship.add(WHITE, shapes.sphere(2.6, [x, y, deck_z(14) + 3.4], detail=2))

    # The funnel on its casing: Cunard red with two black bands and a black top.
    # Its front is upright and its back slopes forward as it rises.
    house(76.0, 106.5, 6.7, 11, 14, front_round=5)
    funnel_base = deck_z(14)
    bands = ((RED, 0.0, 5.6), (BLACK, 5.6, 6.3), (RED, 6.3, 8.3), (BLACK, 8.3, 9.0), (RED, 9.0, 11.2), (BLACK, 11.2, 15.4))
    funnel_height = bands[-1][2]

    def funnel_section(height):
        t = height / funnel_height
        return oval(90.8 + 1.8 * t, 9.4 - 1.8 * t, 4.1 - 0.5 * t)
    for color, low, high in bands:
        ship.add(color, shapes.loft_outlines([(funnel_section(low), funnel_base + low),
                                              (funnel_section(high), funnel_base + high)]))

    # A taller deckhouse amidships with a slim stack, dark at the top.
    house(128.0, 158.0, 9.0, 11, 15, front_round=4)
    ship.add(WHITE, at(shapes.frustum(2.0, 1.5, deck_z(15) - 0.2, deck_z(15) + 8.6, sections=12), 138.0))
    ship.add(BLACK, at(shapes.frustum(1.5, 1.4, deck_z(15) + 8.6, deck_z(15) + 10.1, sections=12), 138.0))

    # The Pavilion's vaulted glass roof, over its pool court between that
    # deckhouse and the forward cabins.
    vault = shapes.ellipsoid([23.5, 10.7, 5.0], [182.0, 0.0, deck_z(11)], detail=3)
    above = shapes.block([LENGTH, BEAM * 2, 20.0], [LENGTH / 2, 0, deck_z(11) + 10.0])
    ship.add(WHITE, shapes.intersection([vault, above]))
    # Its glazing lines: dark bars across the roof and one along the ridge, set
    # into the surface.
    bars = [shapes.block([0.7, 30.0, 12.0], [182.0 + 4.6 * i, 0, deck_z(11) + 6.0]) for i in range(-4, 5)]
    bars.append(shapes.block([44.0, 0.7, 12.0], [182.0, 0, deck_z(11) + 6.0]))
    under_skin = shapes.ellipsoid([23.5 - 0.8, 10.7 - 0.8, 5.0 - 0.8], [182.0, 0.0, deck_z(11)], detail=3)
    bars = shapes.difference(shapes.union(bars), [under_skin])
    details[BLACK].append((shapes.intersection([bars, vault, above]), bars))

    # --- Decks 12 and 14, on the forward block: the glazed lounge, with an open
    # play deck behind it, and the observation deck on its roof ---
    lounge_aft, lounge_front = 222.0, LENGTH - 42.5
    open_deck(cabins, lounge_aft + 1.0, 12)
    ship.deck(12, lounge_aft, lounge_front, wide - 0.6, front_round=12, glazed=True)
    open_deck(lounge_aft, lounge_front, 14)
    house(226.0, 266.0, 7.0, 14, 15, front_round=6)
    # The mast.  Its radar platform flares out at 45 degrees underneath so it
    # prints without support.
    mast_x, mast_base = 249.0, deck_z(15)
    ship.add(WHITE, at(shapes.frustum(2.2, 1.1, mast_base - 0.2, mast_base + 12.8, sections=12), mast_x))
    ship.add(WHITE, at(shapes.lathe([(1.6, mast_base + 5.0), (3.0, mast_base + 6.4), (3.0, mast_base + 7.1)], sections=16), mast_x))

    # --- Lifeboats and tenders ---
    # Each boat is flat on top and reaches the deck above, so the boats carry
    # that deck's overhang and it only has to bridge the short gaps between them.
    boat_floor, boat_top = deck_z(4), deck_z(5)

    def boat_section(half_length, half_width, side, x):
        return [(px, py + side * 16.15) for px, py in oval(x, half_length, half_width, points=20)]
    for zone_aft, zone_fwd, boats in boat_zones:
        zone_boats = []
        for x, half_length in boats:
            for side in (-1, 1):
                zone_boats.append(shapes.loft_outlines([
                    (boat_section(half_length - 0.6, 1.1, side, x), boat_floor),
                    (boat_section(half_length, 1.5, side, x), boat_floor + 1.0),
                    (boat_section(half_length, 1.5, side, x), boat_top),
                ]))
        # Between the boats, the wall behind them slopes out at 45 degrees to
        # meet the deck above.
        reach = wide - RIB_DEPTH - 14.6
        slope = shapes.loft_outlines([(tier_outline(zone_aft, zone_fwd, 14.6), boat_top - reach),
                                      (tier_outline(zone_aft, zone_fwd, 14.6 + reach), boat_top)])
        ship.add(WHITE, shapes.difference(slope, zone_boats))
        for boat in zone_boats:
            ship.add(ORANGE, boat)

    return hull, raised_ends, ship.pieces, open_decks, details


def merged(meshes):
    return meshes[0] if len(meshes) == 1 else shapes.union(meshes)


def lay_decking(open_decks, standing, structure):
    """Decking for the open decks: a thin layer set into the top of the
    structure at each one, with holes where something stands on it."""
    planks = []
    for points, z_top in open_decks:
        plank = shapes.intersection([slab(points, z_top - PLANK, z_top), structure])
        if plank.is_empty:
            continue
        # A thin slice of whatever stands on this deck, stretched down through the decking.
        footprint = shapes.intersection([standing, shapes.block([LENGTH * 2, BEAM * 2, 0.3], [LENGTH / 2, 0, z_top + 0.2])])
        if not footprint.is_empty:
            footprint.apply_translation([0, 0, -(z_top + 0.05)])
            footprint.apply_scale([1, 1, (PLANK + 0.6) / 0.3])
            footprint.apply_translation([0, 0, z_top - PLANK - 0.3])
            plank = shapes.difference(plank, [footprint])
        if not plank.is_empty and plank.volume > 1.0:
            planks.append(plank)
    return merged(planks)


def hull_lettering(hull, text, scale):
    """The name on both sides of the hull near the bow, as a thin layer of the
    hull's skin.  Letters are at least MIN_LETTER_MM tall when printed, which is
    larger than true scale on a small model.  Returns the lettering and the
    shape to cut out of the hull for it."""
    height = max(LETTER_HEIGHT, MIN_LETTER_MM / scale)
    pixel = height / 7
    width = (6 * len(text) - 1) * pixel
    x_aft = LENGTH - 62.0 - width / 2
    z_top = HULL_TOP - 1.2
    blocks = []
    for side in (-1, 1):
        for i, letter in enumerate(text):
            for row, line in enumerate(FONT[letter]):
                for column, mark in enumerate(line):
                    if mark != '#':
                        continue
                    along = (6 * i + column) * pixel
                    # Seen from outside, the name reads from stern to bow on the
                    # starboard side and from bow to stern on the port side.
                    x = x_aft + along if side < 0 else x_aft + width - along - pixel
                    blocks.append(shapes.block([pixel * 1.04, HALF_BEAM, pixel * 1.04],
                                               [x + pixel / 2, side * HALF_BEAM, z_top - (row + 0.5) * pixel]))
    inside = hull.copy()
    inside.apply_scale([1, (HALF_BEAM - 0.9) / HALF_BEAM, 1])
    cutter = shapes.difference(shapes.union(blocks), [inside])     # Carries on outside the hull
    return shapes.intersection([cutter, hull]), cutter


def make_parts(length_mm, deck_lines=True, lifeboats=True, decks=True, pools=True, lettering=True):
    """The ship as named, colored parts in millimeters, centered on the origin
    and sitting on z = 0."""
    scale = length_mm / LENGTH
    hull, raised_ends, pieces, open_decks, details = build_ship(deck_lines, lifeboats)
    # The raised ends of the hull hide the foot of the tiers behind them.
    white = shapes.difference(merged(pieces[WHITE]), raised_ends)
    black = shapes.difference(merged(pieces[BLACK]), raised_ends)
    funnel = merged(pieces[RED][1:])
    boats = merged(pieces[ORANGE]) if pieces[ORANGE] else None

    name = None
    if lettering:
        name, cutter = hull_lettering(hull, 'CUNARD', scale)
        hull = shapes.difference(hull, [cutter])

    decking = None
    if decks:
        decking = lay_decking(open_decks, shapes.union([white, black, funnel] + ([boats] if boats else [])),
                              shapes.union([white, hull]))
        # Cut with the decking carried a little above the deck, for the same
        # reason as the details below.
        lifted = decking.copy()
        lifted.apply_translation([0, 0, PLANK * 0.8])
        white = shapes.difference(white, [decking, lifted])
        hull = shapes.difference(hull, [decking, lifted])

    # Small things set into the surface: cut each out of whatever is there, and
    # add it to its own color.
    if not pools:
        details[BLUE] = []
    cutters = [cutter for color in details for _, cutter in details[color]]
    white = shapes.difference(white, [cutter for color in (BLACK, BLUE) for _, cutter in details[color]])
    hull = shapes.difference(hull, cutters)
    if decking:
        decking = shapes.difference(decking, cutters)
    white = shapes.union([white] + [shape for shape, _ in details[WHITE]])
    black = shapes.union([black] + [shape for shape, _ in details[BLACK]])
    water = merged([shape for shape, _ in details[BLUE]]) if details[BLUE] else None

    named = [
        ('Superstructure', WHITE, white),
        ('Hull', BLACK, hull),
        ('Windows and funnel top', BLACK, black),
        ('Waterline stripe', RED, merged(pieces[RED][:1])),
        ('Funnel', RED, funnel),
    ]
    if name:
        named.append(('Name on the hull', RED, name))
    if boats:
        named.append(('Lifeboats', ORANGE, boats))
    if decking:
        named.append(('Decks', TAN, decking))
    if water:
        named.append(('Pools', BLUE, water))

    parts = []
    for part_name, color, mesh in named:
        mesh.apply_translation([-LENGTH / 2, 0, 0])
        mesh.apply_scale(scale)
        filament, hex_color = FILAMENTS[color]
        parts.append(Part(part_name, mesh, hex_color, filament))
    return parts


def main():
    parser = argparse.ArgumentParser(description="Generate a multi-color waterline model of Cunard's MS Queen Anne.")
    parser.add_argument('--length', type=dimensions_type((1,), '--length'), default=None, metavar='LENGTH',
                        help='Length of the model, in millimeters or with a unit like 12in (default: the longest that fits the bed straight).')
    parser.add_argument('--diagonal', action='store_true',
                        help='Make the model as long as fits the bed corner to corner instead of straight (about 27%% longer on a square bed). Ignored with --length.')
    parser.add_argument('--bed', type=dimensions_type((1, 2), '--bed'), default=(256.0,), metavar='SIZE|WxD',
                        help='Printer bed size (default: 256, the Bambu Lab X1/P1/X2D bed).')
    parser.add_argument('--deck_lines', action=argparse.BooleanOptionalAction, default=True,
                        help='Set the balcony fronts back from the deck edges to give the sides horizontal lines (default: on).')
    parser.add_argument('--lifeboats', action=argparse.BooleanOptionalAction, default=True,
                        help='Include the orange lifeboats; --no-lifeboats saves a filament (default: on).')
    parser.add_argument('--decks', action=argparse.BooleanOptionalAction, default=True,
                        help='Lay light brown decking on the open decks; --no-decks saves a filament (default: on).')
    parser.add_argument('--pools', action=argparse.BooleanOptionalAction, default=True,
                        help='Include the blue pools; --no-pools saves a filament (default: on).')
    parser.add_argument('--lettering', action=argparse.BooleanOptionalAction, default=True,
                        help='Put CUNARD on the hull in red. The letters are printed larger than true scale on a small model so their strokes are printable (default: on).')
    parser.add_argument('--output', type=str, default=None, help='Output filename; .stl or .3mf (default: queen_anne.3mf).')
    add_format_argument(parser, default='3mf')
    args = parser.parse_args()

    bed = args.bed * 2 if len(args.bed) == 1 else args.bed
    max_width, max_depth = bed[0] - 2 * BED_MARGIN, bed[1] - 2 * BED_MARGIN
    beam_ratio = BEAM / LENGTH
    if args.length:
        length = args.length[0]
    elif args.diagonal:
        # Longest model whose length and matching beam still fit corner to corner.
        length = max_width
        for _ in range(60):
            length = longest_fit(length * beam_ratio, max_width, max_depth)
    else:
        length = max(max_width, max_depth)
    beam = length * beam_ratio
    if length > longest_fit(beam, max_width, max_depth) + 1e-6:
        print(f"Error: A {format_mm(length)}mm model is too long for the bed, even corner to corner. "
              f"The longest that fits is about {format_mm(longest_fit(beam, max_width, max_depth))}mm.")
        sys.exit(1)

    scale = length / LENGTH
    print(f"Creating Queen Anne: {format_mm(length)}mm long, scale 1:{1 / scale * 1000:.0f}")
    parts = make_parts(length, args.deck_lines, args.lifeboats, args.decks, args.pools, args.lettering)

    for part in parts:
        if not part.mesh.is_watertight:
            print(f"Warning: The {part.name.lower()} mesh is not watertight; the slicer may need to repair it.")

    # A model longer than the bed is turned so it prints corner to corner.
    turn = print_rotation(length, beam, max_width, max_depth)
    if turn:
        for part in parts:
            shapes.rotate_z(part.mesh, turn)

    output_filename = output_path(args.output, 'queen_anne', args.format)
    if output_filename.lower().endswith('.3mf'):
        save(parts, output_filename, assembly='Queen Anne')
        written = [output_filename]
    else:
        # STL has no colors or parts, so write one file per filament.
        base = output_filename[:-len('.stl')] if output_filename.lower().endswith('.stl') else output_filename
        written = []
        for color, (filament, _) in FILAMENTS.items():
            meshes = [part.mesh for part in parts if part.filament == filament]
            if meshes:
                name = f"{base}_{filament}_{color.replace(' ', '_')}.stl"
                trimesh.util.concatenate(meshes).export(name)
                written.append(name)
    for name in written:
        print(f"Model exported to '{name}'.")

    height = max(part.mesh.bounds[1][2] for part in parts)
    print("\nModel:")
    print(f"  Size: {format_mm(length)} x {format_mm(beam)} x {format_mm(height)}mm (length x beam x height)")
    print(f"  Scale: 1:{1 / scale * 1000:.0f}")
    if turn:
        print(f"  Longer than the bed: it is turned {turn:g} degrees in the file to print corner to corner.")
    print("  Filaments:")
    for color, (filament, _) in FILAMENTS.items():
        names = [part.name.lower() for part in parts if part.filament == filament]
        if names:
            print(f"    {filament} {color}: {', '.join(names)}")
    if not written[0].lower().endswith('.3mf'):
        print("  STL files hold no colors: load them together as one object with several parts and "
              "give each its filament, or use --format 3mf.")


if __name__ == "__main__":
    main()
