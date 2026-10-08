"""Parsing lengths and dimensions typed on the command line.

Lengths are millimeters unless they carry a unit: mm, cm, in (also inch,
inches or ").  Inches can be decimals or tape-measure fractions:

    250          250 mm
    9.5in        241.3 mm
    15-3/4in     15 3/4 inches (a space works too if the argument is quoted)
    3/8in        3/8 inch

Dimensions join lengths with x.  A unit on the last length also applies to any
length without its own unit, so these are the same:

    20-1/2x15x3in     20-1/2inx15inx3in
"""

import argparse
import re

MM_PER_UNIT = {'mm': 1.0, 'cm': 10.0, 'in': 25.4, 'inch': 25.4, 'inches': 25.4, '"': 25.4}
_UNITS_LONGEST_FIRST = sorted(MM_PER_UNIT, key=len, reverse=True)
_NUMBER = re.compile(r'^(?:(?P<whole>\d+(?:\.\d+)?)-)?(?P<num>\d+)/(?P<den>\d+)$|^(?P<plain>\d+(?:\.\d*)?|\.\d+)$')


def _split_unit(token):
    for unit in _UNITS_LONGEST_FIRST:
        if token.endswith(unit):
            return token[:-len(unit)], unit
    return token, None


def _number(text):
    match = _NUMBER.match(text)
    if not match:
        raise ValueError(f"'{text}' is not a number")
    if match['plain'] is not None:
        return float(match['plain'])
    if float(match['den']) == 0:
        raise ValueError(f"'{text}' divides by zero")
    whole = float(match['whole']) if match['whole'] else 0.0
    return whole + float(match['num']) / float(match['den'])


def parse_dimensions(text, counts=(1, 2, 3)):
    """Parse '250', '300x200', '20-1/2x15x3in' and the like into a tuple of
    millimeter values.  counts is how many values are allowed."""
    cleaned = text.strip().lower()
    cleaned = re.sub(r'(\d)\s+(\d+/\d+)', r'\1-\2', cleaned)   # "15 3/4" -> "15-3/4"
    cleaned = cleaned.replace(' ', '')

    parts = [_split_unit(p) for p in cleaned.split('x')]
    default_unit = parts[-1][1] or 'mm'
    values = tuple(_number(number) * MM_PER_UNIT[unit or default_unit] for number, unit in parts)

    if len(values) not in counts:
        raise ValueError(f"expected {' or '.join(str(c) for c in counts)} value(s), got {len(values)}")
    if min(values) <= 0:
        raise ValueError("sizes must be greater than zero")
    return values


def dimensions_type(counts, name):
    """An argparse type for parse_dimensions with a helpful error message."""
    forms = {1: 'SIZE', 2: 'WIDTHxDEPTH', 3: 'WIDTHxDEPTHxHEIGHT'}

    def parse(text):
        try:
            return parse_dimensions(text, counts)
        except ValueError as error:
            wanted = ' or '.join(forms[c] for c in counts)
            raise argparse.ArgumentTypeError(
                f"{name} must be {wanted}, in mm or with a unit like 'in' "
                f"(e.g. 400x300 or 15-3/4x12in): {error}") from None
    return parse


def format_mm(value):
    """Short millimeter text, e.g. 520 or 241.3."""
    return f"{value:.1f}".rstrip('0').rstrip('.')
