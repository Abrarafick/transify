"""
Two-Way Braille Studio and Perkins 6-Dot Interactive Simulator for Transify Module 02.
Supports bidirectional translation between English text and Unified English Braille (UEB) Unicode,
plus interactive 6-dot cell decoding for tactile learning and typing.
"""

from typing import Dict, List, Tuple, Any, Optional

DOT_OFFSET = 0x2800

# 6-dot pattern to character mappings
DOTS_TO_ALPHA: Dict[Tuple[int, int, int, int, int, int], str] = {
    (1, 0, 0, 0, 0, 0): 'a',
    (1, 1, 0, 0, 0, 0): 'b',
    (1, 0, 0, 1, 0, 0): 'c',
    (1, 0, 0, 1, 1, 0): 'd',
    (1, 0, 0, 0, 1, 0): 'e',
    (1, 1, 0, 1, 0, 0): 'f',
    (1, 1, 0, 1, 1, 0): 'g',
    (1, 1, 0, 0, 1, 0): 'h',
    (0, 1, 0, 1, 0, 0): 'i',
    (0, 1, 0, 1, 1, 0): 'j',
    (1, 0, 1, 0, 0, 0): 'k',
    (1, 1, 1, 0, 0, 0): 'l',
    (1, 0, 1, 1, 0, 0): 'm',
    (1, 0, 1, 1, 1, 0): 'n',
    (1, 0, 1, 0, 1, 0): 'o',
    (1, 1, 1, 1, 0, 0): 'p',
    (1, 1, 1, 1, 1, 0): 'q',
    (1, 1, 1, 0, 1, 0): 'r',
    (0, 1, 1, 1, 0, 0): 's',
    (0, 1, 1, 1, 1, 0): 't',
    (1, 0, 1, 0, 0, 1): 'u',
    (1, 1, 1, 0, 0, 1): 'v',
    (0, 1, 0, 1, 1, 1): 'w',
    (1, 0, 1, 1, 0, 1): 'x',
    (1, 0, 1, 1, 1, 1): 'y',
    (1, 0, 1, 0, 1, 1): 'z',
}

ALPHA_TO_DOTS: Dict[str, Tuple[int, int, int, int, int, int]] = {
    v: k for k, v in DOTS_TO_ALPHA.items()
}

NUM_TO_DOTS: Dict[str, Tuple[int, int, int, int, int, int]] = {
    '1': (1, 0, 0, 0, 0, 0),
    '2': (1, 1, 0, 0, 0, 0),
    '3': (1, 0, 0, 1, 0, 0),
    '4': (1, 0, 0, 1, 1, 0),
    '5': (1, 0, 0, 0, 1, 0),
    '6': (1, 1, 0, 1, 0, 0),
    '7': (1, 1, 0, 1, 1, 0),
    '8': (1, 1, 0, 0, 1, 0),
    '9': (0, 1, 0, 1, 0, 0),
    '0': (0, 1, 0, 1, 1, 0),
}

PUNCT_TO_DOTS: Dict[str, Tuple[int, int, int, int, int, int]] = {
    '.': (0, 1, 0, 0, 1, 1),
    ',': (0, 1, 0, 0, 0, 0),
    ';': (0, 1, 1, 0, 0, 0),
    ':': (0, 1, 0, 0, 1, 0),
    '?': (0, 1, 1, 0, 0, 1),
    '!': (0, 1, 1, 0, 1, 0),
    '-': (0, 0, 1, 0, 0, 1),
    "'": (0, 0, 1, 0, 0, 0),
    '"': (0, 0, 1, 0, 1, 1),
}

CAPITAL_PREFIX = (0, 0, 0, 0, 0, 1)  # Dot 6
NUMBER_PREFIX = (0, 0, 1, 1, 1, 1)   # Dots 3-4-5-6
SPACE_DOTS = (0, 0, 0, 0, 0, 0)


def dots_to_unicode(dots: Tuple[int, int, int, int, int, int]) -> str:
    """Converts a 6-tuple of binary dot states to a Unicode Braille character."""
    mask = 0
    for i, active in enumerate(dots):
        if active:
            mask |= (1 << i)
    return chr(DOT_OFFSET + mask)


def unicode_to_dots(char: str) -> Tuple[int, int, int, int, int, int]:
    """Converts a Unicode Braille character back to a 6-tuple of binary dot states."""
    code = ord(char)
    if not (0x2800 <= code <= 0x28FF):
        return SPACE_DOTS
    mask = code - 0x2800
    return tuple((mask >> i) & 1 for i in range(6))


def text_to_braille(text: str) -> Dict[str, Any]:
    """
    Translates standard English text into Unified English Braille (UEB) Grade 1.
    Handles capital letter indicators, numeric mode prefixes, and punctuation.
    """
    braille_chars = []
    cells_metadata = []
    in_numeric_mode = False

    for ch in text:
        if ch == ' ':
            braille_chars.append(dots_to_unicode(SPACE_DOTS))
            cells_metadata.append({
                "char": "SPACE",
                "unicode": "⠀",
                "dots": [],
                "dot_vector": list(SPACE_DOTS),
                "is_space": True
            })
            in_numeric_mode = False
        elif ch.isdigit():
            if not in_numeric_mode:
                braille_chars.append(dots_to_unicode(NUMBER_PREFIX))
                cells_metadata.append({
                    "char": "#",
                    "unicode": dots_to_unicode(NUMBER_PREFIX),
                    "dots": [3, 4, 5, 6],
                    "dot_vector": list(NUMBER_PREFIX),
                    "is_indicator": True
                })
                in_numeric_mode = True
            dots = NUM_TO_DOTS.get(ch, SPACE_DOTS)
            braille_chars.append(dots_to_unicode(dots))
            active_dots = [i + 1 for i, val in enumerate(dots) if val]
            cells_metadata.append({
                "char": ch,
                "unicode": dots_to_unicode(dots),
                "dots": active_dots,
                "dot_vector": list(dots)
            })
        elif ch.isalpha():
            in_numeric_mode = False
            if ch.isupper():
                braille_chars.append(dots_to_unicode(CAPITAL_PREFIX))
                cells_metadata.append({
                    "char": "CAP",
                    "unicode": dots_to_unicode(CAPITAL_PREFIX),
                    "dots": [6],
                    "dot_vector": list(CAPITAL_PREFIX),
                    "is_indicator": True
                })
            dots = ALPHA_TO_DOTS.get(ch.lower(), SPACE_DOTS)
            braille_chars.append(dots_to_unicode(dots))
            active_dots = [i + 1 for i, val in enumerate(dots) if val]
            cells_metadata.append({
                "char": ch,
                "unicode": dots_to_unicode(dots),
                "dots": active_dots,
                "dot_vector": list(dots)
            })
        elif ch in PUNCT_TO_DOTS:
            in_numeric_mode = False
            dots = PUNCT_TO_DOTS[ch]
            braille_chars.append(dots_to_unicode(dots))
            active_dots = [i + 1 for i, val in enumerate(dots) if val]
            cells_metadata.append({
                "char": ch,
                "unicode": dots_to_unicode(dots),
                "dots": active_dots,
                "dot_vector": list(dots)
            })
        else:
            in_numeric_mode = False
            braille_chars.append(ch)

    braille_unicode = "".join(braille_chars)

    return {
        "success": True,
        "input_text": text,
        "braille_unicode": braille_unicode,
        "cell_count": len(cells_metadata),
        "cells": cells_metadata
    }


def cell_to_text(dot_vector: List[int]) -> Dict[str, Any]:
    """
    Decodes an interactive 6-dot cell configuration into character and Unicode Braille.
    """
    if len(dot_vector) != 6:
        dot_vector = (dot_vector + [0] * 6)[:6]

    dots = tuple(1 if x else 0 for x in dot_vector)
    unicode_char = dots_to_unicode(dots)
    active_dots = [i + 1 for i, val in enumerate(dots) if val]

    # Look up in alphabet, punctuation, and numeric
    char = DOTS_TO_ALPHA.get(dots)
    char_type = "letter"
    if not char:
        for punct, p_dots in PUNCT_TO_DOTS.items():
            if p_dots == dots:
                char = punct
                char_type = "punctuation"
                break
    if not char:
        if dots == CAPITAL_PREFIX:
            char = "[CAPITAL]"
            char_type = "indicator"
        elif dots == NUMBER_PREFIX:
            char = "[NUMERIC]"
            char_type = "indicator"
        elif dots == SPACE_DOTS:
            char = " "
            char_type = "space"
        else:
            char = "?"
            char_type = "unknown"

    return {
        "success": True,
        "dots": active_dots,
        "dot_vector": list(dots),
        "unicode": unicode_char,
        "char": char,
        "char_type": char_type
    }


def braille_to_text(braille_unicode: str) -> Dict[str, Any]:
    """
    Decodes a string of Unicode Braille characters (e.g. '⠠⠓⠑⠇⠇⠕') into plain English.
    """
    text_chars = []
    capital_next = False
    numeric_mode = False

    for ch in braille_unicode:
        code = ord(ch)
        if 0x2800 <= code <= 0x28FF:
            dots = unicode_to_dots(ch)
            if dots == SPACE_DOTS:
                text_chars.append(" ")
                numeric_mode = False
                capital_next = False
            elif dots == CAPITAL_PREFIX:
                capital_next = True
            elif dots == NUMBER_PREFIX:
                numeric_mode = True
            elif numeric_mode:
                # Look up numeric
                num_char = None
                for n, d in NUM_TO_DOTS.items():
                    if d == dots:
                        num_char = n
                        break
                if num_char:
                    text_chars.append(num_char)
                else:
                    numeric_mode = False
                    letter = DOTS_TO_ALPHA.get(dots)
                    if letter:
                        text_chars.append(letter)
                    else:
                        punct = None
                        for p, d in PUNCT_TO_DOTS.items():
                            if d == dots:
                                punct = p
                                break
                        text_chars.append(punct if punct else "?")
            else:
                letter = DOTS_TO_ALPHA.get(dots)
                if letter:
                    if capital_next:
                        letter = letter.upper()
                        capital_next = False
                    text_chars.append(letter)
                else:
                    punct = None
                    for p, d in PUNCT_TO_DOTS.items():
                        if d == dots:
                            punct = p
                            break
                    text_chars.append(punct if punct else "?")
                    capital_next = False
        else:
            text_chars.append(ch)

    decoded_text = "".join(text_chars)
    return {
        "success": True,
        "input_braille": braille_unicode,
        "decoded_text": decoded_text
    }
