"""Widgets assembled once and reused, rather than re-assembled per screen.

A component earns a place here by fixing something the property vocabulary in
``kimix_gui.qt.styling`` cannot reach on its own. That vocabulary already covers
*appearance*: an overline is ``style(label, role=Role.OVERLINE)`` and an icon
button is ``style(button, variant=Variant.ICON)``, so wrapping either in a class
would add a name and nothing else. What it cannot cover is *arrangement* --
which order buttons sit in, which one Enter activates, how a key column lines up
with its values -- because none of that is expressible in a style sheet.

So the rule for this package is: a component owns structure, keyboard behaviour,
and accessibility. It does not own copy, and it never sets a fixed width to make
text fit, because the text changes length with the language.

The other thing that earns a class here is *reacting*. A measurement taken from
the font has to be retaken when the font changes, and Qt only offers that as a
``changeEvent`` override -- which needs a class. ``SettingsList`` exists for that
reason and no other: it replaced a free function that stamped a row height on
once and left it wrong for the rest of the session.
"""

from kimix_gui.qt.components.card import Card
from kimix_gui.qt.components.footer import DialogFooter
from kimix_gui.qt.components.key_value import KeyValueList
from kimix_gui.qt.components.settings_list import SettingsList

# ``CardLevel`` is not re-exported: it is part of the property vocabulary and
# belongs with ``Variant`` and ``Role`` in :mod:`kimix_gui.qt.styling`, which is
# where callers already import from.
__all__ = ["Card", "DialogFooter", "KeyValueList", "SettingsList"]
