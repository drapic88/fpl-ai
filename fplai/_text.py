"""One place for the name folding that every matcher in the project depends on.

Four call sites compare a human-typed or headline-scraped name against the way
FPL spells it: the planner and the watcher resolve `my_squad.json`, the news
layer matches headlines, and the external layer matches Understat rows. They
must agree character-for-character -- the day they disagree is the day a player
resolves in one and vanishes in another -- so the table lives here rather than
being copied into each of them.
"""

from __future__ import annotations

#: Letters NFKD leaves alone, because Unicode treats them as letters in their own
#: right rather than as an accented Latin one. FPL prints them as they are spelt;
#: a hand-typed squad file and an English-language headline almost always use the
#: keyboard spelling, so 'Gross' would never reach 'Groß' and 'Odegaard' never
#: reaches 'Ødegaard'.
LETTER_FOLD = str.maketrans({
    "ß": "ss", "æ": "ae", "œ": "oe", "ø": "o", "đ": "d",
    "ð": "d", "ł": "l", "þ": "th", "ħ": "h", "ı": "i",
})


def fold_letters(text: str) -> str:
    """Lower-case `text` and fold the letters NFKD will not touch.

    Folding after lower-casing is what covers the capital forms, since 'Ø' and
    'ẞ' lower-case onto keys that are in the table.
    """
    return str(text).lower().translate(LETTER_FOLD)
