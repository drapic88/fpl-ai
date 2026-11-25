"""Save what FPL overwrites before each deadline, and the results after it.

    python snapshot.py              # pre-deadline snapshot if one is due, then any finished gameweeks
    python snapshot.py --list       # what has been saved so far
    python snapshot.py --post-only  # only results of finished gameweeks

Safe to run at any time and as often as you like: a pre-deadline snapshot is
taken only within 30 hours of the next deadline, never after it, and nothing
already saved is overwritten. See fplai/snapshot.py for what each one holds.
"""

import sys

from fplai.snapshot import main

if __name__ == "__main__":
    sys.exit(main())
