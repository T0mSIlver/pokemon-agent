"""``scope`` — read a live run the way a supervising agent has to read it.

The consumer of every command in this package is a language model with a finite
context window, not a person with a scrollbar. A single Pi session transcript is
several hundred JSONL lines with base64 PNGs inline; dumping it is not an option
at any budget. So the contract each command keeps is: **one question, one
answer, about fifty lines of plain text**, aggregated hard, numbers over prose,
no colour escapes, and never a byte of base64 on stdout. ``--full`` widens a
report only when it is asked for; ``--json`` hands the same figures to a
program.

Everything is read-only. A run is playing while these commands are used, so the
readers tolerate a file growing under them and a final line that is half
written, and nothing here opens a file for writing.
"""

from pokemon_agent.scope.discover import Paths, discover

__all__ = ["Paths", "discover"]
