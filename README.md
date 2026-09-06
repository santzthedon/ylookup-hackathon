https://www.loom.com/share/1b5cb70ab95549dda7363a02c7cef155
# Migration decision review

Finds every human decision a fund administration data migration needs, in one
pass, before anything is built.

## The problem

Moving a quarter of accounting records from one fund accounting system to
another takes about a month by hand. The time does not go on the mapping. It
goes on iterations: you build, you hit something that needs a decision from the
fund manager, you email, you wait, you rebuild, you hit the next one.

The fund manager in the client interviews put it the same way from his side.
He said he is not sensitive to whether a turn took an hour or forty eight
hours. What he cares about is the count of turns, because that is the drag on
his time.

This tool finds every question in a single pass over the source data and
presents them as one consolidated list, so the loop runs once instead of four
times.

## What it does

1. Reads the source general ledger and the migration mapping workbook
2. Resolves every row through six crosswalks: legal entity, chart of accounts,
   investor, deal, position, and transaction type to account
3. Records every failure with a reason rather than dropping the row
4. Frames each failure as a decision a non technical fund manager can answer,
   with evidence, valuation and answer options attached
5. Separates decisions that block the migration from gaps that net to zero

Validated against the workbook's own verified output: both gaps listed in the
published `Mapping Gaps` sheet are detected, with row counts and amounts
agreeing exactly.

## Setup

Requires Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## The dataset

Not included in this repository. Place the hackathon dataset here:

```
data/
  02-investor-level-gl-to-loader/
    source/Investor-Level GL - Q2 activity - all entities (anonymised).xlsx
    output/Tranche 1 - reference and verified loader v4c (anonymised).xlsx
```

If it lives elsewhere, set `YLOOKUP_DATA_DIR` to the folder containing
`02-investor-level-gl-to-loader`.

## Run

```bash
./run.sh          # tests, then the full pipeline
./run.sh test     # tests only
./run.sh web      # browser based decision review, on http://localhost:8000
```

Reports are written to `out/`.

## Layout

```
src/recon/
  config.py      every file path and column name, in one place
  load.py        reads and validates the sheets, with caching
  normalise.py   safe key cleanup only, deliberately no fuzzy matching
  lookup.py      reusable exact match crosswalk that records misses
  mappings.py    the six concrete crosswalks
  gaps.py        aggregates unmatched rows, classifies by materiality
  decisions.py   frames gaps as questions a fund manager can answer

scripts/
  step1_inventory.py    proves the files load and the columns are present
  step2_mappings.py     resolution and gap reports
  step3_decisions.py    the consolidated decision report

tests/                  104 tests
```

## Design notes

**Nothing is silently dropped.** A row that does not resolve flows through with
empty target values and a status saying why. Money disappearing between two
systems is the failure this tool exists to catch.

**Ambiguity is reported, never guessed.** A mapping key that appears twice with
conflicting values is treated as unmatched. Taking the first match would
produce a total that balances and is wrong.

**Normalisation is deliberately narrow.** Whitespace and casing are cleaned.
Punctuation and legal suffixes are not, because a fund and its general partner
often differ only by a suffix and joining them would corrupt a total invisibly.

**Materiality separates urgent from real.** Several gaps total to floating
point residue. They are genuine gaps and are reported, but they do not block.
