# data/golden/ - the curated evaluation set (Phase 2)

This folder stays EMPTY during Phase 1.

Later it will hold a small, carefully verified, double-labeled eval set
(see docs/LABELING_GUIDE.md, section 5). Rules for anything that lands here:

- **never** contains synthetic rows or public_weak rows
- every row is real, consented, scrubbed, and agreed upon by two labelers
- it is the ONLY data any future accuracy number may be computed on

This folder itself is NOT gitignored (it must stay empty of data files);
any *.parquet written here later is ignored by the global *.parquet rule.
