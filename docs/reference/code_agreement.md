# Inter-Coder Agreement From Code Assignments

Chance-corrected agreement between two or more coders' code assignments.
Reports Krippendorff's alpha and Cohen's/Fleiss' kappa (via irr) plus
percent agreement, Gwet's AC1, and PABAK computed inline. AC1 and PABAK
stay stable under skewed code prevalence, where kappa collapses (the
kappa paradox).

With `align = "grid"` (default), coders are assumed to share units: rows
pivot on `unit_id` when present, else `doc_id`. When a coder assigned
more than one code to a unit, the highest-confidence code is kept (first
row when no `confidence` column exists); per-code multi-label agreement
is in `by_code`. With `align = "coverage"`, coders may have different
span boundaries: for each ordered coder pair, the proportion of one
coder's spans that overlap a same-code span from the other is reported
instead (chance-corrected metrics do not apply across unaligned units,
so `metrics` and `units` are ignored).

## Usage

``` r
code_agreement(
  assignments,
  metrics = c("alpha", "kappa", "ac1", "pabak", "percent"),
  units = c("intersection", "union"),
  by_code = TRUE,
  align = c("grid", "coverage"),
  codebook_authors = NULL
)
```

## Arguments

- assignments:

  Data frame with `doc_id`, `code`, and `coder` columns. `unit_id` and
  `confidence` are used when present. Coverage additionally requires
  `start` and `end`.

- metrics:

  Metrics to report: any of "alpha", "kappa", "ac1", "pabak", "percent"
  (all by default). Grid alignment only. `pabak` is defined for two
  coders and returns `NA` for more.

- units:

  "intersection" (default, units coded by every coder) or "union"
  (uncoded units count as missing). Grid alignment only.

- by_code:

  Logical; also report per-code agreement.

- align:

  "grid" (default) or "coverage".

- codebook_authors:

  Character vector of `coder` values that wrote or revised the codebook.
  When supplied, metrics are also computed among the remaining coders
  alone. Agreement with a coder who shaped the coding frame reflects
  that shared calibration, so the two figures answer different
  questions. Ignored when `align = "coverage"`.

## Value

A list with `overall`, `by_code` (NULL when `by_code` is FALSE),
`disagree`, and `independent` (NULL unless `codebook_authors` is
supplied and at least two other coders remain). For grid alignment these
are the metric, per-code, and disagreement tables; for coverage they
hold per-coder-pair span coverage and the uncovered spans. In the metric
tables `n` counts the units each statistic was computed on: units
carrying a code from every coder for `percent`, `pabak`, and `kappa`;
units carrying a code from at least two for `ac1` and `alpha`. `n` is
`NA` wherever the estimate is.

## See also

[`apply_codes()`](https://mshin77.github.io/TextAnalysisR/reference/apply_codes.md)
to produce assignments;
[`merge_codes()`](https://mshin77.github.io/TextAnalysisR/reference/merge_codes.md)
to combine coders;
[`code_retest()`](https://mshin77.github.io/TextAnalysisR/reference/code_retest.md)
for AI stability.
