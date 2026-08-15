# Qualitative Coding

Qualitative coding assigns codes from a codebook to units of text, then
measures how far coders agree. AI supplies suggestions; a human
confirms, corrects, or rejects each one, and only reviewed rows are
exported. Nothing is coded silently.

The sections below follow the app’s Qualitative Coding tabs in order.
Agreement and merging run live on constructed data. The coding steps
call an AI provider, so they are shown as code rather than run here.

``` r

library(TextAnalysisR)
packageVersion("TextAnalysisR")
```

    ## [1] '0.1.4.9000'

## 1. Codebook

A codebook is a data frame with `code` and `definition` columns, plus an
optional `example`. Definitions carry most of the weight: a code without
one gives the model nothing to match against.

``` r

codebook <- tibble::tibble(
  code = c("access", "instruction", "assessment"),
  definition = c(
    "Availability of technology, materials, or services to students",
    "Teaching methods, strategies, or curriculum delivery",
    "Measurement of student performance or progress"
  ),
  example = c(
    "tablets were provided to every student",
    "the teacher modeled each step before practice",
    "progress was measured with weekly probes"
  )
)
codebook
```

    ## # A tibble: 3 × 3
    ##   code        definition                                                 example
    ##   <chr>       <chr>                                                      <chr>  
    ## 1 access      Availability of technology, materials, or services to stu… tablet…
    ## 2 instruction Teaching methods, strategies, or curriculum delivery       the te…
    ## 3 assessment  Measurement of student performance or progress             progre…

Topic labels make reasonable starting codes. The mapping is many-to-many
rather than one-to-one, so treat seeded codes as a draft to edit rather
than a finished scheme.

## 2. AI coding

[`apply_codes()`](https://mshin77.github.io/TextAnalysisR/reference/apply_codes.md)
splits each document into units, sends each unit with the codebook, and
returns one row per unit-code pair.

``` r

texts <- SpecialEduTech$abstract[1:20]
names(texts) <- paste0("doc", seq_along(texts))

suggestions <- apply_codes(
  texts,
  codebook,
  unit = "paragraph",
  max_codes = 3,
  provider = "gemini"
)
```

Three arguments decide what the output means.

| Argument | Effect |
|----|----|
| `unit` | Splits on paragraphs, sentences, or whole documents. Paragraph is the usual unit of analysis. |
| `max_codes` | Allows a unit to carry none, one, or several codes. A unit that fits nothing returns `NA`. |
| `provider` | Selects OpenAI or Gemini. `auto` picks whichever key is available. |

Output columns `start` and `end` give character offsets of the unit
inside its document, so a suggestion can be traced back to the exact
text it came from.

## 3. Review

Suggestions are not results. In the app each row carries a status of
pending, accepted, edited, or rejected, and only accepted and edited
rows reach the export. Working from R, filter on confidence and read the
units before keeping anything.

``` r

accepted <- suggestions[!is.na(suggestions$code) & suggestions$confidence > 0.7, ]
accepted$coder <- "coder1"
```

Low-temperature settings do not guarantee stable output.
[`code_retest()`](https://mshin77.github.io/TextAnalysisR/reference/code_retest.md)
codes the same sample more than once and reports how often the runs
agree, next to a baseline from shuffled labels. Retest agreement close
to that baseline means the codes are not reproducible enough to analyze.

``` r

stability <- code_retest(texts, codebook, n_runs = 2, sample_n = 20)
stability$summary
```

## 4. Agreement

[`code_agreement()`](https://mshin77.github.io/TextAnalysisR/reference/code_agreement.md)
takes assignments from two or more coders and reports chance-corrected
agreement.

``` r

assignments <- tibble::tibble(
  doc_id = rep(paste0("doc", 1:8), each = 2),
  code   = c("access", "access", "access", "access", "access", "access",
             "access", "access", "access", "access", "access", "access",
             "access", "instruction", "instruction", "access"),
  coder  = rep(c("c1", "c2"), times = 8)
)

agreement <- code_agreement(assignments)
agreement$overall
```

    ## # A tibble: 5 × 3
    ##   metric  estimate     n
    ##   <chr>      <dbl> <int>
    ## 1 percent   0.75       8
    ## 2 pabak     0.5        8
    ## 3 ac1       0.68       8
    ## 4 kappa    -0.143      8
    ## 5 alpha    -0.0714     8

Percent agreement is high here while kappa is near zero. That gap is the
kappa paradox: when one code dominates, expected agreement approaches
observed agreement and kappa collapses even though the coders rarely
disagree. Gwet’s AC1 and PABAK stay interpretable under that skew, which
is why all five statistics are reported together rather than kappa
alone.

Disagreements are listed separately so the underlying units can be
re-read.

``` r

agreement$disagree
```

    ## # A tibble: 2 × 3
    ##   doc_id c1          c2         
    ##   <chr>  <chr>       <chr>      
    ## 1 doc7   access      instruction
    ## 2 doc8   instruction access

Per-code agreement shows which codes carry the disagreement.

``` r

agreement$by_code
```

    ## # A tibble: 10 × 4
    ##    code        metric  estimate     n
    ##    <chr>       <chr>      <dbl> <int>
    ##  1 access      percent   0.75       8
    ##  2 access      pabak     0.5        8
    ##  3 access      ac1       0.68       8
    ##  4 access      kappa    -0.143      8
    ##  5 access      alpha    -0.0714     8
    ##  6 instruction percent   0.75       8
    ##  7 instruction pabak     0.5        8
    ##  8 instruction ac1       0.68       8
    ##  9 instruction kappa    -0.143      8
    ## 10 instruction alpha    -0.0714     8

### Coders who segment differently

Grid alignment assumes coders share units. When each coder marks their
own spans, boundaries rarely match, and pivoting on unit identity would
compare rows that describe different text. Coverage alignment instead
reports, for each ordered coder pair, the share of one coder’s spans
that overlap a same-code span from the other.

``` r

spans <- tibble::tibble(
  doc_id = c("doc1", "doc1", "doc1"),
  coder  = c("c1", "c1", "c2"),
  code   = c("access", "instruction", "access"),
  start  = c(1, 200, 50),
  end    = c(100, 300, 80)
)

code_agreement(spans, align = "coverage")$overall
```

    ## # A tibble: 2 × 5
    ##   coder other metric   estimate     n
    ##   <chr> <chr> <chr>       <dbl> <int>
    ## 1 c1    c2    coverage      0.5     2
    ## 2 c2    c1    coverage      1       1

Coverage is directional. One coder marking broad spans that contain
another’s narrow ones scores differently in each direction, and that
asymmetry is informative rather than a defect.

## Combining coder files

Each coder exports their accepted rows;
[`merge_codes()`](https://mshin77.github.io/TextAnalysisR/reference/merge_codes.md)
binds the files into one table for
[`code_agreement()`](https://mshin77.github.io/TextAnalysisR/reference/code_agreement.md).

``` r

combined <- merge_codes(c("coder1.rds", "coder2.rds"))
code_agreement(combined)$overall
```

## Reporting

Report the unit of analysis, how many units were coded, which agreement
statistic was used and why, and the share of AI suggestions a human
changed. The last figure is what separates assisted coding from
automated labeling, and it is only available when review status is
recorded.
