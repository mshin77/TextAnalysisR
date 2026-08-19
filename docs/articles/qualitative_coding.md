# Qualitative Coding

Codes from a codebook are assigned to units of text, then inter-coder
agreement is measured. AI supplies suggestions; a human confirms,
corrects, or rejects each one, and only reviewed rows are exported.

Steps that call an AI provider are shown as code rather than run. The
rest run on synthetic data.

``` r

library(TextAnalysisR)
```

## Codebook

A data frame with `code` and `definition`, plus an optional `example`.

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

## AI Coding

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

| Argument | Effect |
|----|----|
| `unit` | Splits on sentences, paragraphs, or whole documents. Paragraph is the usual unit of analysis. |
| `max_codes` | Allows a unit to carry none, one, or several codes. A unit that fits nothing returns `NA`. |
| `provider` | Selects OpenAI or Gemini. `auto` picks whichever key is available. |

Columns `start` and `end` give character offsets of the unit inside its
document.

## Review

Only accepted and edited rows reach the export.

``` r

accepted <- suggestions[!is.na(suggestions$code) & suggestions$confidence > 0.7, ]
accepted$coder <- "coder1"
```

[`code_retest()`](https://mshin77.github.io/TextAnalysisR/reference/code_retest.md)
codes the same sample more than once and reports how often the runs
agree, next to a baseline from shuffled labels.

``` r

stability <- code_retest(texts, codebook, n_runs = 2, sample_n = 20)
stability$summary
```

[`uncoded_units()`](https://mshin77.github.io/TextAnalysisR/reference/uncoded_units.md)
returns the units that received no code, with the text sliced from the
recorded offsets.

``` r

# synthetic units, paraphrased from the corpus
suggestions <- tibble::tibble(
  doc_id  = c("d1", "d1", "d2"),
  unit_id = c("d1.1", "d1.2", "d2.1"),
  start   = c(1L, 68L, 1L),
  end     = c(66L, 128L, 74L),
  code    = c("access", NA_character_, NA_character_)
)

texts <- c(
  d1 = paste("Text-to-speech tools give students access to grade-level readings.",
             "Teachers report growing confidence after the training series."),
  d2 = "The review summarizes methodological features across the included studies."
)

uncoded_units(suggestions, texts)
```

    ## # A tibble: 2 × 5
    ##   doc_id unit_id start   end unit_text                                          
    ##   <chr>  <chr>   <int> <int> <chr>                                              
    ## 1 d1     d1.2       68   128 Teachers report growing confidence after the train…
    ## 2 d2     d2.1        1    74 The review summarizes methodological features acro…

## Agreement

[`code_agreement()`](https://mshin77.github.io/TextAnalysisR/reference/code_agreement.md)
takes assignments from two or more coders.

``` r

# synthetic assignments
assignments <- tibble::tibble(
  doc_id = rep(paste0("doc", 1:10), each = 2),
  code   = c("access", "access", "access", "access", "access", "access",
             "instruction", "instruction", "instruction", "instruction",
             "instruction", "assessment", "assessment", "assessment",
             "assessment", "assessment", "assessment", "access",
             "access", "access"),
  coder  = rep(c("c1", "c2"), times = 10)
)

agreement <- code_agreement(assignments)
agreement$overall
```

    ## # A tibble: 5 × 3
    ##   metric  estimate     n
    ##   <chr>      <dbl> <int>
    ## 1 percent    0.8      10
    ## 2 pabak      0.6      10
    ## 3 ac1        0.705    10
    ## 4 kappa      0.692    10
    ## 5 alpha      0.705    10

`n` is how many units each statistic used.

| `metric` | Reads as |
|----|----|
| `percent` | Raw share of units both coders labeled identically. No correction for chance. |
| `pabak` | Prevalence-adjusted bias-adjusted kappa: percent agreement rescaled to the -1 to 1 range. |
| `ac1` | Gwet’s agreement coefficient. Chance-corrected, stable when one code dominates. |
| `kappa` | Cohen’s kappa for two coders, Fleiss’ for more. Chance-corrected against observed marginals. |
| `alpha` | Krippendorff’s alpha. Chance-corrected, handles missing units. |

Which units the coders differed on:

``` r

agreement$disagree
```

    ## # A tibble: 2 × 3
    ##   doc_id c1          c2        
    ##   <chr>  <chr>       <chr>     
    ## 1 doc6   instruction assessment
    ## 2 doc9   assessment  access

Which codes carry those disagreements:

``` r

agreement$by_code
```

    ## # A tibble: 15 × 4
    ##    code        metric  estimate     n
    ##    <chr>       <chr>      <dbl> <int>
    ##  1 access      percent    0.9      10
    ##  2 access      pabak      0.8      10
    ##  3 access      ac1        0.802    10
    ##  4 access      kappa      0.8      10
    ##  5 access      alpha      0.808    10
    ##  6 assessment  percent    0.8      10
    ##  7 assessment  pabak      0.6      10
    ##  8 assessment  ac1        0.655    10
    ##  9 assessment  kappa      0.524    10
    ## 10 assessment  alpha      0.548    10
    ## 11 instruction percent    0.9      10
    ## 12 instruction pabak      0.8      10
    ## 13 instruction ac1        0.84     10
    ## 14 instruction kappa      0.737    10
    ## 15 instruction alpha      0.747    10

Each code is scored as a yes/no indicator, so a two-code corpus returns
the same figures twice, one indicator being the other’s complement.

Chance-corrected statistics turn negative when observed agreement falls
below what the marginals predict, which happens when nearly every unit
carries the same code:

``` r

# synthetic assignments, one code dominating
skewed <- tibble::tibble(
  doc_id = rep(paste0("doc", 1:8), each = 2),
  code   = c("access", "access", "access", "access", "access", "access",
             "access", "access", "access", "access", "access", "access",
             "access", "instruction", "instruction", "access"),
  coder  = rep(c("c1", "c2"), times = 8)
)

code_agreement(skewed, by_code = FALSE)$overall
```

    ## # A tibble: 5 × 3
    ##   metric  estimate     n
    ##   <chr>      <dbl> <int>
    ## 1 percent   0.75       8
    ## 2 pabak     0.5        8
    ## 3 ac1       0.68       8
    ## 4 kappa    -0.143      8
    ## 5 alpha    -0.0714     8

Chance alone accounts for 0.78 against an observed 0.75, so `kappa`
lands at -0.14. Only `percent` is bounded at zero.

`codebook_authors` adds an `independent` table computed among the
remaining coders alone.

``` r

code_agreement(assignments, codebook_authors = "c1")$independent
```

Everything above assumes coders shared units. When each coder highlights
their own stretch of text, `align = "coverage"` compares the highlights
instead. `start` and `end` are character positions.

``` r

# synthetic spans
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

`c2`’s span sits inside `c1`’s first one. One of `c1`’s two spans was
matched, so 0.50; `c2`’s only span was matched, so 1.00. Both directions
are reported.

## Combining Coder Files

``` r

combined <- merge_codes(c("coder1.csv", "coder2.xlsx"))
code_agreement(combined)$overall
```
