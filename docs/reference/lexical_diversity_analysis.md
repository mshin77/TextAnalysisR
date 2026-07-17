# Lexical Diversity Analysis

Calculates multiple lexical diversity metrics for a document-feature
matrix (DFM) or tokens object. Supports all quanteda.textstats measures
plus MTLD (Measure of Textual Lexical Diversity), which is the most
recommended measure according to McCarthy & Jarvis (2010) for being
independent of text length.

## Usage

``` r
lexical_diversity_analysis(x, measures = "all", texts = NULL, cache_key = NULL)
```

## Arguments

- x:

  A quanteda DFM or tokens object. Tokens object is preferred for
  accurate MTLD calculation since it preserves token order.

- measures:

  Character vector of measures to calculate. Default is "all" which
  includes: TTR, C, R, CTTR, U, S, K, I, D, Vm, Maas, MATTR, MSTTR,
  MTLD, and HDD. Most recommended: "MTLD", "MATTR", or "HDD" for
  length-independent measures.

- texts:

  Optional character vector of original texts. Required for MTLD
  calculation when using DFM input (since DFM loses token order).

- cache_key:

  Optional cache key (e.g., from digest::digest) for caching expensive
  calculations. Use the same cache_key to retrieve cached results.

## Value

A list containing:

- `lexical_diversity`: Data frame with per-document lexical diversity
  scores

- `summary_stats`: List of summary statistics (mean, median, sd) for
  each measure

## Details

MTLD (Measure of Textual Lexical Diversity) is calculated using the
algorithm from McCarthy & Jarvis (2010). It counts the number of
"factors" needed to reduce TTR below 0.72, then divides the number of
tokens by the number of factors. This provides a length-independent
measure of lexical diversity.

Important notes:

- For MTLD accuracy, pass a tokens object (not DFM) as input

- If using DFM, provide the 'texts' parameter for MTLD calculation

- MATTR and MSTTR window sizes are automatically adjusted for short
  documents

- Raw TTR falls mechanically as documents lengthen; compare TTR only
  across documents of similar length

- MTLD and MATTR are most reliable at 100+ tokens per document

- Results are cached when cache_key is provided for repeated analysis

## References

McCarthy, P. M., & Jarvis, S. (2010). MTLD, vocd-D, and HD-D: A
validation study of sophisticated approaches to lexical diversity
assessment. Behavior Research Methods, 42(2), 381-392.

## See also

[`calculate_text_readability()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_text_readability.md)
for grade-level / Flesch metrics on the same input;
[`calculate_lexical_dispersion()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_lexical_dispersion.md)
for term spread across documents;
[`plot_lexical_diversity_distribution()`](https://mshin77.github.io/TextAnalysisR/reference/plot_lexical_diversity_distribution.md)
to visualize

## Examples

``` r
# \donttest{
data(SpecialEduTech)
texts <- SpecialEduTech$abstract[1:10]
corp <- quanteda::corpus(texts)
toks <- quanteda::tokens(corp)
# Preferred: pass tokens object for accurate MTLD
lex_div <- lexical_diversity_analysis(toks, texts = texts)
# With caching for repeated analysis
cache_key <- digest::digest(texts)
lex_div <- lexical_diversity_analysis(toks, texts = texts, cache_key = cache_key)
# Alternative: pass DFM with texts for MTLD accuracy
dfm_obj <- quanteda::dfm(toks)
lex_div <- lexical_diversity_analysis(dfm_obj, texts = texts)
print(lex_div)
#> $lexical_diversity
#>    document       TTR         C        R     CTTR        U         S         K
#> 1     Doc 1 0.7831325 0.9446793 7.134677 5.044978 34.69006 0.9126943  92.90173
#> 2     Doc 2 0.9166667 0.9726212 4.490731 3.175426 50.41163 0.9138502  69.44444
#> 3     Doc 3 0.7812500 0.9287712 4.419417 3.125000 21.13121 0.8192855 156.25000
#> 4     Doc 4 0.9069767 0.9740406 5.947444 4.205478 62.92399 0.9463991  43.26663
#> 5     Doc 5 0.5757576 0.8798576 5.728716 4.050814 16.61059 0.8147581 155.08622
#> 6     Doc 6 0.4415205 0.8598873 8.165145 5.773629 18.08563 0.8376507 163.46910
#> 7     Doc 7 0.8823529 0.9681667 6.301260 4.455664 53.64094 0.9395388  53.82545
#> 8     Doc 8 0.4392157 0.8515204 7.013712 4.959443 16.20788 0.8169738 190.38831
#> 9     Doc 9 0.5882353 0.8919875 6.859943 4.850713 19.75270 0.8491609 149.22145
#> 10   Doc 10 0.8510638 0.9581138 5.834600 4.125685 39.91999 0.9167662  90.53871
#>             I           D         Vm      Maas     lgV0     lgeV0      MTLD
#> 1   51.524390 0.009403468 0.07716055 0.1697843 5.527252 12.726967 107.16222
#> 2   80.666667 0.007246377 0.05618332 0.1408428 5.776437 13.300738  80.64000
#> 3   27.173913 0.016129032 0.08291562 0.2175393 3.771555  8.684327  40.96000
#> 4  126.750000 0.004429679 0.04406190 0.1260642 7.028498 16.183714 172.57333
#> 5   16.747423 0.015666873 0.08980964 0.2453621 3.694732  8.507436  57.53291
#> 6   10.842130 0.016394848 0.11246497 0.2351436 4.268454  9.828479  71.37616
#> 7  101.250000 0.005490196 0.05261336 0.1365375 6.604754 15.208008 121.38000
#> 8    9.083273 0.019113787 0.11845602 0.2483916 3.908323  8.999247  64.44045
#> 9   19.277108 0.015032680 0.09886904 0.2250022 4.209816  9.693460  73.67553
#> 10  59.259259 0.009250694 0.07301004 0.1582722 5.594022 12.880713 123.70400
#>        MATTR     MSTTR       HDD Avg Sentence Length
#> 1  0.9319444 0.9166667 0.8622512            27.66667
#> 2  0.9166667 0.9166667        NA            24.00000
#> 3  0.7962963 0.7500000        NA            16.00000
#> 4  0.9520833 0.9583333 0.9318937            21.50000
#> 5  0.9144737 0.8854167 0.7757072            24.75000
#> 6  0.8797582 0.8541667 0.8173608            21.68750
#> 7  0.9553571 0.9583333 0.9005762            17.00000
#> 8  0.8915975 0.8977273 0.7917416            24.00000
#> 9  0.8727876 0.8750000 0.7919821            22.66667
#> 10 0.9392361 1.0000000 0.9007746            23.50000
#> 
#> $summary_stats
#> $summary_stats$n_documents
#> [1] 10
#> 
#> $summary_stats$measures_calculated
#>  [1] "TTR"                 "C"                   "R"                  
#>  [4] "CTTR"                "U"                   "S"                  
#>  [7] "K"                   "I"                   "D"                  
#> [10] "Vm"                  "Maas"                "lgV0"               
#> [13] "lgeV0"               "MTLD"                "MATTR"              
#> [16] "MSTTR"               "HDD"                 "Avg Sentence Length"
#> 
#> $summary_stats$TTR_mean
#> [1] 0.7166172
#> 
#> $summary_stats$TTR_median
#> [1] 0.7821913
#> 
#> $summary_stats$TTR_sd
#> [1] 0.1883719
#> 
#> $summary_stats$C_mean
#> [1] 0.9229646
#> 
#> $summary_stats$C_median
#> [1] 0.9367253
#> 
#> $summary_stats$C_sd
#> [1] 0.04802694
#> 
#> $summary_stats$R_mean
#> [1] 6.189565
#> 
#> $summary_stats$R_median
#> [1] 6.124352
#> 
#> $summary_stats$R_sd
#> [1] 1.171595
#> 
#> $summary_stats$CTTR_mean
#> [1] 4.376683
#> 
#> $summary_stats$CTTR_median
#> [1] 4.330571
#> 
#> $summary_stats$CTTR_sd
#> [1] 0.8284429
#> 
#> $summary_stats$U_mean
#> [1] 33.33746
#> 
#> $summary_stats$U_median
#> [1] 27.91063
#> 
#> $summary_stats$U_sd
#> [1] 17.52347
#> 
#> $summary_stats$S_mean
#> [1] 0.8767078
#> 
#> $summary_stats$S_median
#> [1] 0.8809276
#> 
#> $summary_stats$S_sd
#> [1] 0.05382214
#> 
#> $summary_stats$K_mean
#> [1] 116.4392
#> 
#> $summary_stats$K_median
#> [1] 121.0616
#> 
#> $summary_stats$K_sd
#> [1] 52.2191
#> 
#> $summary_stats$I_mean
#> [1] 50.25742
#> 
#> $summary_stats$I_median
#> [1] 39.34915
#> 
#> $summary_stats$I_sd
#> [1] 41.26223
#> 
#> $summary_stats$D_mean
#> [1] 0.01181576
#> 
#> $summary_stats$D_median
#> [1] 0.01221807
#> 
#> $summary_stats$D_sd
#> [1] 0.005226621
#> 
#> $summary_stats$Vm_mean
#> [1] 0.08055445
#> 
#> $summary_stats$Vm_median
#> [1] 0.08003808
#> 
#> $summary_stats$Vm_sd
#> [1] 0.02506939
#> 
#> $summary_stats$Maas_mean
#> [1] 0.190294
#> 
#> $summary_stats$Maas_median
#> [1] 0.1936618
#> 
#> $summary_stats$Maas_sd
#> [1] 0.04861752
#> 
#> $summary_stats$lgV0_mean
#> [1] 5.038384
#> 
#> $summary_stats$lgV0_median
#> [1] 4.897853
#> 
#> $summary_stats$lgV0_sd
#> [1] 1.223525
#> 
#> $summary_stats$lgeV0_mean
#> [1] 11.60131
#> 
#> $summary_stats$lgeV0_median
#> [1] 11.27772
#> 
#> $summary_stats$lgeV0_sd
#> [1] 2.817271
#> 
#> $summary_stats$MTLD_mean
#> [1] 91.34446
#> 
#> $summary_stats$MTLD_median
#> [1] 77.15776
#> 
#> $summary_stats$MTLD_sd
#> [1] 39.48101
#> 
#> $summary_stats$MATTR_mean
#> [1] 0.9050201
#> 
#> $summary_stats$MATTR_median
#> [1] 0.9155702
#> 
#> $summary_stats$MATTR_sd
#> [1] 0.0477814
#> 
#> $summary_stats$MSTTR_mean
#> [1] 0.9012311
#> 
#> $summary_stats$MSTTR_median
#> [1] 0.907197
#> 
#> $summary_stats$MSTTR_sd
#> [1] 0.06895207
#> 
#> $summary_stats$HDD_mean
#> [1] 0.8465359
#> 
#> $summary_stats$HDD_median
#> [1] 0.839806
#> 
#> $summary_stats$HDD_sd
#> [1] 0.06004944
#> 
#> $summary_stats$`Avg Sentence Length_mean`
#> [1] 22.27708
#> 
#> $summary_stats$`Avg Sentence Length_median`
#> [1] 23.08333
#> 
#> $summary_stats$`Avg Sentence Length_sd`
#> [1] 3.511061
#> 
#> 
# }
```
