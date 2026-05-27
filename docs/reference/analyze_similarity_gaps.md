# Analyze Similarity Gaps Between Categories

Identifies unique items, missing content, and cross-category learning
opportunities based on similarity thresholds. Useful for gap analysis in
policy documents, topic comparisons, or any cross-category similarity
study.

## Usage

``` r
analyze_similarity_gaps(
  similarity_data,
  ref_var = "ref_id",
  other_var = "other_id",
  similarity_var = "similarity",
  category_var = "other_category",
  ref_label_var = NULL,
  other_label_var = NULL,
  unique_threshold = 0.6,
  cross_policy_min = 0.6,
  cross_policy_max = 0.8
)
```

## Arguments

- similarity_data:

  A data frame with cross-category similarities, containing:

  ref_var

  :   Reference item identifier

  other_var

  :   Comparison item identifier

  similarity_var

  :   Similarity score

  category_var

  :   Category of comparison item

- ref_var:

  Name of column with reference item IDs (default: "ref_id").

- other_var:

  Name of column with comparison item IDs (default: "other_id").

- similarity_var:

  Name of column with similarity values (default: "similarity").

- category_var:

  Name of column with category information (default: "other_category").

- ref_label_var:

  Optional column with reference item labels (for output).

- other_label_var:

  Optional column with comparison item labels (for output).

- unique_threshold:

  Threshold below which reference items are considered unique (default:
  0.6).

- cross_policy_min:

  Minimum similarity for cross-policy opportunities (default: 0.6).

- cross_policy_max:

  Maximum similarity for cross-policy opportunities (default: 0.8).

## Value

A list containing:

- unique_items:

  Data frame of reference items with low similarity (unique content)

- missing_items:

  Data frame of comparison items with low similarity (content gaps)

- cross_policy:

  Data frame of items with moderate similarity (learning opportunities)

- summary_stats:

  Summary statistics by category

## Examples

``` r
# \donttest{
articles <- TextAnalysisR::SpecialEduTech[1:6, ]
articles$display_name <- paste0("d", seq_len(nrow(articles)))
term_matrix <- as.matrix(quanteda::dfm(quanteda::tokens(articles$abstract)))
normalized_matrix <- term_matrix / sqrt(rowSums(term_matrix ^ 2))
similarity_matrix <- normalized_matrix %*% t(normalized_matrix)
dimnames(similarity_matrix) <- list(articles$display_name, articles$display_name)
cross_similarities <- extract_cross_category_similarities(
  similarity_matrix  = similarity_matrix,
  docs_data          = articles,
  reference_category = "thesis",
  compare_categories = "journal_article",
  category_var       = "reference_type",
  id_var             = "display_name"
)
gap_analysis <- analyze_similarity_gaps(
  similarity_data = cross_similarities,
  ref_var = "ref_id",
  other_var = "other_id",
  similarity_var = "similarity",
  category_var = "other_category",
  unique_threshold = 0.6
)
print(gap_analysis$summary_stats)
#> # A tibble: 1 × 7
#>   other_category  mean_similarity median_similarity sd_similarity min_similarity
#>   <fct>                     <dbl>             <dbl>         <dbl>          <dbl>
#> 1 journal_article           0.386             0.336         0.146          0.184
#> # ℹ 2 more variables: max_similarity <dbl>, n_pairs <int>
# }
```
