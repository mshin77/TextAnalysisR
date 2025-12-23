# Analyze Contrastive Similarity (Alias)

Alias for
[`analyze_similarity_gaps`](https://mshin77.github.io/TextAnalysisR/reference/analyze_similarity_gaps.md).
Identifies unique items, missing content, and cross-category
opportunities based on similarity thresholds.

## Usage

``` r
analyze_contrastive_similarity(
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

A list containing unique_items, missing_items, cross_policy, and
summary_stats.

## See also

[`analyze_similarity_gaps`](https://mshin77.github.io/TextAnalysisR/reference/analyze_similarity_gaps.md)

Other ai:
[`call_ollama()`](https://mshin77.github.io/TextAnalysisR/reference/call_ollama.md),
[`check_ollama()`](https://mshin77.github.io/TextAnalysisR/reference/check_ollama.md),
[`create_label_selection_data()`](https://mshin77.github.io/TextAnalysisR/reference/create_label_selection_data.md),
[`format_label_candidates()`](https://mshin77.github.io/TextAnalysisR/reference/format_label_candidates.md),
[`generate_survey_items()`](https://mshin77.github.io/TextAnalysisR/reference/generate_survey_items.md),
[`generate_topic_content()`](https://mshin77.github.io/TextAnalysisR/reference/generate_topic_content.md),
[`generate_topic_labels_langgraph()`](https://mshin77.github.io/TextAnalysisR/reference/generate_topic_labels_langgraph.md),
[`get_content_type_prompt()`](https://mshin77.github.io/TextAnalysisR/reference/get_content_type_prompt.md),
[`get_content_type_user_template()`](https://mshin77.github.io/TextAnalysisR/reference/get_content_type_user_template.md),
[`get_recommended_ollama_model()`](https://mshin77.github.io/TextAnalysisR/reference/get_recommended_ollama_model.md),
[`list_ollama_models()`](https://mshin77.github.io/TextAnalysisR/reference/list_ollama_models.md),
[`run_rag_search()`](https://mshin77.github.io/TextAnalysisR/reference/run_rag_search.md),
[`validate_topic_labels_langgraph()`](https://mshin77.github.io/TextAnalysisR/reference/validate_topic_labels_langgraph.md)
