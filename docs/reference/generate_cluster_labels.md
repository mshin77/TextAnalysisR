# Generate Cluster Labels with AI

Generates descriptive labels for clusters using either Ollama (local,
default) or OpenAI's API. When running locally, Ollama is preferred for
privacy and cost-free operation.

## Usage

``` r
generate_cluster_labels(
  cluster_keywords,
  provider = "auto",
  model = NULL,
  temperature = 0.3,
  max_tokens = 50,
  verbose = TRUE
)
```

## Arguments

- cluster_keywords:

  List of keywords for each cluster.

- provider:

  AI provider to use: "auto" (default), "ollama", or "openai". "auto"
  will use Ollama if available, otherwise OpenAI.

- model:

  Model name. For Ollama (default: "phi3:mini"). For OpenAI (default:
  "gpt-3.5-turbo").

- temperature:

  Temperature parameter (default: 0.3).

- max_tokens:

  Maximum tokens for response (default: 50).

- verbose:

  Logical, if TRUE, prints progress messages.

## Value

A list of generated labels.

## See also

Other semantic:
[`analyze_document_clustering()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_document_clustering.md),
[`analyze_similarity_gaps()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_similarity_gaps.md),
[`calculate_clustering_metrics()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_clustering_metrics.md),
[`calculate_cross_similarity()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_cross_similarity.md),
[`calculate_document_similarity()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_document_similarity.md),
[`calculate_similarity_robust()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_similarity_robust.md),
[`cluster_embeddings()`](https://mshin77.github.io/TextAnalysisR/reference/cluster_embeddings.md),
[`cross_analysis_validation()`](https://mshin77.github.io/TextAnalysisR/reference/cross_analysis_validation.md),
[`export_document_clustering()`](https://mshin77.github.io/TextAnalysisR/reference/export_document_clustering.md),
[`extract_cross_category_similarities()`](https://mshin77.github.io/TextAnalysisR/reference/extract_cross_category_similarities.md),
[`fit_semantic_model()`](https://mshin77.github.io/TextAnalysisR/reference/fit_semantic_model.md),
[`generate_cluster_labels_auto()`](https://mshin77.github.io/TextAnalysisR/reference/generate_cluster_labels_auto.md),
[`generate_embeddings()`](https://mshin77.github.io/TextAnalysisR/reference/generate_embeddings.md),
[`reduce_dimensions()`](https://mshin77.github.io/TextAnalysisR/reference/reduce_dimensions.md),
[`semantic_document_clustering()`](https://mshin77.github.io/TextAnalysisR/reference/semantic_document_clustering.md),
[`semantic_similarity_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/semantic_similarity_analysis.md),
[`temporal_semantic_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/temporal_semantic_analysis.md),
[`validate_cross_models()`](https://mshin77.github.io/TextAnalysisR/reference/validate_cross_models.md)

## Examples

``` r
if (FALSE) { # \dontrun{
keywords <- list("1" = c("machine", "learning", "neural"), "2" = c("data", "analysis"))
labels_ollama <- generate_cluster_labels(keywords, provider = "ollama")
labels_openai <- generate_cluster_labels(keywords, provider = "openai")
} # }
```
