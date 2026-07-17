# Calculate Document Similarity

Calculates similarity between documents using traditional NLP methods or
modern embedding-based approaches. Metrics are automatically computed
unless disabled.

## Usage

``` r
calculate_document_similarity(
  texts,
  document_feature_type = "words",
  semantic_ngram_range = 2,
  similarity_method = "cosine",
  use_embeddings = FALSE,
  embedding_model = "all-MiniLM-L6-v2",
  calculate_metrics = TRUE,
  verbose = TRUE
)
```

## Arguments

- texts:

  A character vector of texts to compare.

- document_feature_type:

  Feature extraction type: "words", "ngrams", or "embeddings".

- semantic_ngram_range:

  Integer, n-gram range for ngram features (default: 2).

- similarity_method:

  Similarity calculation method: "cosine", "jaccard", "euclidean",
  "manhattan".

- use_embeddings:

  Logical, use embedding-based similarity (default: FALSE).

- embedding_model:

  Sentence transformer model name (default: "all-MiniLM-L6-v2").

- calculate_metrics:

  Logical, compute similarity metrics (default: TRUE).

- verbose:

  Logical, if TRUE, prints progress messages.

## Value

A list containing:

- similarity_matrix:

  N x N similarity matrix

- feature_matrix:

  Document feature matrix used for calculation

- method_info:

  Information about the method used

- metrics:

  Similarity metrics (if calculate_metrics = TRUE)

- execution_time:

  Time taken for analysis

## Examples

``` r
# \donttest{
  data(SpecialEduTech)
  texts <- SpecialEduTech$abstract[1:5]

  result <- calculate_document_similarity(
    texts = texts,
    document_feature_type = "words",
    similarity_method = "cosine"
  )
#> Starting document similarity analysis...
#> Feature type: words
#> Similarity method: cosine
#> Use embeddings: FALSE
#> Step 1: Generating feature matrix...
#> Using word-based features...
#> Step 2: Calculating similarity matrix...
#> Step 3: Calculating metrics...
#> Document similarity analysis completed in 0.23 seconds
#> Documents analyzed: 5
#> Feature dimensions: 30

  print(result$similarity_matrix)
#>        docs
#> docs         text1     text2      text3     text4      text5
#>   text1 1.00000000 0.1961161 0.17541160 0.1132277 0.05883484
#>   text2 0.19611614 1.0000000 0.33541020 0.0000000 0.05000000
#>   text3 0.17541160 0.3354102 1.00000000 0.0000000 0.02236068
#>   text4 0.11322770 0.0000000 0.00000000 1.0000000 0.51961524
#>   text5 0.05883484 0.0500000 0.02236068 0.5196152 1.00000000
  print(result$metrics)
#> $n_docs
#> [1] 5
#> 
#> $mean_similarity
#> [1] 0.1471
#> 
#> $median_similarity
#> [1] 0.086
#> 
#> $std_similarity
#> [1] 0.1637
#> 
#> $min_similarity
#> [1] 0
#> 
#> $max_similarity
#> [1] 0.5196
#> 
#> $similarity_range
#> [1] "0 to 0.52"
#> 
#> $sparsity
#> [1] 0.5
#> 
#> $connectivity
#> [1] 0.2
#> 
#> $skewness
#> [1] 1.1952
#> 
#> $kurtosis
#> [1] 3.3699
#> 
#> $silhouette_score
#> [1] NA
#> 
#> $modularity
#> [1] NA
#> 
#> $method
#> [1] "traditional_cosine"
#> 
#> $model_name
#> [1] "N/A"
#> 
# }
```
