# Extract Cross-Category Similarities from Full Similarity Matrix

Given a full similarity matrix and category information, extracts
pairwise similarities between a reference category and other categories
into a long-format data frame suitable for visualization and analysis.

## Usage

``` r
extract_cross_category_similarities(
  similarity_matrix,
  docs_data,
  reference_category,
  compare_categories = NULL,
  category_var = "category",
  id_var = "display_name",
  name_var = NULL
)
```

## Arguments

- similarity_matrix:

  A square similarity matrix (n x n).

- docs_data:

  A data frame containing document metadata with at least:

  category_var

  :   Column indicating category membership

  id_var

  :   Column with unique document identifiers

- reference_category:

  Character string specifying the reference category to compare against.

- compare_categories:

  Character vector of categories to compare with the reference. If NULL,
  compares with all categories except reference.

- category_var:

  Name of the column containing category information (default:
  "category").

- id_var:

  Name of the column containing document IDs (default: "display_name").

- name_var:

  Optional name of column with display names (default: NULL, uses
  id_var).

## Value

A data frame with columns:

- ref_id:

  Reference document ID

- ref_name:

  Reference document name (if name_var provided)

- other_id:

  Comparison document ID

- other_name:

  Comparison document name (if name_var provided)

- other_category:

  Category of comparison document

- similarity:

  Cosine similarity value

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
  id_var             = "display_name",
  name_var           = "title"
)
# }
```
