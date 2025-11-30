# Analyze and Visualize Word Correlation Networks

This function creates a word correlation network based on a
document-feature matrix (dfm).

## Usage

``` r
plot_correlation_network(
  dfm_object,
  doc_var = NULL,
  common_term_n = 130,
  corr_n = 0.4,
  top_node_n = 40,
  nrows = 1,
  height = 1000,
  width = 900,
  use_category_specific = FALSE,
  category_params = NULL
)
```

## Arguments

- dfm_object:

  A quanteda document-feature matrix (dfm).

- doc_var:

  A document-level metadata variable (default: NULL).

- common_term_n:

  Minimum number of common terms for filtering terms (default: 30).

- corr_n:

  Minimum correlation value for filtering terms (default: 0.4).

- top_node_n:

  Number of top nodes to display (default: 40).

- nrows:

  Number of rows to display in the table (default: 1).

- height:

  The height of the resulting Plotly plot, in pixels (default: 1000).

- width:

  The width of the resulting Plotly plot, in pixels (default: 900).

- use_category_specific:

  Logical; if TRUE, uses category-specific parameters (default: FALSE).

- category_params:

  A named list of parameters for each category level (default: NULL).

## Value

A list containing the Plotly plot, a data frame of the network layout,
and the igraph graph object.

## Examples

``` r
if (interactive()) {
  mydata <- TextAnalysisR::SpecialEduTech

  united_tbl <- TextAnalysisR::unite_cols(
    mydata,
    listed_vars = c("title", "keyword", "abstract")
  )

  tokens <- TextAnalysisR::prep_texts(united_tbl, text_field = "united_texts")

  dfm_object <- quanteda::dfm(tokens)

  # Overall
  word_correlation_network_results <- TextAnalysisR::plot_correlation_network(
                                        dfm_object,
                                        doc_var = "reference_type",
                                        common_term_n = 30,
                                        corr_n = 0.4,
                                        top_node_n = 0,
                                        nrows = 1,
                                        height = 800,
                                        width = 900)

  print(word_correlation_network_results$plot)
  print(word_correlation_network_results$table)
  print(word_correlation_network_results$summary)

  # Journal article
 category_params <- list(
   "journal_article" = list(common_term_n = 30, corr_n = 0.4, top_node_n = 20),
   "thesis" = list(common_term_n = 20, corr_n = 0.4, top_node_n = 20)
)

 word_correlation_category <- TextAnalysisR::plot_correlation_network(
   dfm_object,
   doc_var = "reference_type",
   use_category_specific = TRUE,
   category_params = category_params)

 print(word_correlation_category$journal_article$plot)
 print(word_correlation_category$journal_article$table)
 print(word_correlation_category$journal_article$summary)

 # Thesis
 print(word_correlation_category$thesis$plot)
 print(word_correlation_category$thesis$table)
 print(word_correlation_category$thesis$summary)

}
```
