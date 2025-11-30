#' @keywords internal
"_PACKAGE"

## usethis namespace: start
#' @importFrom tibble tibble
#' @importFrom stats as.dist kmeans median runif sd setNames var
## usethis namespace: end
NULL

#' @title TextAnalysisR: Comprehensive Text Analysis Package
#'
#' @description
#' TextAnalysisR provides a comprehensive suite of text analysis tools including:
#' - Text preprocessing and corpus preparation
#' - Structural topic modeling
#' - Word networks and co-occurrence analysis
#' - Semantic similarity analysis
#' - Dimensionality reduction and clustering
#' - Interactive visualizations
#' - Shiny web application interface
#'
#' The package is designed to work both as a standalone R package and as the backend
#' for a comprehensive Shiny web application, providing flexibility for different user needs.
#'
#' @section Main Functions:
#' 
#' **Data Processing:**
#' - `import_files()`: Process various file formats
#' - `unite_cols()`: Combine text columns
#' - `prep_texts()`: Enhanced text preprocessing with validation
#'
#' **Topic Modeling:**
#' - `find_optimal_k()`: Find optimal number of topics
#' - `get_topic_terms()`: Extract top terms per topic
#' - `generate_topic_labels()`: Auto-generate topic labels using OpenAI
#' - `plot_word_probability()`: Visualize topic terms
#' - `plot_topic_probability()`: Visualize topic prevalence
#'
#' **Network Analysis:**
#' - `plot_cooccurrence_network()`: Word co-occurrence networks
#' - `plot_correlation_network()`: Word correlation networks
#' - `calculate_word_frequency()`: Word frequency over variables
#' 
#' **Semantic Analysis:**
#' - `semantic_similarity_analysis()`: Multi-method document similarity analysis
#' - `reduce_dimensions()`: PCA, t-SNE, UMAP dimensionality reduction
#' - `cluster_embeddings()`: Advanced clustering with k-means, hierarchical, DBSCAN
#' - `plot_semantic_viz()`: Interactive semantic analysis visualizations
#'
#' **Workflow Functions:**
#' - `run_text_workflow()`: End-to-end text preprocessing workflow
#'
#' @section Usage Examples:
#' 
#' **Standalone R Package Usage:**
#' \preformatted{
#' # Load the package
#' library(TextAnalysisR)
#'
#' # Basic preprocessing workflow
#' data <- TextAnalysisR::SpecialEduTech
#'
#' # Unite text columns
#' united_data <- unite_cols(data, c("title", "keyword", "abstract"))
#'
#' # Preprocess texts
#' tokens <- prep_texts(
#'   united_tbl = united_data,
#'   text_field = "united_texts",
#'   min_char = 2,
#'   verbose = TRUE
#' )
#'
#' # Create document-feature matrix
#' dfm_obj <- quanteda::dfm(tokens)
#'
#' # Word frequency visualization
#' freq_plot <- plot_word_frequency(dfm_obj, n = 20)
#'
#' # Network analysis
#' cooccur_network <- plot_cooccurrence_network(
#'   dfm_obj,
#'   co_occur_n = 10,
#'   top_node_n = 50
#' )
#'
#' # Topic modeling
#' stm_data <- quanteda::convert(dfm_obj, to = "stm")
#' topic_model <- stm::stm(
#'   documents = stm_data$documents,
#'   vocab = stm_data$vocab,
#'   K = 10,
#'   verbose = TRUE
#' )
#'
#' # Get and visualize topic terms
#' topic_terms <- get_topic_terms(topic_model, top_term_n = 10)
#' topic_plot <- plot_word_probability(topic_terms, ncol = 3)
#'
#' # Document similarity analysis
#' texts <- united_data$united_texts
#' sim_result <- semantic_similarity_analysis(
#'   texts = texts,
#'   document_feature_type = "words",
#'   similarity_method = "cosine"
#' )
#'
#' # Visualize similarity matrix
#' sim_plot <- plot_semantic_viz(
#'   sim_result,
#'   plot_type = "similarity"
#' )
#'
#' # Clustering analysis
#' data_matrix <- as.matrix(dfm_obj)
#' cluster_result <- cluster_embeddings(
#'   data_matrix,
#'   method = "kmeans",
#'   n_clusters = 5
#' )
#'
#' # Visualize clusters
#' cluster_plot <- plot_semantic_viz(
#'   cluster_result,
#'   plot_type = "clustering"
#' )
#' }
#' 
#' **Shiny Application Usage:**
#' \preformatted{
#' # Launch the Shiny application
#' TextAnalysisR::run_app()
#'
#' # The Shiny app provides an interactive interface for all functions
#' # and automatically calls the underlying R package functions
#' }
#'
#' @section Package Features:
#' 
#' **Flexible Input Support:**
#' - Excel files (.xlsx, .xls)
#' - CSV files
#' - PDF documents
#' - Word documents (.docx)
#' - Text files
#' - Copy-paste text input
#' 
#' **Advanced Text Analysis:**
#' - Multi-method similarity analysis (words, n-grams, topics, embeddings)
#' - Dimensionality reduction (PCA, t-SNE, UMAP)
#' - Multiple clustering algorithms (k-means, hierarchical, UMAP+DBSCAN)
#' - Network analysis and visualization
#' - Topic modeling with STM
#' 
#' **Interactive Visualizations:**
#' - Plotly-based interactive plots
#' - Network visualizations
#' - Similarity heatmaps
#' - Dimensionality reduction scatter plots
#' - Topic term plots
#' 
#' **Integration Capabilities:**
#' - OpenAI API integration for topic labeling
#' - Python integration for embeddings (sentence-transformers)
#' - Comprehensive export options
#' - Memory-efficient processing
#'
#' @section Getting Started:
#'
#' 1. **For Text Preprocessing:** Use `run_text_workflow()` for end-to-end preprocessing
#' 2. **For Interactive Use:** Launch the Shiny app with `run_app()`
#' 3. **For Custom Workflows:** Use individual functions as building blocks
#' 4. **For Advanced Users:** Combine with your own R code and other packages
#'
#' @author Shin Mikyung 
#' @docType package
#' @name TextAnalysisR
