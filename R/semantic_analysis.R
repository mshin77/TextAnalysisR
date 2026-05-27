#' @importFrom utils modifyList
#' @importFrom stats cor hclust dist cutree
NULL

# Semantic Analysis Functions
# Functions for semantic analysis, embeddings, and document clustering

.cosine_sim <- function(a, b) {
  sum(a * b) / (sqrt(sum(a^2)) * sqrt(sum(b^2)))
}

.resolve_llm_setup <- function(provider, model, api_key, defaults, strict_validate = FALSE) {
  if (file.exists(".env") && requireNamespace("dotenv", quietly = TRUE)) {
    dotenv::load_dot_env()
  }
  if (is.null(model)) model <- defaults[[provider]]
  if (provider %in% c("openai", "gemini")) {
    if (is.null(api_key)) {
      env_var <- switch(provider, openai = "OPENAI_API_KEY", gemini = "GEMINI_API_KEY")
      api_key <- Sys.getenv(env_var)
    }
    if (!nzchar(api_key)) stop(.missing_api_key_message(provider, "package"), call. = FALSE)
    if (strict_validate) {
      validation <- validate_api_key(api_key, strict = FALSE)
      if (!validation$valid) stop(sprintf("Invalid API key format: %s", validation$error))
    }
  }
  list(provider = provider, model = model, api_key = api_key)
}

.empty_sentiment_row <- function(doc_name) {
  n <- length(doc_name)
  data.frame(
    document = doc_name,
    sentiment = rep(NA_character_, n),
    sentiment_score = rep(NA_real_, n),
    confidence = rep(NA_real_, n),
    explanation = rep(NA_character_, n),
    stringsAsFactors = FALSE
  )
}

#' @title Calculate Document Similarity
#'
#' @description
#' Calculates similarity between documents using traditional NLP methods or
#' modern embedding-based approaches. Metrics are automatically
#' computed unless disabled.
#'
#' @param texts A character vector of texts to compare.
#' @param document_feature_type Feature extraction type: "words", "ngrams", "embeddings", or "topics".
#' @param semantic_ngram_range Integer, n-gram range for ngram features (default: 2).
#' @param similarity_method Similarity calculation method: "cosine", "jaccard", "euclidean", "manhattan".
#' @param use_embeddings Logical, use embedding-based similarity (default: FALSE).
#' @param embedding_model Sentence transformer model name (default: "all-MiniLM-L6-v2").
#' @param calculate_metrics Logical, compute similarity metrics (default: TRUE).
#' @param verbose Logical, if TRUE, prints progress messages.
#'
#' @return A list containing:
#'   \describe{
#'     \item{similarity_matrix}{N x N similarity matrix}
#'     \item{feature_matrix}{Document feature matrix used for calculation}
#'     \item{method_info}{Information about the method used}
#'     \item{metrics}{Similarity metrics (if calculate_metrics = TRUE)}
#'     \item{execution_time}{Time taken for analysis}
#'   }
#'
#' @concept semantic
#' @keywords internal
#' @export
#'
#' @examples
#' if (interactive()) {
#'   data(SpecialEduTech)
#'   texts <- SpecialEduTech$abstract[1:5]
#'
#'   result <- calculate_document_similarity(
#'     texts = texts,
#'     document_feature_type = "words",
#'     similarity_method = "cosine"
#'   )
#'
#'   print(result$similarity_matrix)
#'   print(result$metrics)
#' }
calculate_document_similarity <- function(texts,
                            document_feature_type = "words",
                            semantic_ngram_range = 2,
                            similarity_method = "cosine",
                            use_embeddings = FALSE,
                            embedding_model = "all-MiniLM-L6-v2",
                            calculate_metrics = TRUE,
                            verbose = TRUE) {

  if (verbose) {
    message("Starting document similarity analysis...")
    message("Feature type: ", document_feature_type)
    message("Similarity method: ", similarity_method)
    message("Use embeddings: ", use_embeddings)
  }

  if (is.null(texts) || length(texts) == 0) {
    stop("No texts provided for analysis")
  }

  valid_texts <- texts[nchar(trimws(texts)) > 0]
  if (length(valid_texts) < 2) {
    stop("Need at least 2 non-empty texts for analysis")
  }

  start_time <- Sys.time()

  tryCatch({
    if (verbose) message("Step 1: Generating feature matrix...")

    feature_matrix <- switch(document_feature_type,
      "words" = {
        if (verbose) message("Using word-based features...")
        corpus <- quanteda::corpus(valid_texts)
        tokens <- quanteda::tokens(corpus,
                                   remove_punct = TRUE,
                                   remove_numbers = TRUE,
                                   remove_symbols = TRUE,
                                   remove_separators = TRUE)
        tokens <- quanteda::tokens_tolower(tokens)
        tokens <- quanteda::tokens_remove(tokens, quanteda::stopwords("english"))
        dfm <- quanteda::dfm(tokens)
        dfm <- quanteda::dfm_trim(dfm, min_termfreq = 2, min_docfreq = 1)
        as.matrix(dfm)
      },
      "ngrams" = {
        if (verbose) message("Using n-gram features (n=", semantic_ngram_range, ")...")
        corpus <- quanteda::corpus(valid_texts)
        tokens <- quanteda::tokens(corpus,
                                   remove_punct = TRUE,
                                   remove_numbers = TRUE,
                                   remove_symbols = TRUE,
                                   remove_separators = TRUE)
        tokens <- quanteda::tokens_tolower(tokens)
        tokens <- quanteda::tokens_remove(tokens, quanteda::stopwords("english"))
        tokens_ngrams <- quanteda::tokens_ngrams(tokens, n = semantic_ngram_range,
                                                  concatenator = "_")
        dfm_ngrams <- quanteda::dfm(tokens_ngrams)
        as.matrix(dfm_ngrams)
      },
      "embeddings" = {
        if (verbose) message("Using embedding features...")
        if (!requireNamespace("reticulate", quietly = TRUE)) {
          stop("reticulate package is required for embedding analysis")
        }

        python_available <- tryCatch({
          reticulate::py_config()
          TRUE
        }, error = function(e) FALSE)

        if (!python_available) {
          stop("Python not available. Please install Python and sentence-transformers: ",
               "pip install sentence-transformers")
        }
        sentence_transformers <- reticulate::import("sentence_transformers")
        model <- sentence_transformers$SentenceTransformer(embedding_model)

        n_docs <- length(valid_texts)
        batch_size <- if (n_docs > 100) 25 else if (n_docs > 50) 50 else n_docs

        if (verbose) message("Processing ", n_docs, " documents with embeddings...")

        embeddings_list <- list()
        for (i in seq(1, n_docs, by = batch_size)) {
          end_idx <- min(i + batch_size - 1, n_docs)
          batch_texts <- valid_texts[i:end_idx]
          batch_embeddings <- model$encode(batch_texts, show_progress_bar = FALSE)
          embeddings_list[[length(embeddings_list) + 1]] <- batch_embeddings
        }

        do.call(rbind, embeddings_list)
      },
      "topics" = {
        if (verbose) message("Using topic features...")
        stop("Topic features require a pre-fitted STM model. ",
             "Please use words, ngrams, or embeddings.")
      },
      stop("Unsupported feature type: ", document_feature_type)
    )

    if (verbose) message("Step 2: Calculating similarity matrix...")

    if (use_embeddings && document_feature_type == "embeddings") {
      if (!requireNamespace("reticulate", quietly = TRUE)) {
        stop("reticulate package is required for embedding similarity")
      }

      sklearn_metrics <- reticulate::import("sklearn.metrics.pairwise")
      similarity_matrix <- sklearn_metrics$cosine_similarity(feature_matrix)
      similarity_matrix <- as.matrix(similarity_matrix)

      method_info <- list(
        method = "embedding_cosine",
        model_name = embedding_model,
        n_docs = length(valid_texts)
      )
    } else {
      if (ncol(feature_matrix) == 0) {
        if (verbose) message("No features remaining. Using basic character similarity...")
        similarity_matrix <- outer(valid_texts, valid_texts, function(x, y) {
          mapply(function(a, b) {
            if (nchar(a) == 0 || nchar(b) == 0) return(0)
            common_chars <- sum(utf8ToInt(a) %in% utf8ToInt(b))
            max(common_chars / max(nchar(a), nchar(b)), 0)
          }, x, y)
        })
        method_info <- list(method = "character_similarity", n_docs = length(valid_texts))
      } else {
        if (similarity_method == "cosine") {
          similarity_matrix <- calculate_cosine_similarity(feature_matrix)
        } else {
          stop("Non-cosine similarity methods require dfm input. Please use cosine similarity or provide a dfm object.")
        }
        method_info <- list(
          method = paste0("traditional_", similarity_method),
          n_docs = length(valid_texts)
        )
      }
    }

    diag(similarity_matrix) <- 1
    similarity_matrix[similarity_matrix > 1] <- 1
    similarity_matrix[similarity_matrix < -1] <- -1

    metrics <- NULL
    if (calculate_metrics) {
      if (verbose) message("Step 3: Calculating metrics...")
      metrics <- calculate_metrics(similarity_matrix, method_info = method_info)
    }

    execution_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

    if (verbose) {
      message("Document similarity analysis completed in ", round(execution_time, 2), " seconds")
      message("Documents analyzed: ", length(valid_texts))
      message("Feature dimensions: ", ncol(feature_matrix))
    }

    result <- list(
      similarity_matrix = similarity_matrix,
      feature_matrix = feature_matrix,
      method_info = method_info,
      metrics = metrics,
      execution_time = execution_time,
      n_documents = length(valid_texts),
      feature_type = document_feature_type,
      similarity_method = similarity_method,
      use_embeddings = use_embeddings
    )

    return(result)

  }, error = function(e) {
    stop("Error in document similarity analysis: ", e$message)
  })
}

#' @title Fit Semantic Model
#'
#' @description
#' Performs semantic analysis including similarity, dimensionality reduction,
#' and clustering. This is a high-level wrapper function.
#'
#' @param texts A character vector of texts to analyze.
#' @param analysis_types Types of analysis to perform: "similarity", "dimensionality_reduction", "clustering".
#' @param document_feature_type Feature extraction type (default: "embeddings").
#' @param similarity_method Similarity calculation method (default: "cosine").
#' @param use_embeddings Logical, use embedding-based approaches (default: TRUE).
#' @param embedding_model Sentence transformer model name (default: "all-MiniLM-L6-v2").
#' @param dimred_method Dimensionality reduction method: "PCA", "t-SNE", "UMAP" (default: "UMAP").
#' @param clustering_method Clustering method: "kmeans", "hierarchical", "umap_dbscan" (default: "umap_dbscan").
#' @param n_components Number of dimensions for reduction (default: 2).
#' @param n_clusters Number of clusters (default: 5).
#' @param seed Random seed for reproducibility (default: 123).
#' @param verbose Logical, if TRUE, prints progress messages.
#'
#' @return A list containing results from requested analyses.
#'
#' @concept semantic
#' @export
#'
#' @examples
#' if (interactive()) {
#'   data(SpecialEduTech)
#'   texts <- SpecialEduTech$abstract[1:10]
#'
#'   results <- fit_semantic_model(
#'     texts = texts,
#'     analysis_types = c("similarity", "clustering")
#'   )
#'
#'   print(results)
#' }
fit_semantic_model <- function(texts,
                             analysis_types = c("similarity", "dimensionality_reduction",
                                                "clustering"),
                             document_feature_type = "embeddings",
                             similarity_method = "cosine",
                             use_embeddings = TRUE,
                             embedding_model = "all-MiniLM-L6-v2",
                             dimred_method = "UMAP",
                             clustering_method = "umap_dbscan",
                             n_components = 2,
                             n_clusters = 5,
                             seed = 123,
                             verbose = TRUE) {

  if (verbose) {
    message("Starting semantic analysis...")
    message("Analysis types: ", paste(analysis_types, collapse = ", "))
  }

  results <- list()
  start_time <- Sys.time()

  if ("similarity" %in% analysis_types || length(analysis_types) > 1) {
    if (verbose) message("Step 1: Document similarity analysis...")

    results$similarity <- calculate_document_similarity(
      texts = texts,
      document_feature_type = document_feature_type,
      similarity_method = similarity_method,
      use_embeddings = use_embeddings,
      embedding_model = embedding_model,
      verbose = verbose
    )
  }

  if ("dimensionality_reduction" %in% analysis_types) {
    if (verbose) message("Step 2: Dimensionality reduction analysis...")

    data_matrix <- if (!is.null(results$similarity)) {
      results$similarity$feature_matrix
    } else {
      similarity_result <- calculate_document_similarity(
        texts = texts,
        document_feature_type = document_feature_type,
        use_embeddings = use_embeddings,
        embedding_model = embedding_model,
        verbose = FALSE
      )
      similarity_result$feature_matrix
    }

    results$dimensionality_reduction <- reduce_dimensions(
      data_matrix = data_matrix,
      method = dimred_method,
      n_components = n_components,
      seed = seed,
      verbose = verbose
    )
  }

  if ("clustering" %in% analysis_types) {
    if (verbose) message("Step 3: Clustering analysis...")

    data_matrix <- if (!is.null(results$similarity)) {
      results$similarity$feature_matrix
    } else {
      similarity_result <- calculate_document_similarity(
        texts = texts,
        document_feature_type = document_feature_type,
        use_embeddings = use_embeddings,
        embedding_model = embedding_model,
        verbose = FALSE
      )
      similarity_result$feature_matrix
    }

    if (clustering_method != "none") {
      clustering_result <- list()

      if (clustering_method == "kmeans") {
        km <- stats::kmeans(data_matrix, centers = n_clusters, nstart = 25)
        clustering_result$clusters <- km$cluster
        clustering_result$centers <- km$centers
      } else if (clustering_method == "hierarchical") {
        dist_matrix <- stats::dist(data_matrix)
        hc <- stats::hclust(dist_matrix, method = "ward.D2")
        clustering_result$clusters <- stats::cutree(hc, k = n_clusters)
        clustering_result$dendrogram <- hc
      } else if (clustering_method == "dbscan") {
        clustering_result$clusters <- rep(1, nrow(data_matrix))
      }

      results$clustering <- clustering_result
    }
  }

  execution_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

  if (verbose) {
    message("Semantic analysis completed in ", round(execution_time, 2), " seconds")
  }

  results$execution_time <- execution_time
  results$analysis_types <- analysis_types
  results$timestamp <- Sys.time()

  return(results)
}

#' @title Dimensionality Reduction Analysis
#'
#' @description
#' This function performs dimensionality reduction using various methods
#' including PCA, t-SNE, and UMAP.
#' For efficiency and consistency, PCA preprocessing is always performed first,
#' and t-SNE/UMAP use the PCA results as input.
#' This follows best practices for high-dimensional data analysis.
#'
#' @param data_matrix A numeric matrix where rows represent documents and
#'   columns represent features.
#' @param method The dimensionality reduction method. Options: "PCA", "t-SNE", "UMAP".
#' @param n_components The number of components/dimensions to reduce to (default: 2).
#' @param pca_dims The number of dimensions for PCA preprocessing (default: 50).
#' @param tsne_perplexity The perplexity parameter for t-SNE (default: 30).
#' @param tsne_max_iter The maximum number of iterations for t-SNE (default: 1000).
#' @param umap_neighbors The number of neighbors for UMAP (default: 15).
#' @param umap_min_dist The minimum distance for UMAP (default: 0.1).
#' @param umap_metric The metric for UMAP (default: "cosine").
#' @param seed Random seed for reproducibility (default: 123).
#' @param verbose Logical, if TRUE, prints progress messages.
#'
#' @return A list containing the reduced dimensions, method used, and additional metadata.
#'
#' @concept semantic
#' @export
#'
#' @examples
#' if (interactive()) {
#'   mydata <- TextAnalysisR::SpecialEduTech
#'
#'   united_tbl <- TextAnalysisR::unite_cols(
#'     mydata,
#'     listed_vars = c("title", "keyword", "abstract")
#'   )
#'
#'   tokens <- TextAnalysisR::prep_texts(united_tbl, text_field = "united_texts")
#'
#'   dfm_object <- quanteda::dfm(tokens)
#'
#'   data_matrix <- as.matrix(dfm_object)
#'
#'   pca_result <- TextAnalysisR::reduce_dimensions(
#'     data_matrix,
#'     method = "PCA"
#'   )
#'   print(pca_result)
#'
#'   tsne_result <- TextAnalysisR::reduce_dimensions(
#'     data_matrix,
#'     method = "t-SNE"
#'   )
#'   print(tsne_result)
#'
#'   umap_result <- TextAnalysisR::reduce_dimensions(
#'     data_matrix,
#'     method = "UMAP"
#'   )
#'   print(umap_result)
#' }
reduce_dimensions <- function(data_matrix,
                                            method = "PCA",
                                            n_components = 2,
                                            pca_dims = 50,
                                            tsne_perplexity = 30,
                                            tsne_max_iter = 1000,
                                            umap_neighbors = 15,
                                            umap_min_dist = 0.1,
                                            umap_metric = "cosine",
                                            seed = 123,
                                            verbose = TRUE) {

  if (verbose) {
    message("Starting dimensionality reduction with method: ", method)
  }

  start_time <- Sys.time()

  withr::with_seed(seed, tryCatch({
    if (verbose) message("Performing PCA preprocessing...")

    col_vars <- apply(data_matrix, 2, var, na.rm = TRUE)
    constant_cols <- which(col_vars == 0 | is.na(col_vars))

    if (length(constant_cols) > 0) {
      if (verbose) message(paste("Removing", length(constant_cols),
                                 "constant/zero columns before PCA"))
      data_matrix <- data_matrix[, -constant_cols, drop = FALSE]
    }

    if (ncol(data_matrix) < 2) {
      stop("Insufficient non-constant features for analysis")
    }

    max_components <- min(nrow(data_matrix) - 1, ncol(data_matrix), pca_dims)
    pca_dims_actual <- min(pca_dims, max_components)

    pca_result <- stats::prcomp(data_matrix, center = TRUE, scale. = TRUE, rank. = pca_dims_actual)


    result <- switch(method,
      "PCA" = {
        if (verbose) message("Returning PCA results...")

        final_components <- min(n_components, ncol(pca_result$x))
        reduced_data <- pca_result$x[, 1:final_components, drop = FALSE]

        list(
          reduced_data = reduced_data,
          method = "PCA",
          pca_result = pca_result,
          variance_explained = summary(pca_result)$importance[2, 1:final_components]
        )
      },
      "t-SNE" = {
        if (verbose) message("Performing t-SNE on PCA results...")

        if (!requireNamespace("Rtsne", quietly = TRUE)) {
          stop("Rtsne package is required for t-SNE analysis. ",
               "Please install it with: install.packages('Rtsne')")
        }

        data_for_tsne <- pca_result$x
        adjusted_perplexity <- min(tsne_perplexity,
                                   floor((nrow(data_matrix) - 1) / 3))

        tsne_result <- Rtsne::Rtsne(data_for_tsne,
                                   dims = n_components,
                                   perplexity = adjusted_perplexity,
                                   max_iter = tsne_max_iter,
                                   check_duplicates = FALSE,
                                   pca = FALSE,
                                   verbose = verbose)

        list(
          reduced_data = tsne_result$Y,
          method = "t-SNE",
          tsne_result = tsne_result,
          pca_result = pca_result,
          perplexity = adjusted_perplexity
        )
      },
      "UMAP" = {
        if (verbose) message("Performing UMAP on PCA results...")

        if (requireNamespace("umap", quietly = TRUE)) {
          safe_n_neighbors <- min(umap_neighbors, nrow(data_matrix) - 1, 15)

          umap_config <- umap::umap.defaults
          umap_config$n_neighbors <- safe_n_neighbors
          umap_config$min_dist <- umap_min_dist
          umap_config$n_components <- n_components
          umap_config$metric <- umap_metric
          umap_config$random_state <- seed

          umap_result <- umap::umap(pca_result$x, config = umap_config)

          list(
            reduced_data = umap_result$layout,
            method = "UMAP",
            umap_result = umap_result,
            pca_result = pca_result,
            umap_params = list(
              n_neighbors = safe_n_neighbors,
              min_dist = umap_min_dist,
              metric = umap_metric
            )
          )
        } else if (requireNamespace("reticulate", quietly = TRUE)) {
          if (!reticulate::py_module_available("umap")) {
            stop("umap-learn Python package is required. ",
                 "Please install it with: pip install umap-learn")
          }

          umap_module <- reticulate::import("umap")
          reducer <- umap_module$UMAP(
            n_neighbors = umap_neighbors,
            min_dist = umap_min_dist,
            n_components = as.integer(n_components),
            metric = umap_metric,
            random_state = as.integer(seed)
          )

          embedding <- reducer$fit_transform(pca_result$x)

          list(
            reduced_data = embedding,
            method = "UMAP",
            pca_result = pca_result,
            umap_params = list(
              n_neighbors = umap_neighbors,
              min_dist = umap_min_dist,
              metric = umap_metric
            )
          )
        } else {
          stop("Either 'umap' R package or 'reticulate' with Python 'umap-learn' ",
               "is required for UMAP analysis")
        }
      },
      stop("Unsupported dimensionality reduction method: ", method)
    )

    execution_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

    if (verbose) {
      message("Dimensionality reduction completed in ", round(execution_time, 2), " seconds")
    }

    result$execution_time <- execution_time
    result$timestamp <- Sys.time()
    result$seed <- seed

    return(result)

  }, error = function(e) {
    stop("Error in dimensionality reduction analysis: ", e$message)
  }))
}

#' @title Embedding-based Document Clustering
#'
#' @description
#' This function performs clustering analysis using various methods, ordered
#' from simple to detailed:
#' k-means (simplest), hierarchical (intermediate), and UMAP+DBSCAN
#' (most detailed).
#'
#' @param data_matrix A numeric matrix where rows represent documents and
#'   columns represent features.
#' @param method The clustering method. Options: "kmeans", "hierarchical",
#'   "umap_dbscan".
#' @param n_clusters The number of clusters (for k-means and hierarchical).
#'   If 0, optimal number is determined automatically.
#' @param umap_neighbors The number of neighbors for UMAP (default: 15).
#' @param umap_min_dist The minimum distance for UMAP (default: 0.1).
#' @param umap_n_components The number of UMAP components (default: 10).
#' @param umap_metric The metric for UMAP (default: "cosine").
#' @param dbscan_eps The eps parameter for DBSCAN. If 0, optimal value is determined automatically.
#' @param dbscan_min_samples The minimum samples for DBSCAN (default: 5).
#' @param reduce_outliers Logical, if TRUE, reassigns noise points (cluster 0) to nearest cluster (default: TRUE).
#' @param outlier_strategy Strategy for outlier reduction: "centroid" (default,
#'   Euclidean distance in UMAP space) or "embeddings" (cosine similarity in
#'   original space). Follows BERTopic methodology.
#' @param seed Random seed for reproducibility (default: 123).
#' @param verbose Logical, if TRUE, prints progress messages.
#'
#' @return A list containing cluster assignments, method used, and quality metrics.
#'
#' @concept semantic
#' @export
#'
#' @examples
#' if (interactive()) {
#'   mydata <- TextAnalysisR::SpecialEduTech
#'
#'   united_tbl <- TextAnalysisR::unite_cols(
#'     mydata,
#'     listed_vars = c("title", "keyword", "abstract")
#'   )
#'
#'   tokens <- TextAnalysisR::prep_texts(united_tbl, text_field = "united_texts")
#'
#'   dfm_object <- quanteda::dfm(tokens)
#'
#'   data_matrix <- as.matrix(dfm_object)
#'
#'   kmeans_result <- TextAnalysisR::cluster_embeddings(
#'     data_matrix,
#'     method = "kmeans",
#'     n_clusters = 5
#'   )
#'   print(kmeans_result)
#'
#'   hierarchical_result <- TextAnalysisR::cluster_embeddings(
#'     data_matrix,
#'     method = "hierarchical",
#'     n_clusters = 5
#'   )
#'   print(hierarchical_result)
#'
#'   umap_dbscan_result <- TextAnalysisR::cluster_embeddings(
#'     data_matrix,
#'     method = "umap_dbscan",
#'     umap_neighbors = 15,
#'     umap_min_dist = 0.1
#'   )
#'   print(umap_dbscan_result)
#' }
cluster_embeddings <- function(data_matrix,
                                       method = "kmeans",
                                       n_clusters = 0,
                                       umap_neighbors = 15,
                                       umap_min_dist = 0.1,
                                       umap_n_components = 10,
                                       umap_metric = "cosine",
                                       dbscan_eps = 0,
                                       dbscan_min_samples = 5,
                                       reduce_outliers = TRUE,
                                       outlier_strategy = "centroid",
                                       seed = 123,
                                       verbose = TRUE) {

  if (verbose) {
    message("Starting clustering analysis with method: ", method)
  }

  start_time <- Sys.time()

  withr::with_seed(seed, tryCatch({
    result <- switch(method,
      "umap_dbscan" = {
        if (verbose) message("Performing UMAP + DBSCAN clustering...")

        umap_result <- reduce_dimensions(
          data_matrix,
          method = "UMAP",
          n_components = umap_n_components,
          umap_neighbors = umap_neighbors,
          umap_min_dist = umap_min_dist,
          umap_metric = umap_metric,
          seed = seed,
          verbose = verbose
        )

        if (!requireNamespace("dbscan", quietly = TRUE)) {
          stop("dbscan package is required for DBSCAN clustering. ",
               "Please install it with: install.packages('dbscan')")
        }

        if (dbscan_eps == 0) {
          if (verbose) message("Determining optimal eps parameter...")
          knn_dist <- dbscan::kNNdist(umap_result$reduced_data, k = dbscan_min_samples)
          dbscan_eps <- stats::quantile(knn_dist, 0.9)
          auto_eps <- TRUE
        } else {
          auto_eps <- FALSE
        }

        dbscan_result <- dbscan::dbscan(umap_result$reduced_data,
                                       eps = dbscan_eps,
                                       minPts = dbscan_min_samples)

        clusters <- dbscan_result$cluster
        n_clusters_found <- length(unique(clusters)) - (0 %in% clusters)
        noise_ratio <- sum(clusters == 0) / length(clusters)

        outliers_reassigned <- 0
        outlier_strategy_used <- outlier_strategy
        if (reduce_outliers && any(clusters == 0) && n_clusters_found > 0) {
          if (verbose) message("Reassigning ", sum(clusters == 0), " noise points using '", outlier_strategy, "' strategy...")

          noise_idx <- which(clusters == 0)
          valid_clusters <- unique(clusters[clusters > 0])

          if (outlier_strategy == "embeddings") {
            embedding_centroids <- sapply(valid_clusters, function(cl) {
              colMeans(data_matrix[clusters == cl, , drop = FALSE])
            })
            if (is.vector(embedding_centroids)) embedding_centroids <- matrix(embedding_centroids, nrow = 1)
            embedding_centroids <- t(embedding_centroids)

            for (idx in noise_idx) {
              point <- data_matrix[idx, ]
              similarities <- apply(embedding_centroids, 1, function(c) .cosine_sim(point, c))
              clusters[idx] <- valid_clusters[which.max(similarities)]
            }
          } else {
            centroids <- sapply(valid_clusters, function(cl) {
              colMeans(umap_result$reduced_data[clusters == cl, , drop = FALSE])
            })
            if (is.vector(centroids)) centroids <- matrix(centroids, nrow = 1)
            centroids <- t(centroids)

            for (idx in noise_idx) {
              point <- umap_result$reduced_data[idx, ]
              distances <- apply(centroids, 1, function(c) sqrt(sum((point - c)^2)))
              clusters[idx] <- valid_clusters[which.min(distances)]
            }
          }
          outliers_reassigned <- length(noise_idx)
        }

        list(
          clusters = clusters,
          method = "umap_dbscan",
          n_clusters = n_clusters_found,
          umap_embedding = umap_result$reduced_data,
          dbscan_result = dbscan_result,
          auto_detected = auto_eps,
          detection_method = if (auto_eps) "Knee Point Detection" else "Manual",
          noise_ratio = noise_ratio,
          reduce_outliers = reduce_outliers,
          outlier_strategy = if (reduce_outliers) outlier_strategy_used else NA_character_,
          outliers_reassigned = outliers_reassigned,
          parameters = list(
            eps = dbscan_eps,
            min_samples = dbscan_min_samples,
            umap_neighbors = umap_neighbors,
            umap_min_dist = umap_min_dist,
            umap_metric = umap_metric
          )
        )
      },
      "kmeans" = {
        if (verbose) message("Performing k-means clustering...")

        if (n_clusters == 0) {
          if (verbose) message("Determining optimal number of clusters...")
          max_k <- min(10, nrow(data_matrix) - 1)
          wss <- sapply(1:max_k, function(k) {
            kmeans(scale(data_matrix), k, nstart = 10, iter.max = 100)$tot.withinss
          })

          n_clusters <- which.max(diff(diff(wss))) + 1
          if (n_clusters < 2) n_clusters <- 2
          auto_detected <- TRUE
        } else {
          auto_detected <- FALSE
        }

        kmeans_result <- stats::kmeans(scale(data_matrix),
                                     centers = n_clusters,
                                     nstart = 25,
                                     iter.max = 100)

        list(
          clusters = kmeans_result$cluster,
          method = "kmeans",
          n_clusters = n_clusters,
          kmeans_result = kmeans_result,
          auto_detected = auto_detected,
          detection_method = if (auto_detected) "Elbow Method" else "Manual"
        )
      },
      "hierarchical" = {
        if (verbose) message("Performing hierarchical clustering...")

        dist_matrix <- stats::dist(scale(data_matrix))
        hclust_result <- stats::hclust(dist_matrix, method = "ward.D2")

        if (n_clusters == 0) {
          if (verbose) message("Determining optimal number of clusters...")
          sil_scores <- sapply(2:min(10, nrow(data_matrix) - 1), function(k) {
            clusters <- stats::cutree(hclust_result, k = k)
            if (!requireNamespace("cluster", quietly = TRUE)) {
              stop("cluster package is required for silhouette analysis. ",
                   "Please install it with: install.packages('cluster')")
            }
            mean(cluster::silhouette(clusters, dist_matrix)[, 3])
          })
          n_clusters <- which.max(sil_scores) + 1
          auto_detected <- TRUE
        } else {
          auto_detected <- FALSE
        }

        clusters <- stats::cutree(hclust_result, k = n_clusters)

        list(
          clusters = clusters,
          method = "hierarchical",
          n_clusters = n_clusters,
          hclust_result = hclust_result,
          auto_detected = auto_detected,
          detection_method = if (auto_detected) "Silhouette Method" else "Manual"
        )
      },
      stop("Unsupported clustering method: ", method)
    )

    if (result$n_clusters > 1 && !all(result$clusters == 0)) {
      if (verbose) message("Calculating quality metrics...")

      if (requireNamespace("cluster", quietly = TRUE)) {
        tryCatch({
          sil <- cluster::silhouette(result$clusters, stats::dist(scale(data_matrix)))
          result$silhouette <- mean(sil[, 3])
        }, error = function(e) {
          result$silhouette <- NA
        })
      }

      if (requireNamespace("clusterCrit", quietly = TRUE)) {
        tryCatch({
          result$davies_bouldin <- clusterCrit::intCriteria(
            as.matrix(scale(data_matrix)),
            result$clusters,
            "Davies_Bouldin"
          )$davies_bouldin
        }, error = function(e) {
          result$davies_bouldin <- NA
        })
      }

      if (requireNamespace("clusterCrit", quietly = TRUE)) {
        tryCatch({
          result$calinski_harabasz <- clusterCrit::intCriteria(
            as.matrix(scale(data_matrix)),
            result$clusters,
            "Calinski_Harabasz"
          )$calinski_harabasz
        }, error = function(e) {
          result$calinski_harabasz <- NA
        })
      }
    }

    execution_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

    if (verbose) {
      message("Clustering analysis completed in ", round(execution_time, 2), " seconds")
      if (result$n_clusters > 0) {
        message("Found ", result$n_clusters, " clusters")
      }
    }

    result$execution_time <- execution_time
    result$timestamp <- Sys.time()
    result$seed <- seed

    return(result)

  }, error = function(e) {
    stop("Error in clustering analysis: ", e$message)
  }))
}

#' @title Generate Embeddings
#'
#' @description
#' Generates embeddings for texts using sentence transformers.
#'
#' @param texts A character vector of texts.
#' @param model Sentence transformer model name (default: "all-MiniLM-L6-v2").
#' @param verbose Logical, if TRUE, prints progress messages.
#'
#' @return A matrix of embeddings.
#'
#' @concept semantic
#' @export
generate_embeddings <- function(texts, model = "all-MiniLM-L6-v2", verbose = TRUE) {

  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("reticulate package is required for embedding generation")
  }

  if (verbose) message("Generating embeddings using model: ", model)

  tryCatch({
    sentence_transformers <- reticulate::import("sentence_transformers")
    embedding_model <- sentence_transformers$SentenceTransformer(model)
    embeddings <- embedding_model$encode(texts, show_progress_bar = verbose)

    if (verbose) message("Embeddings generated successfully")

    return(embeddings)

  }, error = function(e) {
    stop("Error generating embeddings: ", e$message)
  })
}

#' @title Semantic Similarity Analysis
#' @description Wrapper for calculate_document_similarity
#' @param ... Arguments passed to calculate_document_similarity
#' @return Similarity analysis results
#' @concept semantic
#' @seealso [calculate_document_similarity()] for the core computation; [plot_similarity_heatmap()] to render the resulting matrix
#' @export
semantic_similarity_analysis <- function(...) {
  return(calculate_document_similarity(...))
}

#' @title Semantic Document Clustering
#' @description Creates a unified visualization of document clustering with optional clustering
#' @param embeddings Document embeddings matrix
#' @param method Dimensionality reduction method ("PCA", "t-SNE", "UMAP")
#' @param clusters Optional cluster assignments
#' @param ... Additional arguments
#'
#' @return A ggplot2 visualization of document clusters
#'
#' @concept semantic
#' @export
semantic_document_clustering <- function(embeddings, method = "UMAP", clusters = NULL, ...) {
  reduced_dims <- reduce_dimensions(embeddings, method = method, ...)

  if (!is.null(clusters)) {
    return(list(
      coordinates = reduced_dims,
      clusters = clusters,
      method = method
    ))
  }

  return(list(
    coordinates = reduced_dims,
    method = method
  ))
}

#' @title Analyze Document Clustering
#' @description Complete document clustering analysis with dimensionality reduction and optional clustering
#' @param feature_matrix Feature matrix (documents x features)
#' @param method Dimensionality reduction method ("PCA", "t-SNE", "UMAP")
#' @param clustering_method Clustering method ("none", "kmeans", "hierarchical", "dbscan", "hdbscan")
#' @param ... Additional parameters for methods
#' @return List containing coordinates, clusters, method info, and quality metrics
#' @concept semantic
#' @export
analyze_document_clustering <- function(feature_matrix,
                                  method = "UMAP",
                                  clustering_method = "none",
                                  ...) {

  coords <- reduce_dimensions(feature_matrix, method = method, ...)

  clusters <- NULL
  quality_metrics <- list()

  if (clustering_method != "none") {
    if (clustering_method == "kmeans") {
      k <- list(...)$k %||% 5
      km_result <- kmeans(coords, centers = k, nstart = 25)
      clusters <- km_result$cluster
    } else if (clustering_method == "hierarchical") {
      k <- list(...)$k %||% 5
      hc_result <- hclust(dist(coords))
      clusters <- cutree(hc_result, k = k)
    } else if (clustering_method == "dbscan") {
      eps <- list(...)$eps %||% 0.5
      minPts <- list(...)$minPts %||% 5
      dbscan_result <- dbscan::dbscan(coords, eps = eps, minPts = minPts)
      clusters <- dbscan_result$cluster
    } else if (clustering_method == "hdbscan") {
      min_size <- list(...)$min_cluster_size %||% 5
      hdbscan_result <- dbscan::hdbscan(coords, minPts = min_size)
      clusters <- hdbscan_result$cluster
    }

    if (!is.null(clusters) && length(unique(clusters)) > 1) {
      sil <- cluster::silhouette(clusters, dist(coords))
      quality_metrics$silhouette <- mean(sil[, 3])
      quality_metrics$n_clusters <- length(unique(clusters[clusters > 0]))

      if (min(clusters) == 0) {
        quality_metrics$outlier_pct <- sum(clusters == 0) / length(clusters) * 100
      }
    }
  }

  return(list(
    coordinates = coords,
    clusters = clusters,
    method = method,
    clustering_method = clustering_method,
    quality_metrics = quality_metrics
  ))
}

#' @title Generate Cluster Labels
#' @description Generate descriptive labels for document clusters
#' @param feature_matrix Feature matrix used for clustering
#' @param clusters Cluster assignments
#' @param method Label generation method ("tfidf", "representative", "frequent")
#' @param n_terms Number of terms per label
#' @return Named list of cluster labels
#' @concept semantic
#' @export
generate_cluster_labels_auto <- function(feature_matrix,
                                         clusters,
                                         method = "tfidf",
                                         n_terms = 3) {

  unique_clusters <- sort(unique(clusters[clusters > 0]))
  labels <- list()

  for (cluster_id in unique_clusters) {
    cluster_docs <- which(clusters == cluster_id)
    cluster_features <- feature_matrix[cluster_docs, , drop = FALSE]

    if (method == "tfidf") {
      tf <- colSums(cluster_features)
      idf <- log(nrow(feature_matrix) / colSums(feature_matrix > 0))
      tfidf <- tf * idf
      top_terms <- names(head(sort(tfidf, decreasing = TRUE), n_terms))
    } else if (method == "representative") {
      cluster_mean <- colMeans(cluster_features)
      overall_mean <- colMeans(feature_matrix)
      diff <- cluster_mean - overall_mean
      top_terms <- names(head(sort(diff, decreasing = TRUE), n_terms))
    } else {
      term_freq <- colSums(cluster_features)
      top_terms <- names(head(sort(term_freq, decreasing = TRUE), n_terms))
    }

    labels[[as.character(cluster_id)]] <- paste(top_terms, collapse = ", ")
  }

  return(labels)
}

#' @title Generate Cluster Label Suggestions (Human-in-the-Loop)
#'
#' @description
#' Suggests descriptive labels for clusters using AI. Labels are suggestions
#' for human review - users should edit and approve before using.
#' Supports OpenAI, Gemini, or Ollama (local) for AI generation.
#' When running locally, Ollama is preferred for privacy and cost-free operation.
#'
#' @param cluster_keywords List of keywords for each cluster.
#' @param provider AI provider to use: "auto" (default), "openai", "gemini", or "ollama".
#'   "auto" will try Ollama first, then check for OpenAI/Gemini keys.
#' @param model Model name. If NULL, uses provider defaults: "gpt-4.1-mini" (OpenAI),
#'   "gemini-2.5-flash-lite" (Gemini), or recommended Ollama model.
#' @param temperature Temperature parameter (default: 0.3).
#' @param max_tokens Maximum tokens for response (default: 50).
#' @param api_key API key for OpenAI or Gemini. If NULL, uses environment variable.
#'   Not required for Ollama.
#' @param verbose Logical, if TRUE, prints progress messages.
#'
#' @return A list of generated labels.
#'
#' @concept semantic
#' @export
#'
#' @examples
#' if (interactive()) {
#'   cluster_keywords <- list(
#'     "1" = c("calculator", "arithmetic", "elementary", "remedial"),
#'     "2" = c("computer", "instruction", "multiplication", "drill")
#'   )
#'   labels_ollama <- generate_cluster_labels(cluster_keywords, provider = "ollama")
#'   labels_openai <- generate_cluster_labels(cluster_keywords, provider = "openai")
#'   labels_gemini <- generate_cluster_labels(cluster_keywords, provider = "gemini")
#' }
generate_cluster_labels <- function(cluster_keywords,
                                   provider = "auto",
                                   model = NULL,
                                   temperature = 0.3,
                                   max_tokens = 50,
                                   api_key = NULL,
                                   verbose = TRUE) {

  if (!requireNamespace("httr", quietly = TRUE) ||
      !requireNamespace("jsonlite", quietly = TRUE)) {
    stop(
      "The 'httr' and 'jsonlite' packages are required for this functionality. ",
      "Please install them using install.packages(c('httr', 'jsonlite'))."
    )
  }

  if (provider == "auto") {
    if (check_ollama(verbose = FALSE)) {
      provider <- "ollama"
      if (verbose) message("Using Ollama (local AI) for label generation")
    } else if (nzchar(Sys.getenv("OPENAI_API_KEY")) || (!is.null(api_key) && grepl("^sk-", api_key))) {
      provider <- "openai"
      if (verbose) message("Using OpenAI for label generation")
    } else if (nzchar(Sys.getenv("GEMINI_API_KEY")) || (!is.null(api_key) && grepl("^AIza", api_key))) {
      provider <- "gemini"
      if (verbose) message("Using Gemini for label generation")
    } else {
      stop("No AI provider available. Install Ollama or set OPENAI_API_KEY/GEMINI_API_KEY.")
    }
  }

  ollama_default <- if (provider == "ollama") {
    get_recommended_ollama_model(verbose = verbose) %||% "tinyllama"
  } else NULL
  setup <- .resolve_llm_setup(
    provider, model, api_key,
    defaults = list(ollama = ollama_default, openai = "gpt-4.1-mini", gemini = "gemini-2.5-flash-lite"),
    strict_validate = TRUE
  )
  model <- setup$model
  api_key <- setup$api_key

  if (verbose) {
    message("Generating AI labels for ", length(cluster_keywords), " clusters using ", provider, " (", model, ")...")
  }

  gen_names <- list()

  for (cluster_id in names(cluster_keywords)) {
    if (cluster_id == "0" || cluster_id == 0) {
      gen_names[[cluster_id]] <- "Outlier"
      next
    }

    keywords <- cluster_keywords[[cluster_id]]
    if (length(keywords) == 0) {
      gen_names[[cluster_id]] <- paste("Cluster", cluster_id)
      next
    }

    top_keywords <- paste(keywords[seq_len(min(10, length(keywords)))], collapse = ", ")

    prompt <- paste0(
      "You are a highly skilled data scientist specializing in generating concise and
descriptive topic labels based on provided top terms for each topic.

Your objective is to create precise and concise labels that capture the essence of each topic by following these guidelines:

1. Use Person-First Language:
   - Prioritize respectful and inclusive language.
   - Avoid terms that may be considered offensive or stigmatizing.
   - For example, use 'students with learning disabilities' instead of 'disabled students'.

2. Analyze the significance of the top terms:
   - Focus primarily on the most significant terms.
   - Include additional terms if they add essential context.

3. Synthesize the Topic Label:
   - Ensure clarity and conciseness (aim for 4-5 words).
   - Reflect the collective meaning of the most influential terms.
   - Use descriptive yet precise phrasing.

4. Maintain consistency:
   - Capitalize the first word using title case.
   - Use uniform formatting and avoid ambiguity.
   - Make concise and complete expressions.

Top 10 Keywords: ", top_keywords, "

Respond with ONLY the topic label, nothing else. Do not include quotes or explanations.

Generated Topic Label:"
    )

    tryCatch({
      # Call LLM
      response_text <- call_llm_api(
        provider = provider,
        system_prompt = "You are a data scientist specializing in generating concise cluster labels.",
        user_prompt = prompt,
        model = model,
        temperature = temperature,
        max_tokens = max_tokens,
        api_key = api_key
      )

      if (!is.null(response_text) && nzchar(response_text)) {
        label <- trimws(response_text)
        label <- gsub("^\"(.*)\"$", "\\1", label)
        label <- gsub("^Generated Topic Label:\\s*", "", label, ignore.case = TRUE)
        label <- trimws(label)
        gen_names[[cluster_id]] <- label
      } else {
        gen_names[[cluster_id]] <- paste("Cluster", cluster_id)
      }

    }, error = function(e) {
      warning("AI call failed for cluster ", cluster_id, ": ", e$message)
      gen_names[[cluster_id]] <- paste("Cluster", cluster_id)
    })

    Sys.sleep(if (provider %in% c("openai", "gemini")) 1 else 0.5)
  }

  if (verbose) message("AI label generation completed")

  return(gen_names)
}

#' @title Export Document Clustering Analysis
#' @description Export document clustering analysis results to CSV
#' @param coordinates Document coordinates
#' @param clusters Cluster assignments (optional)
#' @param labels Cluster labels (optional)
#' @param doc_ids Document IDs
#' @param file_path Path to save the CSV file
#'
#' @return Invisible data frame of the exported data
#'
#' @concept semantic
#' @export
export_document_clustering <- function(coordinates,
                                 clusters = NULL,
                                 labels = NULL,
                                 doc_ids = NULL,
                                 file_path) {

  export_data <- data.frame(
    doc_id = doc_ids %||% paste0("doc_", seq_len(nrow(coordinates))),
    x = coordinates[, 1],
    y = coordinates[, 2]
  )

  if (!is.null(clusters)) {
    export_data$cluster <- clusters

    if (!is.null(labels)) {
      export_data$cluster_label <- labels[match(clusters, names(labels))]
    }
  }

  utils::write.csv(export_data, file_path, row.names = FALSE)
  invisible(export_data)
}


#' @title Cross-Analysis Validation
#'
#' @description
#' Performs cross-validation between different analysis types (STM, semantic, clustering).
#'
#' @param semantic_results Results from semantic analysis.
#' @param stm_results Optional STM results for comparison.
#' @param verbose Logical, if TRUE, prints progress messages.
#'
#' @return A list containing cross-validation metrics.
#'
#' @concept semantic
#' @export
validate_cross_models <- function(semantic_results,
                                    stm_results = NULL,
                                    verbose = TRUE) {

  if (verbose) message("Running cross-analysis validation...")

  validation_results <- list()

  if (!is.null(semantic_results$clustering$topic_keywords) &&
      !is.null(stm_results$topic_keywords)) {

    if (verbose) message("Calculating topic-cluster correspondence...")

    correspondence <- calculate_topic_cluster_correspondence(
      semantic_results$clustering$topic_keywords,
      stm_results$topic_keywords
    )
    validation_results$topic_cluster_correspondence <- correspondence
  }

  if (!is.null(semantic_results$embeddings) &&
      !is.null(semantic_results$clustering$topic_assignments)) {

    if (verbose) message("Validating semantic coherence...")

    coherence <- validate_semantic_coherence(
      embeddings = semantic_results$embeddings,
      topic_assignments = semantic_results$clustering$topic_assignments
    )
    validation_results$semantic_coherence <- coherence
  }

  if (!is.null(semantic_results$clustering$topic_assignments) &&
      !is.null(stm_results$topic_assignments)) {

    if (verbose) message("Calculating assignment consistency...")

    consistency <- calculate_assignment_consistency(
      semantic_results$clustering$topic_assignments,
      stm_results$topic_assignments
    )
    validation_results$assignment_consistency <- consistency
  }

  if (verbose) message("Cross-analysis validation completed")

  return(validation_results)
}

#' @title Calculate Document Similarity with Fallbacks
#'
#' @description
#' Calculates document similarity with fallback methods and diagnostics.
#' Attempts embeddings first, falls back to Jaccard similarity if needed.
#'
#' @param texts Character vector of texts
#' @param method Similarity method ("embeddings" or "jaccard")
#' @param embedding_model Model name for embeddings (default: "all-MiniLM-L6-v2")
#' @param cache_embeddings Logical, cache embeddings (default: TRUE)
#' @param min_word_length Minimum word length for Jaccard (default: 3)
#' @param doc_names Optional document names
#'
#' @return List containing similarity matrix, method used, embeddings, and diagnostics
#'
#' @concept semantic
#' @export
#'
#' @examples
#' if (interactive()) {
#'   abstracts <- TextAnalysisR::SpecialEduTech$abstract[1:5]
#'   similarity_result <- calculate_similarity_robust(abstracts)
#'   print(similarity_result$similarity_matrix)
#'   print(similarity_result$diagnostics)
#' }
calculate_similarity_robust <- function(texts,
                                       method = "embeddings",
                                       embedding_model = "all-MiniLM-L6-v2",
                                       cache_embeddings = TRUE,
                                       min_word_length = 3,
                                       doc_names = NULL) {

  diagnostics <- list(
    attempted_methods = character(),
    warnings = character(),
    computation_time = NULL
  )

  start_time <- Sys.time()

  if (is.null(doc_names)) {
    doc_names <- paste0("doc", seq_along(texts))
  }

  if (method == "embeddings" && requireNamespace("reticulate", quietly = TRUE)) {
    diagnostics$attempted_methods <- c(diagnostics$attempted_methods, "embeddings")

    embedding_result <- tryCatch({
      embeddings <- generate_embeddings(
        texts = texts,
        model = embedding_model,
        verbose = FALSE
      )

      sklearn_metrics <- reticulate::import("sklearn.metrics.pairwise")
      similarity_matrix <- sklearn_metrics$cosine_similarity(embeddings)
      similarity_matrix <- as.matrix(similarity_matrix)

      rownames(similarity_matrix) <- doc_names
      colnames(similarity_matrix) <- doc_names

      list(
        success = TRUE,
        similarity_matrix = similarity_matrix,
        embeddings = embeddings,
        method = "embeddings"
      )
    }, error = function(e) {
      list(success = FALSE, error = e$message)
    })

    if (!embedding_result$success) {
      diagnostics$warnings <- c(diagnostics$warnings,
                                paste("Embeddings failed:", embedding_result$error))
    }

    if (embedding_result$success) {
      diagnostics$computation_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

      return(list(
        similarity_matrix = embedding_result$similarity_matrix,
        method_used = "embeddings",
        embeddings = embedding_result$embeddings,
        diagnostics = diagnostics
      ))
    }
  }

  diagnostics$attempted_methods <- c(diagnostics$attempted_methods, "jaccard")

  get_words <- function(text) {
    text <- tolower(text)
    text <- gsub("[^[:alnum:]\\s]", "", text)
    words <- unlist(strsplit(text, "\\s+"))
    words <- words[nchar(words) >= min_word_length]
    unique(words)
  }

  word_lists <- lapply(texts, get_words)
  n_docs <- length(texts)
  similarity_matrix <- matrix(0, nrow = n_docs, ncol = n_docs)

  for (i in seq_len(n_docs)) {
    for (j in seq_len(n_docs)) {
      if (i == j) {
        similarity_matrix[i, j] <- 1.0
      } else {
        words_i <- word_lists[[i]]
        words_j <- word_lists[[j]]

        if (length(words_i) == 0 && length(words_j) == 0) {
          similarity_matrix[i, j] <- 1.0
        } else if (length(words_i) == 0 || length(words_j) == 0) {
          similarity_matrix[i, j] <- 0.0
        } else {
          intersection <- length(intersect(words_i, words_j))
          union <- length(union(words_i, words_j))
          similarity_matrix[i, j] <- intersection / union
        }
      }
    }
  }

  rownames(similarity_matrix) <- doc_names
  colnames(similarity_matrix) <- doc_names

  diagnostics$computation_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

  return(list(
    similarity_matrix = similarity_matrix,
    method_used = "jaccard",
    embeddings = NULL,
    diagnostics = diagnostics
  ))
}


#' Calculate Clustering Quality Metrics
#'
#' @description
#' Calculates common clustering evaluation metrics including Silhouette Score,
#' Davies-Bouldin Index, and Calinski-Harabasz Index.
#'
#' @param clusters Integer vector of cluster assignments
#' @param data_matrix Numeric matrix of data points (rows = observations, cols = features)
#' @param dist_matrix Optional distance matrix. If NULL, computed from data_matrix
#' @param metrics Character vector of metrics to calculate.
#'   Options: "silhouette", "davies_bouldin", "calinski_harabasz", or "all" (default)
#'
#' @return A named list containing:
#'   \describe{
#'     \item{silhouette}{Silhouette score (-1 to 1, higher is better)}
#'     \item{davies_bouldin}{Davies-Bouldin index (lower is better)}
#'     \item{calinski_harabasz}{Calinski-Harabasz index (higher is better)}
#'     \item{n_clusters}{Number of clusters}
#'     \item{cluster_sizes}{Table of cluster sizes}
#'   }
#'
#' @details
#' - Silhouette Score: Measures how similar an object is to its own cluster compared
#'   to other clusters. Range: -1 to 1, higher is better.
#' - Davies-Bouldin Index: Average similarity between each cluster and its most
#'   similar cluster. Lower values indicate better clustering.
#' - Calinski-Harabasz Index: Ratio of between-cluster to within-cluster variance.
#'   Higher values indicate better-defined clusters.
#'
#' @concept semantic
#' @export
#'
#' @examples
#' \donttest{
#' abstracts <- TextAnalysisR::SpecialEduTech$abstract[1:20]
#' term_matrix <- as.matrix(quanteda::dfm(quanteda::tokens(abstracts)))
#' kmeans_result <- stats::kmeans(term_matrix, centers = 2)
#' metrics <- calculate_clustering_metrics(kmeans_result$cluster, term_matrix)
#' print(metrics)
#' }
calculate_clustering_metrics <- function(clusters,
                                            data_matrix,
                                            dist_matrix = NULL,
                                            metrics = "all") {
  if (is.null(clusters) || length(clusters) == 0) {
    return(list(
      silhouette = NA,
      davies_bouldin = NA,
      calinski_harabasz = NA,
      n_clusters = 0,
      cluster_sizes = NULL
    ))
  }

  data_matrix <- as.matrix(data_matrix)
  clusters <- as.numeric(as.factor(clusters))
  unique_clusters <- unique(clusters)
  n_clusters <- length(unique_clusters)
  n_points <- nrow(data_matrix)

  if (n_clusters <= 1) {
    return(list(
      silhouette = NA,
      davies_bouldin = NA,
      calinski_harabasz = NA,
      n_clusters = n_clusters,
      cluster_sizes = table(clusters)
    ))
  }

  if (metrics == "all") {
    metrics <- c("silhouette", "davies_bouldin", "calinski_harabasz")
  }

  result <- list(
    n_clusters = n_clusters,
    cluster_sizes = table(clusters)
  )

  if (is.null(dist_matrix)) {
    dist_matrix <- stats::dist(data_matrix)
  }

  if ("silhouette" %in% metrics) {
    result$silhouette <- tryCatch({
      if (requireNamespace("cluster", quietly = TRUE)) {
        sil_result <- cluster::silhouette(clusters, dist_matrix)
        mean(sil_result[, 3])
      } else {
        NA
      }
    }, error = function(e) NA)
  }

  if ("davies_bouldin" %in% metrics) {
    result$davies_bouldin <- tryCatch({
      centers <- matrix(0, nrow = n_clusters, ncol = ncol(data_matrix))
      for (i in seq_len(n_clusters)) {
        cluster_id <- unique_clusters[i]
        cluster_points <- data_matrix[clusters == cluster_id, , drop = FALSE]
        if (nrow(cluster_points) > 0) {
          centers[i, ] <- colMeans(cluster_points)
        }
      }

      within_distances <- numeric(n_clusters)
      for (i in seq_len(n_clusters)) {
        cluster_id <- unique_clusters[i]
        cluster_points <- data_matrix[clusters == cluster_id, , drop = FALSE]
        if (nrow(cluster_points) > 0) {
          distances <- apply(cluster_points, 1, function(x) sqrt(sum((x - centers[i, ])^2)))
          within_distances[i] <- mean(distances)
        }
      }

      db_values <- numeric(n_clusters)
      for (i in seq_len(n_clusters)) {
        max_ratio <- 0
        for (j in seq_len(n_clusters)) {
          if (i != j) {
            between_distance <- sqrt(sum((centers[i, ] - centers[j, ])^2))
            if (between_distance > 0) {
              ratio <- (within_distances[i] + within_distances[j]) / between_distance
              max_ratio <- max(max_ratio, ratio)
            }
          }
        }
        db_values[i] <- max_ratio
      }

      mean(db_values)
    }, error = function(e) NA)
  }

  if ("calinski_harabasz" %in% metrics) {
    result$calinski_harabasz <- tryCatch({
      if (n_clusters >= n_points) {
        return(NA)
      }

      overall_centroid <- colMeans(data_matrix)

      cluster_centroids <- matrix(0, nrow = n_clusters, ncol = ncol(data_matrix))
      cluster_sizes <- numeric(n_clusters)

      for (i in seq_len(n_clusters)) {
        cluster_id <- unique_clusters[i]
        cluster_points <- data_matrix[clusters == cluster_id, , drop = FALSE]
        if (nrow(cluster_points) > 0) {
          cluster_centroids[i, ] <- colMeans(cluster_points)
          cluster_sizes[i] <- nrow(cluster_points)
        }
      }

      ssb <- 0
      for (i in seq_len(n_clusters)) {
        if (cluster_sizes[i] > 0) {
          distance_to_overall <- sum((cluster_centroids[i, ] - overall_centroid)^2)
          ssb <- ssb + cluster_sizes[i] * distance_to_overall
        }
      }

      ssw <- 0
      for (i in seq_len(n_clusters)) {
        cluster_id <- unique_clusters[i]
        cluster_points <- data_matrix[clusters == cluster_id, , drop = FALSE]
        if (nrow(cluster_points) > 0) {
          for (j in seq_len(nrow(cluster_points))) {
            distance_to_centroid <- sum((cluster_points[j, ] - cluster_centroids[i, ])^2)
            ssw <- ssw + distance_to_centroid
          }
        }
      }

      if (ssw > 0 && (n_points - n_clusters) > 0) {
        (ssb / (n_clusters - 1)) / (ssw / (n_points - n_clusters))
      } else {
        NA
      }
    }, error = function(e) NA)
  }

  result
}


#' Calculate Cross-Matrix Cosine Similarity
#'
#' Calculates cosine similarity between two different embedding matrices,
#' useful for comparing documents/topics across different categories or groups.
#'
#' @param embeddings1 A numeric matrix where rows are items and columns are embedding dimensions.
#' @param embeddings2 A numeric matrix where rows are items and columns are embedding dimensions.
#'   Must have the same number of columns as embeddings1.
#' @param labels1 Optional character vector of labels for items in embeddings1.
#' @param labels2 Optional character vector of labels for items in embeddings2.
#' @param normalize Logical, whether to L2-normalize embeddings before computing similarity (default: TRUE).
#'
#' @return A list containing:
#'   \item{similarity_matrix}{Matrix of cosine similarities (nrow(embeddings1) x nrow(embeddings2))}
#'   \item{similarity_df}{Long-format data frame with columns: row_idx, col_idx, similarity, and optionally label1, label2}
#'
#' @concept semantic
#' @export
#'
#' @examples
#' \donttest{
#' abstracts <- TextAnalysisR::SpecialEduTech$abstract[1:6]
#' term_matrix <- as.matrix(quanteda::dfm(quanteda::tokens(abstracts)))
#' similarity_result <- calculate_cross_similarity(
#'   term_matrix[1:3, ], term_matrix[4:6, ],
#'   labels1 = paste("Doc", 1:3),
#'   labels2 = paste("Doc", 4:6)
#' )
#' similarity_result$similarity_matrix
#' }
calculate_cross_similarity <- function(embeddings1,
                                        embeddings2,
                                        labels1 = NULL,
                                        labels2 = NULL,
                                        normalize = TRUE) {

  embeddings1 <- as.matrix(embeddings1)
  embeddings2 <- as.matrix(embeddings2)

  if (ncol(embeddings1) != ncol(embeddings2)) {
    stop("Embedding matrices must have the same number of columns (dimensions)")
  }

  if (normalize) {
    norms1 <- sqrt(rowSums(embeddings1^2))
    norms1[norms1 == 0] <- 1
    norm_emb1 <- embeddings1 / norms1

    norms2 <- sqrt(rowSums(embeddings2^2))
    norms2[norms2 == 0] <- 1
    norm_emb2 <- embeddings2 / norms2
  } else {
    norm_emb1 <- embeddings1
    norm_emb2 <- embeddings2
  }

  similarity_matrix <- norm_emb1 %*% t(norm_emb2)

  similarity_df <- tidyr::expand_grid(
    row_idx = seq_len(nrow(embeddings1)),
    col_idx = seq_len(nrow(embeddings2))
  ) %>%
    dplyr::mutate(
      similarity = as.vector(similarity_matrix)
    )

  if (!is.null(labels1)) {
    if (length(labels1) != nrow(embeddings1)) {
      warning("labels1 length does not match number of rows in embeddings1")
    } else {
      similarity_df <- similarity_df %>%
        dplyr::mutate(label1 = labels1[row_idx])
    }
  }

  if (!is.null(labels2)) {
    if (length(labels2) != nrow(embeddings2)) {
      warning("labels2 length does not match number of rows in embeddings2")
    } else {
      similarity_df <- similarity_df %>%
        dplyr::mutate(label2 = labels2[col_idx])
    }
  }

  list(
    similarity_matrix = similarity_matrix,
    similarity_df = similarity_df
  )
}


#' Extract Cross-Category Similarities from Full Similarity Matrix
#'
#' Given a full similarity matrix and category information, extracts pairwise
#' similarities between a reference category and other categories into a long-format
#' data frame suitable for visualization and analysis.
#'
#' @param similarity_matrix A square similarity matrix (n x n).
#' @param docs_data A data frame containing document metadata with at least:
#'   \describe{
#'     \item{category_var}{Column indicating category membership}
#'     \item{id_var}{Column with unique document identifiers}
#'   }
#' @param reference_category Character string specifying the reference category to compare against.
#' @param compare_categories Character vector of categories to compare with the reference.
#'   If NULL, compares with all categories except reference.
#' @param category_var Name of the column containing category information (default: "category").
#' @param id_var Name of the column containing document IDs (default: "display_name").
#' @param name_var Optional name of column with display names (default: NULL, uses id_var).
#'
#' @return A data frame with columns:
#'   \item{ref_id}{Reference document ID}
#'   \item{ref_name}{Reference document name (if name_var provided)}
#'   \item{other_id}{Comparison document ID}
#'   \item{other_name}{Comparison document name (if name_var provided)}
#'   \item{other_category}{Category of comparison document}
#'   \item{similarity}{Cosine similarity value}
#'
#' @concept semantic
#' @export
#'
#' @examples
#' \donttest{
#' articles <- TextAnalysisR::SpecialEduTech[1:6, ]
#' articles$display_name <- paste0("d", seq_len(nrow(articles)))
#' term_matrix <- as.matrix(quanteda::dfm(quanteda::tokens(articles$abstract)))
#' normalized_matrix <- term_matrix / sqrt(rowSums(term_matrix ^ 2))
#' similarity_matrix <- normalized_matrix %*% t(normalized_matrix)
#' dimnames(similarity_matrix) <- list(articles$display_name, articles$display_name)
#' cross_similarities <- extract_cross_category_similarities(
#'   similarity_matrix  = similarity_matrix,
#'   docs_data          = articles,
#'   reference_category = "thesis",
#'   compare_categories = "journal_article",
#'   category_var       = "reference_type",
#'   id_var             = "display_name",
#'   name_var           = "title"
#' )
#' }
extract_cross_category_similarities <- function(similarity_matrix,
                                                 docs_data,
                                                 reference_category,
                                                 compare_categories = NULL,
                                                 category_var = "category",
                                                 id_var = "display_name",
                                                 name_var = NULL) {

  if (!requireNamespace("purrr", quietly = TRUE)) {
    stop("Package 'purrr' is required. Install with: install.packages('purrr')")
  }

  if (!category_var %in% names(docs_data)) {
    stop("category_var '", category_var, "' not found in docs_data")
  }
  if (!id_var %in% names(docs_data)) {
    stop("id_var '", id_var, "' not found in docs_data")
  }

  docs_data <- docs_data %>%
    dplyr::mutate(.row_idx = dplyr::row_number())

  ref_indices <- docs_data %>%
    dplyr::filter(.data[[category_var]] == reference_category) %>%
    dplyr::pull(.row_idx)

  if (length(ref_indices) == 0) {
    stop("No documents found in reference category: ", reference_category)
  }

  if (is.null(compare_categories)) {
    compare_categories <- unique(docs_data[[category_var]])
    compare_categories <- compare_categories[compare_categories != reference_category]
  }

  result_list <- list()

  for (comp_cat in compare_categories) {
    comp_indices <- docs_data %>%
      dplyr::filter(.data[[category_var]] == comp_cat) %>%
      dplyr::pull(.row_idx)

    if (length(comp_indices) == 0) next

    pairs <- tidyr::expand_grid(
      ref_idx = ref_indices,
      other_idx = comp_indices
    ) %>%
      dplyr::mutate(
        similarity = purrr::map2_dbl(ref_idx, other_idx, ~similarity_matrix[.x, .y]),
        other_category = comp_cat
      )

    result_list[[comp_cat]] <- pairs
  }

  result <- dplyr::bind_rows(result_list)

  ref_lookup <- docs_data %>%
    dplyr::select(.row_idx, !!rlang::sym(id_var)) %>%
    dplyr::rename(ref_id = !!rlang::sym(id_var))

  other_lookup <- docs_data %>%
    dplyr::select(.row_idx, !!rlang::sym(id_var)) %>%
    dplyr::rename(other_id = !!rlang::sym(id_var))

  result <- result %>%
    dplyr::left_join(ref_lookup, by = c("ref_idx" = ".row_idx")) %>%
    dplyr::left_join(other_lookup, by = c("other_idx" = ".row_idx"))

  if (!is.null(name_var) && name_var %in% names(docs_data)) {
    ref_names <- docs_data %>%
      dplyr::select(.row_idx, !!rlang::sym(name_var)) %>%
      dplyr::rename(ref_name = !!rlang::sym(name_var))

    other_names <- docs_data %>%
      dplyr::select(.row_idx, !!rlang::sym(name_var)) %>%
      dplyr::rename(other_name = !!rlang::sym(name_var))

    result <- result %>%
      dplyr::left_join(ref_names, by = c("ref_idx" = ".row_idx")) %>%
      dplyr::left_join(other_names, by = c("other_idx" = ".row_idx")) %>%
      dplyr::select(-ref_idx, -other_idx) %>%
      dplyr::select(ref_id, ref_name, other_id, other_name, other_category, similarity)
  } else {
    result <- result %>%
      dplyr::select(-ref_idx, -other_idx) %>%
      dplyr::select(ref_id, other_id, other_category, similarity)
  }

  result %>%
    dplyr::mutate(other_category = factor(other_category, levels = compare_categories))
}


#' Analyze Similarity Gaps Between Categories
#'
#' Identifies unique items, missing content, and cross-category learning opportunities
#' based on similarity thresholds. Useful for gap analysis in policy documents,
#' topic comparisons, or any cross-category similarity study.
#'
#' @param similarity_data A data frame with cross-category similarities, containing:
#'   \describe{
#'     \item{ref_var}{Reference item identifier}
#'     \item{other_var}{Comparison item identifier}
#'     \item{similarity_var}{Similarity score}
#'     \item{category_var}{Category of comparison item}
#'   }
#' @param ref_var Name of column with reference item IDs (default: "ref_id").
#' @param other_var Name of column with comparison item IDs (default: "other_id").
#' @param similarity_var Name of column with similarity values (default: "similarity").
#' @param category_var Name of column with category information (default: "other_category").
#' @param ref_label_var Optional column with reference item labels (for output).
#' @param other_label_var Optional column with comparison item labels (for output).
#' @param unique_threshold Threshold below which reference items are considered unique (default: 0.6).
#' @param cross_policy_min Minimum similarity for cross-policy opportunities (default: 0.6).
#' @param cross_policy_max Maximum similarity for cross-policy opportunities (default: 0.8).
#'
#' @return A list containing:
#'   \item{unique_items}{Data frame of reference items with low similarity (unique content)}
#'   \item{missing_items}{Data frame of comparison items with low similarity (content gaps)}
#'   \item{cross_policy}{Data frame of items with moderate similarity (learning opportunities)}
#'   \item{summary_stats}{Summary statistics by category}
#'
#' @concept semantic
#' @export
#'
#' @examples
#' \donttest{
#' articles <- TextAnalysisR::SpecialEduTech[1:6, ]
#' articles$display_name <- paste0("d", seq_len(nrow(articles)))
#' term_matrix <- as.matrix(quanteda::dfm(quanteda::tokens(articles$abstract)))
#' normalized_matrix <- term_matrix / sqrt(rowSums(term_matrix ^ 2))
#' similarity_matrix <- normalized_matrix %*% t(normalized_matrix)
#' dimnames(similarity_matrix) <- list(articles$display_name, articles$display_name)
#' cross_similarities <- extract_cross_category_similarities(
#'   similarity_matrix  = similarity_matrix,
#'   docs_data          = articles,
#'   reference_category = "thesis",
#'   compare_categories = "journal_article",
#'   category_var       = "reference_type",
#'   id_var             = "display_name"
#' )
#' gap_analysis <- analyze_similarity_gaps(
#'   similarity_data = cross_similarities,
#'   ref_var = "ref_id",
#'   other_var = "other_id",
#'   similarity_var = "similarity",
#'   category_var = "other_category",
#'   unique_threshold = 0.6
#' )
#' print(gap_analysis$summary_stats)
#' }
analyze_similarity_gaps <- function(similarity_data,
                                     ref_var = "ref_id",
                                     other_var = "other_id",
                                     similarity_var = "similarity",
                                     category_var = "other_category",
                                     ref_label_var = NULL,
                                     other_label_var = NULL,
                                     unique_threshold = 0.6,
                                     cross_policy_min = 0.6,
                                     cross_policy_max = 0.8) {

  for (v in c(ref_var, other_var, similarity_var, category_var)) {
    if (!v %in% names(similarity_data)) {
      stop("Column '", v, "' not found in similarity_data")
    }
  }

  summary_stats <- similarity_data %>%
    dplyr::group_by(.data[[category_var]]) %>%
    dplyr::summarise(
      mean_similarity = round(mean(.data[[similarity_var]], na.rm = TRUE), 3),
      median_similarity = round(stats::median(.data[[similarity_var]], na.rm = TRUE), 3),
      sd_similarity = round(stats::sd(.data[[similarity_var]], na.rm = TRUE), 3),
      min_similarity = round(min(.data[[similarity_var]], na.rm = TRUE), 3),
      max_similarity = round(max(.data[[similarity_var]], na.rm = TRUE), 3),
      n_pairs = dplyr::n(),
      .groups = "drop"
    )

  group_vars <- ref_var
  if (!is.null(ref_label_var) && ref_label_var %in% names(similarity_data)) {
    group_vars <- c(group_vars, ref_label_var)
  }

  unique_items <- similarity_data %>%
    dplyr::group_by(dplyr::across(dplyr::all_of(group_vars))) %>%
    dplyr::summarise(
      max_similarity = max(.data[[similarity_var]], na.rm = TRUE),
      min_similarity = min(.data[[similarity_var]], na.rm = TRUE),
      .max_idx = which.max(.data[[similarity_var]]),
      best_match = .data[[other_var]][.max_idx],
      best_match_category = .data[[category_var]][.max_idx],
      .groups = "drop"
    ) %>%
    dplyr::select(-.max_idx) %>%
    dplyr::filter(max_similarity < unique_threshold) %>%
    dplyr::mutate(gap_type = "Unique")

  other_group_vars <- c(other_var, category_var)
  if (!is.null(other_label_var) && other_label_var %in% names(similarity_data)) {
    other_group_vars <- c(other_group_vars, other_label_var)
  }

  missing_items <- similarity_data %>%
    dplyr::group_by(dplyr::across(dplyr::all_of(other_group_vars))) %>%
    dplyr::summarise(
      max_similarity = max(.data[[similarity_var]], na.rm = TRUE),
      min_similarity = min(.data[[similarity_var]], na.rm = TRUE),
      .max_idx = which.max(.data[[similarity_var]]),
      best_match = .data[[ref_var]][.max_idx],
      .groups = "drop"
    ) %>%
    dplyr::select(-.max_idx) %>%
    dplyr::filter(max_similarity < unique_threshold) %>%
    dplyr::mutate(gap_type = "Missing")

  cross_policy <- similarity_data %>%
    dplyr::group_by(dplyr::across(dplyr::all_of(group_vars))) %>%
    dplyr::summarise(
      max_similarity = max(.data[[similarity_var]], na.rm = TRUE),
      min_similarity = min(.data[[similarity_var]], na.rm = TRUE),
      .max_idx = which.max(.data[[similarity_var]]),
      best_match = .data[[other_var]][.max_idx],
      best_match_category = .data[[category_var]][.max_idx],
      .groups = "drop"
    ) %>%
    dplyr::select(-.max_idx) %>%
    dplyr::filter(
      max_similarity >= cross_policy_min,
      max_similarity < cross_policy_max
    ) %>%
    dplyr::mutate(gap_type = "Cross-Policy")

  list(
    unique_items = unique_items,
    missing_items = missing_items,
    cross_policy = cross_policy,
    summary_stats = summary_stats
  )
}


# Sentiment analysis

#' Analyze Text Sentiment
#'
#' @description
#' Performs sentiment analysis on text data using the syuzhet package.
#' Returns sentiment scores and classifications.
#'
#' @param texts Character vector of texts to analyze
#' @param method Sentiment analysis method: "syuzhet", "bing", "afinn", or "nrc" (default: "syuzhet")
#' @param doc_ids Optional character vector of document identifiers (default: NULL)
#'
#' @return A data frame with columns:
#'   \describe{
#'     \item{document}{Document identifier}
#'     \item{text}{Original text}
#'     \item{sentiment_score}{Numeric sentiment score}
#'     \item{sentiment}{Classification: "positive", "negative", or "neutral"}
#'   }
#'
#' @concept sentiment
#' @export
#'
#' @examples
#' \donttest{
#' abstracts <- TextAnalysisR::SpecialEduTech$abstract[1:10]
#' sentiment_results <- analyze_sentiment(abstracts)
#' print(sentiment_results)
#' }
analyze_sentiment <- function(texts,
                              method = "syuzhet",
                              doc_ids = NULL) {

  if (!requireNamespace("syuzhet", quietly = TRUE)) {
    stop("Package 'syuzhet' is required. Please install it with: install.packages('syuzhet')")
  }

  if (is.null(doc_ids)) {
    doc_ids <- paste0("doc", seq_along(texts))
  }

  sentiment_scores <- syuzhet::get_sentiment(texts, method = method)

  sentiment_classification <- ifelse(
    sentiment_scores > 0, "positive",
    ifelse(sentiment_scores < 0, "negative", "neutral")
  )

  data.frame(
    document = doc_ids,
    text = texts,
    sentiment_score = sentiment_scores,
    sentiment = sentiment_classification,
    stringsAsFactors = FALSE
  )
}


#' Plot Sentiment Distribution
#'
#' @description
#' Creates a bar plot showing the distribution of sentiment classifications.
#'
#' @param sentiment_data Data frame from analyze_sentiment() or with 'sentiment' column
#' @param title Plot title (default: "Sentiment Distribution")
#'
#' @return A ggplot2 bar chart
#'
#' @concept sentiment
#' @export
#'
#' @examples
#' \donttest{
#' abstracts <- TextAnalysisR::SpecialEduTech$abstract[1:10]
#' sentiment_data <- analyze_sentiment(abstracts)
#' sentiment_plot <- plot_sentiment_distribution(sentiment_data)
#' print(sentiment_plot)
#' }
plot_sentiment_distribution <- function(sentiment_data,
                                        title = "Sentiment Distribution") {

  if (!"sentiment" %in% names(sentiment_data)) {
    stop("Data must contain a 'sentiment' column. Use analyze_sentiment() first.")
  }

  sentiment_counts <- table(sentiment_data$sentiment)

  ordered_sentiments <- c("positive", "negative", "neutral")
  sentiment_counts <- sentiment_counts[ordered_sentiments[ordered_sentiments %in% names(sentiment_counts)]]

  colors <- get_sentiment_colors()

  plot_df <- data.frame(
    sentiment = factor(names(sentiment_counts), levels = names(sentiment_counts)),
    count = as.numeric(sentiment_counts)
  )

  plot_df$hover_text <- paste("Sentiment:", plot_df$sentiment, "<br>Count:", plot_df$count)

  ggplot2::ggplot(plot_df, ggplot2::aes(x = .data$sentiment, y = .data$count, fill = .data$sentiment,
                                         text = .data$hover_text)) +
    ggplot2::geom_col() +
    ggplot2::scale_fill_manual(values = colors, guide = "none") +
    ggplot2::labs(title = title, x = "Sentiment", y = "Number of Documents") +
    ggplot2::theme_minimal(base_size = 11)
}


#' Plot Sentiment by Category
#'
#' @description
#' Creates a grouped or stacked bar plot showing sentiment distribution across categories.
#'
#' @param sentiment_data Data frame with 'sentiment' column
#' @param category_var Name of the category variable column
#' @param plot_type Type of plot: "bar" or "stacked" (default: "bar")
#' @param title Plot title (default: auto-generated)
#'
#' @return A ggplot2 grouped/stacked bar chart
#'
#' @concept sentiment
#' @export
#'
#' @examples
#' \donttest{
#' articles <- TextAnalysisR::SpecialEduTech[1:20, ]
#' sentiment_results <- analyze_sentiment(articles$abstract)
#' sentiment_data <- cbind(
#'   reference_type = articles$reference_type,
#'   sentiment_results
#' )
#' sentiment_plot <- plot_sentiment_by_category(sentiment_data, "reference_type")
#' print(sentiment_plot)
#' }
plot_sentiment_by_category <- function(sentiment_data,
                                       category_var,
                                       plot_type = "bar",
                                       title = NULL) {

  if (!category_var %in% names(sentiment_data)) {
    stop(paste("Category variable", category_var, "not found in data"))
  }

  if (!"sentiment" %in% names(sentiment_data)) {
    stop("Data must contain a 'sentiment' column. Use analyze_sentiment() first.")
  }

  grouped_data <- sentiment_data %>%
    dplyr::group_by(!!rlang::sym(category_var), sentiment) %>%
    dplyr::summarise(count = dplyr::n(), .groups = "drop") %>%
    dplyr::group_by(!!rlang::sym(category_var)) %>%
    dplyr::mutate(proportion = count / sum(count)) %>%
    dplyr::ungroup()

  names(grouped_data)[1] <- "category_var"

  colors <- get_sentiment_colors()

  if (is.null(title)) {
    title <- paste("Sentiment by", category_var)
  }

  bar_position <- if (plot_type == "stacked") "stack" else "dodge"

  grouped_data$hover_text <- paste("Category:", grouped_data$category_var,
                                   "<br>Sentiment:", grouped_data$sentiment,
                                   "<br>Proportion:", round(grouped_data$proportion, 3))

  ggplot2::ggplot(grouped_data, ggplot2::aes(x = .data$category_var, y = .data$proportion, fill = .data$sentiment,
                                              text = .data$hover_text)) +
    ggplot2::geom_col(position = bar_position) +
    ggplot2::scale_fill_manual(values = colors) +
    ggplot2::labs(title = title, x = category_var, y = "Proportion", fill = "Sentiment") +
    ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))
}


#' Plot Document Sentiment Trajectory
#'
#' @description
#' Creates a line chart showing sentiment scores across documents with color gradient.
#'
#' @param sentiment_data Data frame from analyze_sentiment() with sentiment_score column
#' @param top_n Number of documents to display (default: NULL for all)
#' @param doc_ids Optional vector of custom document IDs for display (default: NULL)
#' @param text_preview Optional vector of text snippets for tooltips (default: NULL)
#' @param title Plot title (default: "Document Sentiment Scores")
#'
#' @return A ggplot2 line chart with color gradient
#'
#' @concept sentiment
#' @export
plot_document_sentiment_trajectory <- function(sentiment_data,
                                               top_n = NULL,
                                               doc_ids = NULL,
                                               text_preview = NULL,
                                               title = "Document Sentiment Scores") {

  doc_data <- sentiment_data %>%
    dplyr::arrange(document) %>%
    dplyr::mutate(doc_index = dplyr::row_number())

  if (!is.null(top_n)) {
    doc_data <- doc_data %>%
      dplyr::slice_head(n = top_n)
  }

  if (!is.null(doc_ids)) {
    doc_data$display_id <- doc_ids[doc_data$document]
  } else {
    doc_data$display_id <- paste("Doc", doc_data$document)
  }

  doc_data$hover_text <- paste("Document:", doc_data$display_id,
                               "<br>Score:", round(doc_data$sentiment_score, 3))

  ggplot2::ggplot(doc_data, ggplot2::aes(x = .data$doc_index, y = .data$sentiment_score, text = .data$hover_text)) +
    ggplot2::geom_line(color = "#6B7280") +
    ggplot2::geom_point(ggplot2::aes(color = .data$sentiment_score), size = 2) +
    ggplot2::scale_color_gradient2(
      low = "#EF4444", mid = "#6B7280", high = "#10B981", midpoint = 0,
      name = "Sentiment Score"
    ) +
    ggplot2::geom_hline(yintercept = 0, linetype = "dashed", color = "#9CA3AF") +
    ggplot2::labs(title = title, x = "Document Index", y = "Sentiment Score") +
    ggplot2::theme_minimal(base_size = 11)
}

#' Analyze Sentiment Using Tidytext Lexicons
#'
#' @description
#' Performs lexicon-based sentiment analysis on a DFM object using tidytext lexicons.
#' Supports AFINN, Bing, and NRC lexicons with scoring and emotion analysis.
#' Now supports n-grams for improved negation and intensifier handling.
#'
#' @param dfm_object A quanteda DFM object (unigram or n-gram)
#' @param lexicon Lexicon to use: "afinn", "bing", or "nrc" (default: "bing")
#' @param texts_df Optional data frame with original texts and metadata (default: NULL)
#' @param feature_type Feature space: "words" (unigrams) or "ngrams" (default: "words")
#' @param ngram_range N-gram size when feature_type = "ngrams" (default: 2 for bigrams)
#' @param texts Optional character vector of texts for n-gram creation (default: NULL)
#'
#' @return A list containing:
#'   \describe{
#'     \item{document_sentiment}{Data frame with sentiment scores per document}
#'     \item{emotion_scores}{Data frame with emotion scores (NRC only)}
#'     \item{summary_stats}{List of summary statistics}
#'     \item{feature_type}{Feature type used for analysis}
#'   }
#'
#' @importFrom tidytext tidy get_sentiments
#' @importFrom dplyr inner_join group_by summarise mutate case_when ungroup n_distinct
#' @importFrom tidyr pivot_wider pivot_longer
#' @concept sentiment
#' @export
#'
#' @examples
#' \donttest{
#' abstracts <- TextAnalysisR::SpecialEduTech$abstract[1:10]
#' corpus <- quanteda::corpus(abstracts)
#' dfm_object <- quanteda::dfm(quanteda::tokens(corpus))
#' lexicon_results <- sentiment_lexicon_analysis(dfm_object, lexicon = "bing")
#' print(lexicon_results$document_sentiment)
#' }
sentiment_lexicon_analysis <- function(dfm_object,
                                       lexicon = "bing",
                                       texts_df = NULL,
                                       feature_type = "words",
                                       ngram_range = 2,
                                       texts = NULL) {

  if (!requireNamespace("tidytext", quietly = TRUE)) {
    stop("Package 'tidytext' is required. Please install it.")
  }

  if (feature_type == "ngrams" && !is.null(texts)) {
    if (!requireNamespace("quanteda", quietly = TRUE)) {
      stop("Package 'quanteda' is required for n-gram analysis.")
    }

    message("Creating ", ngram_range, "-gram DFM for sentiment analysis (handles negation and intensifiers)")

    corp <- quanteda::corpus(texts)
    toks <- quanteda::tokens(corp, remove_punct = TRUE, remove_symbols = TRUE)
    toks_ngrams <- quanteda::tokens_ngrams(toks, n = ngram_range)
    dfm_object <- quanteda::dfm(toks_ngrams)
  }

  tidy_dfm <- tidytext::tidy(dfm_object)
  lexicon_name <- tolower(lexicon)

  sentiment_lexicon <- switch(
    lexicon_name,
    "afinn" = tidytext::get_sentiments("afinn"),
    "bing" = tidytext::get_sentiments("bing"),
    "nrc" = tidytext::get_sentiments("nrc"),
    stop("Invalid lexicon. Choose 'afinn', 'bing', or 'nrc'.")
  )

  doc_names <- quanteda::docnames(dfm_object)

  if (lexicon_name == "afinn") {
    doc_sentiment <- tidy_dfm %>%
      dplyr::inner_join(sentiment_lexicon, by = c("term" = "word"), relationship = "many-to-many") %>%
      dplyr::group_by(document) %>%
      dplyr::summarise(
        sentiment_score = sum(value * count, na.rm = TRUE),
        n_words = sum(count),
        avg_sentiment = sentiment_score / n_words,
        sentiment = dplyr::case_when(
          avg_sentiment > 0.5 ~ "positive",
          avg_sentiment < -0.5 ~ "negative",
          TRUE ~ "neutral"
        ),
        .groups = "drop"
      )
  } else if (lexicon_name == "bing") {
    doc_sentiment <- tidy_dfm %>%
      dplyr::inner_join(sentiment_lexicon, by = c("term" = "word"), relationship = "many-to-many") %>%
      dplyr::group_by(document, sentiment) %>%
      dplyr::summarise(n = sum(count), .groups = "drop") %>%
      tidyr::pivot_wider(names_from = sentiment, values_from = n, values_fill = 0)

    if (!("positive" %in% names(doc_sentiment))) {
      doc_sentiment$positive <- 0
    }
    if (!("negative" %in% names(doc_sentiment))) {
      doc_sentiment$negative <- 0
    }

    doc_sentiment <- doc_sentiment %>%
      dplyr::mutate(
        sentiment_score = positive - negative,
        total_sentiment_words = positive + negative,
        sentiment = dplyr::case_when(
          sentiment_score > 0 ~ "positive",
          sentiment_score < 0 ~ "negative",
          TRUE ~ "neutral"
        )
      )
  } else if (lexicon_name == "nrc") {
    doc_sentiment <- tidy_dfm %>%
      dplyr::inner_join(sentiment_lexicon, by = c("term" = "word"), relationship = "many-to-many") %>%
      dplyr::group_by(document, sentiment) %>%
      dplyr::summarise(n = sum(count), .groups = "drop") %>%
      tidyr::pivot_wider(names_from = sentiment, values_from = n, values_fill = 0)

    if ("positive" %in% names(doc_sentiment) && "negative" %in% names(doc_sentiment)) {
      doc_sentiment <- doc_sentiment %>%
        dplyr::mutate(
          sentiment_score = positive - negative,
          sentiment = dplyr::case_when(
            sentiment_score > 0 ~ "positive",
            sentiment_score < 0 ~ "negative",
            TRUE ~ "neutral"
          )
        )
    }
  }

  emotion_data <- NULL
  if (lexicon_name == "nrc") {
    emotion_cols <- c("anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust")
    available_emotions <- intersect(emotion_cols, names(doc_sentiment))
    if (length(available_emotions) > 0) {
      emotion_data <- doc_sentiment %>%
        dplyr::select(document, dplyr::all_of(available_emotions)) %>%
        tidyr::pivot_longer(cols = -document, names_to = "emotion", values_to = "score") %>%
        dplyr::group_by(emotion) %>%
        dplyr::summarise(total_score = sum(score, na.rm = TRUE), .groups = "drop")
    }
  }

  total_docs_in_corpus <- length(doc_names)
  docs_analyzed <- dplyr::n_distinct(doc_sentiment$document)

  summary_stats <- list(
    total_documents = total_docs_in_corpus,
    documents_analyzed = docs_analyzed,
    documents_without_sentiment = total_docs_in_corpus - docs_analyzed,
    coverage_percentage = round((docs_analyzed / total_docs_in_corpus) * 100, 1),
    positive_docs = sum(doc_sentiment$sentiment == "positive", na.rm = TRUE),
    negative_docs = sum(doc_sentiment$sentiment == "negative", na.rm = TRUE),
    neutral_docs = sum(doc_sentiment$sentiment == "neutral", na.rm = TRUE),
    avg_sentiment_score = mean(doc_sentiment$sentiment_score, na.rm = TRUE)
  )

  list(
    document_sentiment = doc_sentiment,
    emotion_scores = emotion_data,
    summary_stats = summary_stats,
    lexicon_used = lexicon_name,
    feature_type = feature_type
  )
}


#' Embedding-based Sentiment Analysis
#'
#' @description
#' Performs sentiment analysis using transformer-based embeddings and neural models.
#' This approach uses pre-trained language models for contextual sentiment detection
#' without requiring sentiment lexicons. Particularly effective for handling:
#' - Complex contextual sentiment
#' - Implicit sentiment and sarcasm
#' - Domain-specific sentiment
#' - Negation and intensifiers (automatically handled by the model)
#'
#' @param texts Character vector of texts to analyze
#' @param embeddings Optional pre-computed embedding matrix (from generate_embeddings)
#' @param model_name Sentiment model name (default: "distilbert-base-uncased-finetuned-sst-2-english")
#' @param doc_names Optional document names/IDs
#' @param use_gpu Whether to use GPU if available (default: FALSE)
#'
#' @return A list containing:
#'   \describe{
#'     \item{document_sentiment}{Data frame with document-level sentiment scores and classifications}
#'     \item{emotion_scores}{NULL (emotion detection not currently supported for embeddings)}
#'     \item{summary_stats}{Summary statistics including document counts and average scores}
#'     \item{model_used}{Name of the transformer model used}
#'     \item{feature_type}{"embeddings"}
#'   }
#'
#' @concept sentiment
#' @export
#'
#' @examples
#' if (interactive()) {
#'   abstracts <- TextAnalysisR::SpecialEduTech$abstract[1:10]
#'   embedding_sentiment <- sentiment_embedding_analysis(abstracts)
#'   print(embedding_sentiment$document_sentiment)
#'   print(embedding_sentiment$summary_stats)
#' }
sentiment_embedding_analysis <- function(texts,
                                        embeddings = NULL,
                                        model_name = "distilbert-base-uncased-finetuned-sst-2-english",
                                        doc_names = NULL,
                                        use_gpu = FALSE) {

  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required for embedding-based sentiment analysis. Please install it.")
  }

  if (is.null(doc_names)) {
    doc_names <- paste0("text", seq_along(texts))
  }

  if (length(texts) != length(doc_names)) {
    stop("Length of texts and doc_names must match")
  }

  tryCatch({
    transformers <- reticulate::import("transformers")

    message("Loading sentiment model: ", model_name)

    device <- if (use_gpu) 0L else -1L

    sentiment_pipeline <- transformers$pipeline(
      "sentiment-analysis",
      model = model_name,
      device = device
    )

    message("Analyzing sentiment for ", length(texts), " documents...")

    results <- sentiment_pipeline(texts, truncation = TRUE, max_length = 512L)

    doc_sentiment <- data.frame(
      document = doc_names,
      label = sapply(results, function(x) tolower(x$label)),
      confidence = sapply(results, function(x) x$score),
      stringsAsFactors = FALSE
    )

    doc_sentiment$sentiment_score <- ifelse(
      doc_sentiment$label == "positive",
      doc_sentiment$confidence,
      -doc_sentiment$confidence
    )

    doc_sentiment$sentiment <- doc_sentiment$label

    if ("neg" %in% doc_sentiment$label || "negative" %in% doc_sentiment$label) {
      doc_sentiment$sentiment <- ifelse(
        doc_sentiment$label %in% c("neg", "negative"),
        "negative",
        "positive"
      )
    }

    summary_stats <- list(
      total_documents = length(texts),
      documents_analyzed = nrow(doc_sentiment),
      documents_without_sentiment = 0,
      coverage_percentage = 100,
      positive_docs = sum(doc_sentiment$sentiment == "positive", na.rm = TRUE),
      negative_docs = sum(doc_sentiment$sentiment == "negative", na.rm = TRUE),
      neutral_docs = sum(doc_sentiment$sentiment == "neutral", na.rm = TRUE),
      avg_sentiment_score = mean(doc_sentiment$sentiment_score, na.rm = TRUE),
      avg_confidence = mean(doc_sentiment$confidence, na.rm = TRUE)
    )

    message("Sentiment analysis completed successfully")

    return(list(
      document_sentiment = doc_sentiment,
      emotion_scores = NULL,
      summary_stats = summary_stats,
      model_used = model_name,
      feature_type = "embeddings"
    ))

  }, error = function(e) {
    stop(
      "Error in embedding-based sentiment analysis: ", e$message, "\n",
      "Please ensure Python transformers library is installed:\n",
      "  reticulate::py_install('transformers')\n",
      "  reticulate::py_install('torch')"
    )
  })
}


#' LLM-based Sentiment Analysis
#'
#' @description
#' Analyzes sentiment using Large Language Models (OpenAI, Gemini, or Ollama).
#' Provides nuanced sentiment understanding including sarcasm detection,
#' mixed emotions, and contextual interpretation that lexicon-based methods miss.
#'
#' @param texts Character vector of texts to analyze.
#' @param doc_names Optional character vector of document names (default: text1, text2, ...).
#' @param provider AI provider to use: "openai" (default), "gemini", or "ollama".
#' @param model Model name. If NULL, uses provider defaults: "gpt-4.1-mini" (OpenAI),
#'   "gemini-2.5-flash-lite" (Gemini), "llama3.2" (Ollama).
#' @param api_key API key for OpenAI or Gemini. If NULL, uses environment variable.
#'   Not required for Ollama.
#' @param batch_size Number of texts to process per API call (default: 5).
#'   Larger batches are more efficient but may hit token limits.
#' @param include_explanation Logical, if TRUE includes natural language explanation
#'   for each sentiment classification (default: FALSE).
#' @param verbose Logical, if TRUE prints progress messages (default: TRUE).
#'
#' @return A list containing:
#' \describe{
#'   \item{document_sentiment}{Data frame with document-level sentiment scores}
#'   \item{summary_stats}{Summary statistics of the analysis}
#'   \item{model_used}{Model name used for analysis}
#'   \item{provider}{AI provider used}
#' }
#'
#' @details
#' LLM-based sentiment analysis offers several advantages over lexicon methods:
#' \itemize{
#'   \item Understands context and nuance
#'   \item Detects sarcasm and irony
#'   \item Handles mixed emotions
#'   \item Works across domains without retraining
#' }
#'
#' @concept sentiment
#' @export
#'
#' @examples
#' if (interactive()) {
#'   abstracts <- TextAnalysisR::SpecialEduTech$abstract[1:5]
#'
#'   sentiment_openai <- analyze_sentiment_llm(abstracts, provider = "openai")
#'
#'   sentiment_gemini <- analyze_sentiment_llm(abstracts, provider = "gemini",
#'                                              include_explanation = TRUE)
#'
#'   sentiment_ollama <- analyze_sentiment_llm(abstracts, provider = "ollama",
#'                                              model = "llama3")
#' }
analyze_sentiment_llm <- function(texts,
                                  doc_names = NULL,
                                  provider = c("openai", "gemini", "ollama"),
                                  model = NULL,
                                  api_key = NULL,
                                  batch_size = 5,
                                  include_explanation = FALSE,
                                  verbose = TRUE) {

  provider <- match.arg(provider)

  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Package 'jsonlite' is required. Install with: install.packages('jsonlite')")
  }

  # Set document names if not provided
  doc_names <- doc_names %||% paste0("text", seq_along(texts))
  if (length(texts) != length(doc_names)) {
    stop("Length of texts and doc_names must match")
  }

  # Resolve provider, model, and API key
  setup <- .resolve_llm_setup(
    provider, model, api_key,
    defaults = list(openai = "gpt-4.1-mini", gemini = "gemini-2.5-flash-lite", ollama = "llama3.2")
  )
  model <- setup$model
  api_key <- setup$api_key

  if (verbose) {
    message(sprintf("Analyzing sentiment for %d texts using %s (%s)...",
                    length(texts), provider, model))
  }

  # System prompt for sentiment analysis
  system_prompt <- "You are an expert sentiment analyst. Analyze the sentiment of each text and respond in valid JSON format only.

For each text, provide:
- sentiment: 'positive', 'negative', or 'neutral'
- score: a number from -1.0 (very negative) to 1.0 (very positive)
- confidence: a number from 0.0 to 1.0 indicating your confidence
"

  if (include_explanation) {
    system_prompt <- paste0(system_prompt, "- explanation: a brief explanation of why you classified it this way\n")
  }

  system_prompt <- paste0(system_prompt, "
Respond with a JSON array. Example:
[{\"sentiment\": \"positive\", \"score\": 0.8, \"confidence\": 0.95}]

Important:
- Detect sarcasm and irony (e.g., 'Oh great, another meeting' is likely negative)
- Consider context and nuance
- Respond ONLY with valid JSON, no additional text")

  # Process texts in batches
  results <- .empty_sentiment_row(character(0))

  n_batches <- ceiling(length(texts) / batch_size)

  for (i in seq_len(n_batches)) {
    start_idx <- (i - 1) * batch_size + 1
    end_idx <- min(i * batch_size, length(texts))
    batch_texts <- texts[start_idx:end_idx]
    batch_names <- doc_names[start_idx:end_idx]

    if (verbose) {
      message(sprintf("Processing batch %d/%d...", i, n_batches))
    }

    # Build user prompt with numbered texts
    user_prompt <- paste0(
      "Analyze the sentiment of these texts:\n\n",
      paste(sprintf("Text %d: %s\n", seq_along(batch_texts), batch_texts), collapse = "\n")
    )

    # Call LLM
    response <- tryCatch(
      call_llm_api(
        provider = provider, system_prompt = system_prompt, user_prompt = user_prompt,
        model = model, temperature = 0, max_tokens = 500, api_key = api_key
      ),
      error = function(e) {
        warning(sprintf("Batch %d failed: %s", i, e$message))
        NULL
      }
    )

    if (is.null(response)) {
      # Failed batch: NA rows
      results <- rbind(results, .empty_sentiment_row(batch_names))
      next
    }

    # Parse JSON response
    parsed <- tryCatch({
      json_str <- response
      start_bracket <- regexpr("\\[", json_str)
      end_bracket <- regexpr("\\](?=[^\\]]*$)", json_str, perl = TRUE)
      if (start_bracket > 0 && end_bracket > 0) {
        json_str <- substr(json_str, start_bracket, end_bracket + attr(end_bracket, "match.length") - 1)
      }
      jsonlite::fromJSON(json_str)
    }, error = function(e) {
      warning(sprintf("Failed to parse JSON response for batch %d: %s", i, e$message))
      NULL
    })

    if (is.null(parsed) || length(parsed) == 0) {
      results <- rbind(results, .empty_sentiment_row(batch_names))
      next
    }

    # Map parsed results to batch
    for (j in seq_along(batch_texts)) {
      if (j <= nrow(parsed)) {
        row <- parsed[j, ]
        results <- rbind(results, data.frame(
          document = batch_names[j],
          sentiment = as.character(row$sentiment),
          sentiment_score = as.numeric(row$score),
          confidence = as.numeric(row$confidence),
          explanation = if (include_explanation && "explanation" %in% names(row)) as.character(row$explanation) else NA_character_,
          stringsAsFactors = FALSE
        ))
      } else {
        results <- rbind(results, .empty_sentiment_row(batch_names[j]))
      }
    }

    # Rate limiting
    Sys.sleep(if (provider %in% c("openai", "gemini")) 1 else 0.5)
  }

  # Remove explanation column if not requested
  if (!include_explanation) {
    results$explanation <- NULL
  }

  # Summary statistics
  valid_results <- results[!is.na(results$sentiment), ]
  summary_stats <- list(
    total_documents = length(texts),
    documents_analyzed = nrow(valid_results),
    documents_without_sentiment = sum(is.na(results$sentiment)),
    coverage_percentage = round(nrow(valid_results) / length(texts) * 100, 1),
    positive_docs = sum(valid_results$sentiment == "positive", na.rm = TRUE),
    negative_docs = sum(valid_results$sentiment == "negative", na.rm = TRUE),
    neutral_docs = sum(valid_results$sentiment == "neutral", na.rm = TRUE),
    avg_sentiment_score = mean(valid_results$sentiment_score, na.rm = TRUE),
    avg_confidence = mean(valid_results$confidence, na.rm = TRUE)
  )

  if (verbose) {
    message(sprintf("Sentiment analysis completed. Analyzed %d/%d documents.",
                    summary_stats$documents_analyzed, summary_stats$total_documents))
    message(sprintf("Distribution: %d positive, %d negative, %d neutral",
                    summary_stats$positive_docs, summary_stats$negative_docs,
                    summary_stats$neutral_docs))
  }

  return(list(
    document_sentiment = results,
    summary_stats = summary_stats,
    model_used = model,
    provider = provider,
    feature_type = "llm"
  ))
}


#' Plot Emotion Radar Chart
#'
#' @description
#' Creates a polar/radar chart for NRC emotion analysis with optional grouping.
#'
#' @param emotion_data Data frame with emotion scores (columns: emotion, total_score)
#' @param group_var Optional grouping variable column name for overlaid radars (default: NULL)
#' @param normalize Logical, whether to normalize scores to 0-100 scale (default: FALSE)
#' @param title Plot title (default: "Emotion Analysis")
#'
#' @return A ggplot2 polar chart
#'
#' @concept sentiment
#' @export
plot_emotion_radar <- function(emotion_data,
                               group_var = NULL,
                               normalize = FALSE,
                               title = "Emotion Analysis") {

  if (!is.null(group_var) && group_var %in% names(emotion_data)) {

    plot_data <- emotion_data

    if (normalize) {
      plot_data <- plot_data %>%
        dplyr::group_by(!!rlang::sym(group_var)) %>%
        dplyr::mutate(
          total_score = if (max(total_score, na.rm = TRUE) > 0) {
            (total_score / max(total_score, na.rm = TRUE)) * 100
          } else {
            total_score
          }
        ) %>%
        dplyr::ungroup()
    }

    categories <- unique(plot_data[[group_var]])

    if (length(categories) == 0) {
      return(
        ggplot2::ggplot() +
          ggplot2::annotate("text", x = 0.5, y = 0.5, label = "No data available", size = 5, color = "#ef4444") +
          ggplot2::theme_void()
      )
    }

    plot_data$group_col <- plot_data[[group_var]]
    plot_data$hover_text <- paste("Emotion:", plot_data$emotion,
                                  "<br>Score:", round(plot_data$total_score, 2),
                                  paste0("<br>", group_var, ":"), plot_data$group_col)

    ggplot2::ggplot(plot_data, ggplot2::aes(x = .data$emotion, y = .data$total_score,
                                             group = .data$group_col, color = .data$group_col, fill = .data$group_col,
                                             text = .data$hover_text)) +
      ggplot2::geom_polygon(alpha = 0.1) +
      ggplot2::geom_point(size = 2) +
      ggplot2::coord_polar() +
      ggplot2::labs(title = title, x = NULL, y = NULL, color = group_var, fill = group_var) +
      ggplot2::theme_minimal(base_size = 11)

  } else {

    scores <- emotion_data$total_score

    if (normalize) {
      max_score <- max(scores, na.rm = TRUE)
      if (max_score > 0) {
        scores <- (scores / max_score) * 100
      }
    }

    radar_df <- data.frame(
      emotion = emotion_data$emotion,
      score = scores
    )

    radar_df$hover_text <- paste("Emotion:", radar_df$emotion, "<br>Score:", round(radar_df$score, 2))

    ggplot2::ggplot(radar_df, ggplot2::aes(x = .data$emotion, y = .data$score, group = 1, text = .data$hover_text)) +
      ggplot2::geom_polygon(fill = "#8B5CF6", alpha = 0.1, color = "#8B5CF6") +
      ggplot2::geom_point(color = "#8B5CF6", size = 2) +
      ggplot2::coord_polar() +
      ggplot2::labs(title = title, x = NULL, y = NULL) +
      ggplot2::theme_minimal(base_size = 11)
  }
}


#' Plot Sentiment Box Plot by Category
#'
#' @description
#' Creates a box plot showing sentiment score distribution by category.
#'
#' @param sentiment_data Data frame from analyze_sentiment() containing sentiment_score
#'   and category columns
#' @param category_var Name of the category variable column (default: "category_var")
#' @param title Plot title (default: "Sentiment Score Distribution")
#'
#' @return A ggplot2 box plot
#'
#' @concept sentiment
#' @export
plot_sentiment_boxplot <- function(sentiment_data,
                                   category_var = "category_var",
                                   title = "Sentiment Score Distribution") {

  if (!category_var %in% names(sentiment_data)) {
    stop("Category variable '", category_var, "' not found in data")
  }

  if (!"sentiment_score" %in% names(sentiment_data)) {
    stop("sentiment_score column not found in data")
  }

  sentiment_data$hover_text <- paste(category_var, ":", sentiment_data[[category_var]],
                                     "<br>Score:", round(sentiment_data$sentiment_score, 3))

  ggplot2::ggplot(sentiment_data, ggplot2::aes(x = .data[[category_var]], y = .data$sentiment_score,
                                                fill = .data[[category_var]], text = .data$hover_text)) +
    ggplot2::geom_boxplot(show.legend = FALSE) +
    ggplot2::labs(title = title, x = category_var, y = "Sentiment Score") +
    ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))
}


#' Plot Sentiment Violin Plot by Category
#'
#' @description
#' Creates a violin plot showing sentiment score distribution by category.
#'
#' @param sentiment_data Data frame from analyze_sentiment() containing sentiment_score
#'   and category columns
#' @param category_var Name of the category variable column (default: "category_var")
#' @param title Plot title (default: "Sentiment Score Distribution")
#'
#' @return A ggplot2 violin plot
#'
#' @concept sentiment
#' @export
plot_sentiment_violin <- function(sentiment_data,
                                  category_var = "category_var",
                                  title = "Sentiment Score Distribution") {

  if (!category_var %in% names(sentiment_data)) {
    stop("Category variable '", category_var, "' not found in data")
  }

  if (!"sentiment_score" %in% names(sentiment_data)) {
    stop("sentiment_score column not found in data")
  }

  sentiment_data$hover_text <- paste(category_var, ":", sentiment_data[[category_var]],
                                     "<br>Score:", round(sentiment_data$sentiment_score, 3))

  ggplot2::ggplot(sentiment_data, ggplot2::aes(x = .data[[category_var]], y = .data$sentiment_score,
                                                fill = .data[[category_var]], text = .data$hover_text)) +
    ggplot2::geom_violin(show.legend = FALSE) +
    ggplot2::labs(title = title, x = category_var, y = "Sentiment Score") +
    ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))
}


# Semantic network analysis (co-occurrence and correlation)

#' @importFrom utils modifyList
#' @importFrom stats cor
#' @importFrom igraph graph_from_data_frame V vcount degree betweenness closeness eigen_centrality layout_with_fr
#' @importFrom dplyr count filter mutate select group_by summarise ungroup
#' @importFrom tibble as_tibble
#' @importFrom tidytext tidy
#' @importFrom widyr pairwise_count pairwise_cor
#' @importFrom stats quantile
#' @importFrom shiny showNotification
#' @importFrom rlang sym %||%
#' @importFrom utils head
#' @importFrom grDevices colorRampPalette
NULL

# Network Analysis Functions
# NOTE: word_co_occurrence_network and word_correlation_network functions
# are now in R/network_analysis.R using ggplot2-based visualization.

#' @title Plot Semantic Analysis Visualization
#'
#' @description
#' Creates interactive visualizations for semantic analysis results including
#' similarity heatmaps, dimensionality reduction plots, and clustering visualizations.
#'
#' @param analysis_result A list containing semantic analysis results from functions like
#'   semantic_similarity_analysis(), semantic_document_clustering(), or reduce_dimensions().
#' @param plot_type Type of visualization: "similarity" for heatmap, "dimensionality_reduction"
#'   for scatter plot, or "clustering" for cluster visualization (default: "similarity").
#' @param data_labels Optional character vector of labels for data points (default: NULL).
#' @param color_by Optional variable to color points by in scatter plots (default: NULL).
#' @param coords Optional pre-computed coordinates for dimensionality reduction plots (default: NULL).
#' @param clusters Optional cluster assignments vector (default: NULL).
#' @param hover_text Optional custom hover text for points (default: NULL).
#' @param hover_config Optional hover configuration list (default: NULL).
#' @param cluster_colors Optional color palette for clusters (default: NULL).
#' @param height The height of the resulting Plotly plot, in pixels (default: 600).
#' @param width The width of the resulting Plotly plot, in pixels (default: 800).
#' @param title Optional custom title for the plot (default: NULL).
#'
#' @return A ggplot2 object showing the specified visualization.
#'
#' @concept visualization
#' @export
#'
#' @examples
#' if (interactive()) {
#'   data(SpecialEduTech)
#'   texts <- SpecialEduTech$abstract[1:5]
#'   result <- semantic_similarity_analysis(texts)
#'   plot <- plot_semantic_viz(result, plot_type = "similarity")
#'   print(plot)
#' }
plot_semantic_viz <- function(analysis_result = NULL,
                                       plot_type = "similarity",
                                       data_labels = NULL,
                                       color_by = NULL,
                                       height = 600,
                                       width = 800,
                                       title = NULL,
                                       coords = NULL,
                                       clusters = NULL,
                                       hover_text = NULL,
                                       hover_config = NULL,
                                       cluster_colors = NULL) {

  tryCatch({
    plot_obj <- switch(plot_type,
      "similarity" = {
        similarity_matrix <- analysis_result$similarity_matrix

        if (is.null(data_labels)) {
          data_labels <- paste0("Doc ", seq_len(nrow(similarity_matrix)))
        }

        plot_title <- title %||% paste("Similarity Heatmap -", analysis_result$method)

        heat_df <- expand.grid(
          x = seq_len(ncol(similarity_matrix)),
          y = seq_len(nrow(similarity_matrix))
        )
        heat_df$similarity <- as.vector(similarity_matrix)
        heat_df$x_label <- factor(data_labels[heat_df$x], levels = data_labels)
        heat_df$y_label <- factor(data_labels[heat_df$y], levels = data_labels)

        heat_df$hover_text <- paste("X:", heat_df$x_label, "<br>Y:", heat_df$y_label,
                                    "<br>Similarity:", round(heat_df$similarity, 3))

        ggplot2::ggplot(heat_df, ggplot2::aes(x = .data$x_label, y = .data$y_label, fill = .data$similarity,
                                               text = .data$hover_text)) +
          ggplot2::geom_tile() +
          ggplot2::scale_fill_viridis_c(name = "Similarity") +
          ggplot2::labs(title = plot_title, x = "Documents", y = "Documents") +
          ggplot2::theme_minimal(base_size = 11) +
          ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))
      },
      "dimensionality_reduction" = {
        reduced_data <- if (!is.null(coords)) {
          coords
        } else if (!is.null(analysis_result$reduced_data)) {
          analysis_result$reduced_data
        } else {
          stop("No dimensionality reduction data available")
        }

        if (is.null(data_labels)) {
          data_labels <- paste0("Doc ", seq_len(nrow(reduced_data)))
        }

        plot_clusters <- !is.null(clusters) || (!is.null(analysis_result) && !is.null(analysis_result$clusters))
        cluster_data <- clusters %||% (if (!is.null(analysis_result)) analysis_result$clusters else NULL)

        scatter_df <- data.frame(
          x = reduced_data[, 1],
          y = if (ncol(reduced_data) > 1) reduced_data[, 2] else rep(0, nrow(reduced_data)),
          label = data_labels
        )

        if (plot_clusters) {
          scatter_df$color_var <- as.factor(cluster_data)
        } else if (!is.null(color_by)) {
          scatter_df$color_var <- color_by
        }

        plot_title <- title %||% paste("Dimensionality Reduction -",
                                        if (!is.null(analysis_result)) analysis_result$method else "Custom")

        x_title <- paste("Component 1",
                          if (!is.null(analysis_result) && !is.null(analysis_result$variance_explained))
                            paste0("(", round(analysis_result$variance_explained[1] * 100, 1), "%)")
                          else "")
        y_title <- paste("Component 2",
                          if (!is.null(analysis_result) && !is.null(analysis_result$variance_explained) &&
                              length(analysis_result$variance_explained) > 1)
                            paste0("(", round(analysis_result$variance_explained[2] * 100, 1), "%)")
                          else "")

        scatter_df$hover_text <- paste("Document:", scatter_df$label,
                                       "<br>X:", round(scatter_df$x, 3),
                                       "<br>Y:", round(scatter_df$y, 3))

        p <- ggplot2::ggplot(scatter_df, ggplot2::aes(x = .data$x, y = .data$y, text = .data$hover_text))

        if ("color_var" %in% names(scatter_df)) {
          p <- p + ggplot2::geom_point(ggplot2::aes(color = .data$color_var), size = 2, alpha = 0.7)
        } else {
          p <- p + ggplot2::geom_point(color = "steelblue", size = 2, alpha = 0.7)
        }

        p + ggplot2::labs(title = plot_title, x = x_title, y = y_title, color = NULL) +
          ggplot2::theme_minimal(base_size = 11)
      },
      "clustering" = {
        plot_data_mat <- if (!is.null(coords)) {
          coords
        } else if (!is.null(analysis_result)) {
          analysis_result$umap_embedding %||% analysis_result$reduced_data
        } else {
          stop("No clustering visualization data available")
        }

        cluster_data <- clusters %||% (if (!is.null(analysis_result)) analysis_result$clusters else NULL)

        plot_title <- title %||% paste("Clustering Results -",
                                        if (!is.null(analysis_result)) analysis_result$method else "Custom")

        if (is.null(plot_data_mat)) {
          if (is.null(data_labels)) {
            data_labels <- paste0("Doc ", seq_along(cluster_data))
          }

          scatter_df <- data.frame(
            x = seq_along(cluster_data),
            y = cluster_data,
            cluster = as.factor(cluster_data),
            label = data_labels
          )
          scatter_df$hover_text <- paste("Document:", scatter_df$label, "<br>Cluster:", scatter_df$cluster)

          ggplot2::ggplot(scatter_df, ggplot2::aes(x = .data$x, y = .data$y, color = .data$cluster, text = .data$hover_text)) +
            ggplot2::geom_point(size = 2, alpha = 0.7) +
            ggplot2::labs(title = plot_title, x = "Document Index", y = "Cluster", color = "Cluster") +
            ggplot2::theme_minimal(base_size = 11)
        } else {
          if (is.null(data_labels)) {
            data_labels <- paste0("Doc ", seq_len(nrow(plot_data_mat)))
          }

          scatter_df <- data.frame(
            x = plot_data_mat[, 1],
            y = if (ncol(plot_data_mat) > 1) plot_data_mat[, 2] else rep(0, nrow(plot_data_mat)),
            cluster = as.factor(cluster_data),
            label = data_labels
          )
          scatter_df$hover_text <- paste("Document:", scatter_df$label, "<br>Cluster:", scatter_df$cluster)

          ggplot2::ggplot(scatter_df, ggplot2::aes(x = .data$x, y = .data$y, color = .data$cluster, text = .data$hover_text)) +
            ggplot2::geom_point(size = 2, alpha = 0.7) +
            ggplot2::labs(title = plot_title, x = "Component 1", y = "Component 2", color = "Cluster") +
            ggplot2::theme_minimal(base_size = 11)
        }
      },
      stop("Unsupported plot type: ", plot_type)
    )

    return(plot_obj)

  }, error = function(e) {
    stop("Error creating semantic visualization: ", e$message)
  })
}


#' Plot Cross-Category Similarity Comparison
#'
#' @description
#' Creates a faceted ggplot heatmap for cross-category document similarity
#' comparison. Accepts either a pre-built long-format data frame or extracts
#' from a similarity matrix.
#'
#' @param similarity_data Either a similarity matrix (square numeric matrix) or
#'   a data frame in long format with columns for row labels, column labels,
#'   similarity values, and category.
#' @param docs_data Data frame with document metadata (required if similarity_data is a matrix)
#' @param row_var Column name for row document labels (default: "ld_doc_name")
#' @param col_var Column name for column document labels (default: "other_doc_name")
#' @param value_var Column name for similarity values (default: "cosine_similarity")
#' @param category_var Column name for category in long-format data or docs_data (default: "other_category")
#' @param row_category Category for row documents (used with matrix input)
#' @param col_categories Categories for column documents (used with matrix input)
#' @param row_display_var Column name for row display labels in tooltip (default: NULL, uses row_var)
#' @param col_display_var Column name for column display labels in tooltip (default: NULL, uses col_var)
#' @param method_name Similarity method name for legend (default: "Cosine")
#' @param title Plot title (default: NULL)
#' @param show_values Logical; show similarity values as text on tiles (default: TRUE)
#' @param row_label Label for y-axis (default: "Documents")
#' @param label_max_chars Maximum characters for axis labels before truncation (default: 25)
#' @param order_by_numeric Logical; order by numeric ID extracted from labels (default: TRUE)
#' @param height Plot height (default: 600)
#' @param width Plot width (default: NULL)
#'
#' @return A ggplot object
#'
#' @concept visualization
#' @export
#'
#' @examples
#' \donttest{
#' articles <- TextAnalysisR::SpecialEduTech[1:7, ]
#' term_matrix <- as.matrix(quanteda::dfm(quanteda::tokens(articles$abstract)))
#' normalized_matrix <- term_matrix / sqrt(rowSums(term_matrix ^ 2))
#' similarity_matrix <- normalized_matrix %*% t(normalized_matrix)
#' thesis_rows <- which(articles$reference_type == "thesis")[1:3]
#' article_cols <- which(articles$reference_type == "journal_article")[1:4]
#' similarity_data <- expand.grid(
#'   ld_doc_name    = paste("Thesis", seq_along(thesis_rows)),
#'   other_doc_name = paste("Article", seq_along(article_cols)),
#'   stringsAsFactors = FALSE
#' )
#' similarity_data$cosine_similarity <- as.vector(
#'   similarity_matrix[thesis_rows, article_cols]
#' )
#' similarity_data$other_category <- articles$reference_type[article_cols]
#' plot_cross_category_heatmap(
#'   similarity_data = similarity_data,
#'   row_var = "ld_doc_name",
#'   col_var = "other_doc_name",
#'   value_var = "cosine_similarity",
#'   category_var = "other_category",
#'   row_label = "Theses"
#' )
#' }
plot_cross_category_heatmap <- function(similarity_data,
                                         docs_data = NULL,
                                         row_var = "ld_doc_name",
                                         col_var = "other_doc_name",
                                         value_var = "cosine_similarity",
                                         category_var = "other_category",
                                         row_category = NULL,
                                         col_categories = NULL,
                                         row_display_var = NULL,
                                         col_display_var = NULL,
                                         method_name = "Cosine",
                                         title = NULL,
                                         show_values = TRUE,
                                         row_label = "Documents",
                                         label_max_chars = 25,
                                         order_by_numeric = TRUE,
                                         height = 600,
                                         width = NULL) {

  if (!requireNamespace("stringr", quietly = TRUE)) {
    stop("Package 'stringr' is required. Install with: install.packages('stringr')")
  }

  # Branch on input type: long-format data frame or matrix
  if (is.data.frame(similarity_data)) {
    plot_data <- similarity_data

    # Validate required columns
    required_cols <- c(row_var, col_var, value_var, category_var)
    missing_cols <- setdiff(required_cols, names(plot_data))
    if (length(missing_cols) > 0) {
      stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
    }

    # Rename to internal names
    plot_data <- plot_data %>%
      dplyr::rename(
        row_doc = !!rlang::sym(row_var),
        col_doc = !!rlang::sym(col_var),
        similarity = !!rlang::sym(value_var),
        col_category = !!rlang::sym(category_var)
      )

    # Display columns for tooltips
    if (!is.null(row_display_var) && row_display_var %in% names(similarity_data)) {
      plot_data$row_display <- similarity_data[[row_display_var]]
    } else {
      plot_data$row_display <- plot_data$row_doc
    }

    if (!is.null(col_display_var) && col_display_var %in% names(similarity_data)) {
      plot_data$col_display <- similarity_data[[col_display_var]]
    } else {
      plot_data$col_display <- plot_data$col_doc
    }

    # Truncated labels
    plot_data <- plot_data %>%
      dplyr::mutate(
        row_label_trunc = stringr::str_trunc(.data$row_doc, label_max_chars),
        col_label_trunc = stringr::str_trunc(.data$col_doc, label_max_chars)
      )

    # Order by numeric ID if requested
    if (order_by_numeric) {
      plot_data <- plot_data %>%
        dplyr::mutate(
          row_numeric_id = as.numeric(stringr::str_extract(.data$row_doc, "\\d+")),
          col_numeric_id = as.numeric(stringr::str_extract(.data$col_doc, "\\d+"))
        )

      row_order <- plot_data %>%
        dplyr::arrange(.data$row_numeric_id) %>%
        dplyr::pull(.data$row_label_trunc) %>%
        unique()

      col_order <- plot_data %>%
        dplyr::arrange(.data$col_numeric_id) %>%
        dplyr::pull(.data$col_label_trunc) %>%
        unique()
    } else {
      row_order <- unique(plot_data$row_label_trunc)
      col_order <- unique(plot_data$col_label_trunc)
    }

    # Build final plot data with factor levels and tooltips
    cat_levels <- unique(plot_data$col_category)

    plot_data <- plot_data %>%
      dplyr::mutate(
        row_label_trunc = factor(.data$row_label_trunc, levels = rev(row_order)),
        col_label_trunc = factor(.data$col_label_trunc, levels = col_order),
        col_category = factor(.data$col_category, levels = cat_levels),
        tooltip_text = paste0(
          row_label, ": ", dplyr::coalesce(as.character(.data$row_display), as.character(.data$row_doc)),
          "<br>", .data$col_category, ": ", dplyr::coalesce(as.character(.data$col_display), as.character(.data$col_doc)),
          "<br>", method_name, " Similarity: ", round(.data$similarity, 3)
        )
      )

  } else if (is.matrix(similarity_data)) {
    # Matrix input: extract cross-category data
    if (is.null(docs_data) || is.null(row_category) || is.null(col_categories)) {
      stop("For matrix input, docs_data, row_category, and col_categories are required")
    }

    if (!category_var %in% names(docs_data)) {
      stop("category_var '", category_var, "' not found in docs_data")
    }

    row_indices <- which(docs_data[[category_var]] == row_category)
    if (length(row_indices) == 0) {
      return(create_empty_plot_message(paste("No documents found for category:", row_category)))
    }

    row_docs <- docs_data[row_indices, ]
    row_short_labels <- row_docs$document_number %||% paste("Doc", row_indices)
    row_full_ids <- row_docs$document_id_display %||% row_short_labels

    plot_data_list <- list()

    for (col_cat in col_categories) {
      col_indices <- which(docs_data[[category_var]] == col_cat)
      if (length(col_indices) == 0) next

      col_docs <- docs_data[col_indices, ]
      col_short_labels <- col_docs$document_number %||% paste("Doc", col_indices)
      col_full_ids <- col_docs$document_id_display %||% col_short_labels

      sub_matrix <- similarity_data[row_indices, col_indices, drop = FALSE]

      for (i in seq_along(row_indices)) {
        for (j in seq_along(col_indices)) {
          plot_data_list[[length(plot_data_list) + 1]] <- data.frame(
            row_label_trunc = row_short_labels[i],
            col_label_trunc = col_short_labels[j],
            row_display = row_full_ids[i],
            col_display = col_full_ids[j],
            similarity = sub_matrix[i, j],
            col_category = col_cat,
            stringsAsFactors = FALSE
          )
        }
      }
    }

    if (length(plot_data_list) == 0) {
      return(create_empty_plot_message("No matching documents found for specified categories"))
    }

    plot_data <- do.call(rbind, plot_data_list)

    row_order <- unique(plot_data$row_label_trunc)
    col_order <- unique(plot_data$col_label_trunc)

    plot_data <- plot_data %>%
      dplyr::mutate(
        row_label_trunc = factor(.data$row_label_trunc, levels = rev(row_order)),
        col_label_trunc = factor(.data$col_label_trunc, levels = col_order),
        col_category = factor(.data$col_category, levels = col_categories),
        tooltip_text = paste0(
          as.character(.data$row_label_trunc), ": ", .data$row_display,
          "<br>", as.character(.data$col_label_trunc), ": ", .data$col_display,
          "<br>Category: ", row_category, " / ", .data$col_category,
          "<br>", method_name, " Similarity: ", round(.data$similarity, 3)
        )
      )

    if (is.null(row_label) || row_label == "Documents") {
      row_label <- paste(row_category, "Documents")
    }

  } else {
    stop("similarity_data must be a data frame or matrix")
  }

  # Build the plot
  p <- ggplot2::ggplot(
    plot_data,
    ggplot2::aes(x = .data$col_label_trunc, y = .data$row_label_trunc, fill = .data$similarity, text = .data$tooltip_text)
  ) +
    ggplot2::geom_tile(color = "white", linewidth = 0.1)

  if (show_values) {
    q75 <- stats::quantile(plot_data$similarity, 0.75, na.rm = TRUE)
    p <- p + ggplot2::geom_text(
      ggplot2::aes(
        label = round(.data$similarity, 2),
        color = ifelse(.data$similarity > q75, "black", "white")
      ),
      size = 3.5,
      fontface = "bold",
      show.legend = FALSE
    ) +
      ggplot2::scale_color_identity()
  }

  p <- p +
    ggplot2::scale_fill_viridis_c(name = paste0(method_name, "\nSimilarity")) +
    ggplot2::facet_wrap(~ col_category, scales = "free_x") +
    ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(
      strip.text.x = ggplot2::element_text(size = 11, color = "#3B3B3B"),
      axis.text.x = ggplot2::element_text(angle = 45, hjust = 1, size = 11),
      axis.text.y = ggplot2::element_text(size = 11),
      axis.title.x = ggplot2::element_blank(),
      legend.title = ggplot2::element_text(size = 11, color = "#3B3B3B"),
      legend.text = ggplot2::element_text(size = 11, color = "#3B3B3B"),
      plot.title = ggplot2::element_text(size = 13, hjust = 0.5)
    ) +
    ggplot2::labs(y = row_label, title = title)

  return(p)
}


#' Plot Document Similarity Heatmap
#'
#' @description
#' Creates an interactive heatmap visualization of document similarity matrices
#' with support for document metadata, feature-specific colorscales, and rich tooltips.
#' Supports both symmetric (all-vs-all) and cross-category comparison modes.
#'
#' @param similarity_matrix A square numeric matrix of similarity scores
#' @param docs_data Optional data frame with document metadata containing:
#'   \itemize{
#'     \item \code{document_number}: Document identifiers for axis labels
#'     \item \code{document_id_display}: Document IDs for hover text
#'     \item \code{category_display}: Category labels for hover text
#'   }
#' @param feature_type Feature space type: "words", "topics", "ngrams", or "embeddings"
#'   (determines colorscale and display name)
#' @param method_name Similarity method name for display (default: "Cosine")
#' @param title Plot title (default: NULL, auto-generated from feature_type)
#' @param category_filter Optional category filter label for title (default: NULL)
#' @param doc_id_var Name of document ID variable (affects label text, default: NULL)
#' @param colorscale Plotly colorscale override (default: NULL, uses feature_type default)
#' @param height Plot height in pixels (default: 600)
#' @param width Plot width in pixels (default: NULL for auto)
#' @param row_category Category for row documents in cross-category mode (default: NULL)
#' @param col_categories Character vector of categories for column documents (default: NULL)
#' @param category_var Name of category variable in docs_data (default: "category_display")
#' @param show_values Logical; show similarity values as text on tiles (default: FALSE)
#' @param facet Logical; facet by column categories (default: TRUE when col_categories specified)
#' @param row_label Label for row axis (default: NULL, uses row_category)
#' @param output_type Output type: "plotly" or "ggplot" (default: "plotly", auto-switches to "ggplot" for faceting)
#'
#' @return A ggplot2 heatmap object
#'
#' @concept visualization
#' @export
#'
#' @examples
#' \donttest{
#' articles <- TextAnalysisR::SpecialEduTech[1:5, ]
#' term_matrix <- as.matrix(quanteda::dfm(quanteda::tokens(articles$abstract)))
#' normalized_matrix <- term_matrix / sqrt(rowSums(term_matrix ^ 2))
#' similarity_matrix <- normalized_matrix %*% t(normalized_matrix)
#' plot_similarity_heatmap(similarity_matrix)
#'
#' document_metadata <- data.frame(
#'   document_number     = paste("Doc", 1:5),
#'   document_id_display = articles$title,
#'   category_display    = articles$reference_type
#' )
#' plot_similarity_heatmap(similarity_matrix, docs_data = document_metadata,
#'                         feature_type = "embeddings")
#'
#' plot_similarity_heatmap(
#'   similarity_matrix,
#'   docs_data      = document_metadata,
#'   row_category   = "thesis",
#'   col_categories = "journal_article",
#'   show_values    = TRUE,
#'   facet          = TRUE
#' )
#' }
plot_similarity_heatmap <- function(similarity_matrix,
                                     docs_data = NULL,
                                     feature_type = "words",
                                     method_name = "Cosine",
                                     title = NULL,
                                     category_filter = NULL,
                                     doc_id_var = NULL,
                                     colorscale = NULL,
                                     height = 600,
                                     width = NULL,
                                     row_category = NULL,
                                     col_categories = NULL,
                                     category_var = "category_display",
                                     show_values = FALSE,
                                     facet = NULL,
                                     row_label = NULL,
                                     output_type = "plotly") {

  if (is.null(similarity_matrix) || nrow(similarity_matrix) < 2) {
    return(
      ggplot2::ggplot() +
        ggplot2::annotate("text", x = 0.5, y = 0.5,
                          label = "Need at least 2 documents for similarity analysis",
                          size = 5, color = "#ef4444") +
        ggplot2::theme_void()
    )
  }

  if (!is.null(row_category) && !is.null(col_categories) && !is.null(docs_data)) {
    if (is.null(facet)) facet <- TRUE
    if (facet || output_type == "ggplot") {
      return(plot_cross_category_heatmap(
        similarity_data = similarity_matrix,
        docs_data = docs_data,
        row_category = row_category,
        col_categories = col_categories,
        category_var = category_var,
        method_name = method_name,
        title = title,
        show_values = show_values,
        row_label = row_label,
        height = height,
        width = width
      ))
    }
  }

  n_docs <- nrow(similarity_matrix)

  feature_config <- switch(feature_type,
    "words" = list(display_name = "Word Co-occurrence", viridis_option = "C"),
    "topics" = list(display_name = "Topic Distribution", viridis_option = "B"),
    "ngrams" = list(display_name = "N-gram Pattern", viridis_option = "D"),
    "embeddings" = list(display_name = "Semantic Embedding", viridis_option = "A"),
    list(display_name = feature_type, viridis_option = "H")
  )

  if (!is.null(docs_data) && nrow(docs_data) >= n_docs) {
    docs_data <- docs_data[1:n_docs, ]
    x_labels <- docs_data$document_number %||% paste("Doc", 1:n_docs)
    y_labels <- x_labels
  } else {
    x_labels <- paste("Doc", 1:n_docs)
    y_labels <- x_labels
  }

  if (is.null(title)) {
    title <- if (!is.null(category_filter) && category_filter != "all") {
      paste("Document", feature_config$display_name, "Similarity:", category_filter)
    } else {
      paste("Document", feature_config$display_name, "Similarity Heatmap")
    }
  }

  heat_df <- expand.grid(
    col_idx = seq_len(n_docs),
    row_idx = seq_len(n_docs)
  )
  heat_df$similarity <- as.vector(similarity_matrix)
  heat_df$x_label <- factor(x_labels[heat_df$col_idx], levels = x_labels)
  heat_df$y_label <- factor(y_labels[heat_df$row_idx], levels = y_labels)

  if (!is.null(docs_data) && nrow(docs_data) >= n_docs) {
    doc_ids <- docs_data$document_id_display %||% x_labels
    cats <- docs_data$category_display %||% rep("", n_docs)
    heat_df$tooltip_text <- paste0(
      y_labels[heat_df$row_idx], ": ", doc_ids[heat_df$row_idx],
      "\n", x_labels[heat_df$col_idx], ": ", doc_ids[heat_df$col_idx],
      "\nCategory: ", cats[heat_df$row_idx], " / ", cats[heat_df$col_idx],
      "\n", method_name, " Similarity: ", round(heat_df$similarity, 3)
    )
  } else {
    heat_df$tooltip_text <- paste0(
      y_labels[heat_df$row_idx], " vs ", x_labels[heat_df$col_idx],
      "\nSimilarity: ", round(heat_df$similarity, 3)
    )
  }

  p <- ggplot2::ggplot(heat_df, ggplot2::aes(x = .data$x_label, y = .data$y_label,
                                               fill = .data$similarity, text = .data$tooltip_text)) +
    ggplot2::geom_tile()

  if (show_values) {
    p <- p + ggplot2::geom_text(ggplot2::aes(label = round(.data$similarity, 2)), size = 3)
  }

  p +
    ggplot2::scale_fill_viridis_c(name = "Similarity\nScore", option = feature_config$viridis_option) +
    ggplot2::labs(title = title, x = "Documents", y = "Documents") +
    ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))
}


#' RAG Semantic Search
#'
#' @description
#' Simple in-memory RAG (Retrieval Augmented Generation) for question-answering
#' over document corpus with source attribution. Uses local (Ollama) or cloud
#' (OpenAI/Gemini) embeddings for semantic search and LLM for answer generation.
#'
#' @param query Character string, user question
#' @param documents Character vector, corpus to search
#' @param provider Character string, provider: "ollama" (local), "openai", or "gemini"
#' @param api_key Character string, API key for cloud providers (or from
#'   OPENAI_API_KEY/GEMINI_API_KEY env). Not required for Ollama.
#' @param embedding_model Character string, embedding model. Defaults:
#'   "nomic-embed-text" (ollama), "text-embedding-3-small" (openai),
#'   "gemini-embedding-001" (gemini)
#' @param chat_model Character string, chat model. Defaults: "llama3.2" (ollama),
#'   "gpt-4.1-mini" (openai), "gemini-2.5-flash-lite" (gemini)
#' @param top_k Integer, number of documents to retrieve (default: 5)
#'
#' @return List with:
#'   - success: Logical
#'   - answer: Generated answer
#'   - confidence: Confidence score (0-1)
#'   - sources: Vector of source document indices
#'   - retrieved_docs: Retrieved document chunks
#'   - scores: Similarity scores
#'
#' @details
#' Simple RAG workflow:
#' 1. Generate embeddings for documents and query
#' 2. Find top-k similar documents via cosine similarity
#' 3. Generate answer using LLM with retrieved context
#'
#' @concept ai
#' @seealso [get_best_embeddings()] for the retrieval step alone; [call_llm_api()] for the answer-generation step alone; [sanitize_llm_input()] for an input safety check before calling
#' @export
#'
#' @examples
#' if (interactive()) {
#' documents <- c(
#'   "Assistive technology helps students with disabilities access curriculum.",
#'   "Universal Design for Learning provides multiple means of engagement.",
#'   "Response to Intervention uses tiered support systems."
#' )
#'
#' # Using local Ollama (free, private)
#' result <- run_rag_search(
#'   query = "How does assistive technology support learning?",
#'   documents = documents,
#'   provider = "ollama"
#' )
#'
#' # Using OpenAI (requires API key)
#' result <- run_rag_search(
#'   query = "How does assistive technology support learning?",
#'   documents = documents,
#'   provider = "openai"
#' )
#'
#' if (result$success) {
#'   cat("Answer:", result$answer, "\n")
#'   cat("Sources:", paste(result$sources, collapse = ", "), "\n")
#' }
#' }
run_rag_search <- function(
  query,
  documents,
  provider = c("ollama", "openai", "gemini"),
  api_key = NULL,
  embedding_model = NULL,
  chat_model = NULL,
  top_k = 5
) {
  provider <- match.arg(provider)

  # API key for cloud providers (Ollama runs locally, no key needed)
  if (provider != "ollama") {
    if (is.null(api_key)) {
      api_key <- switch(provider,
        "openai" = Sys.getenv("OPENAI_API_KEY"),
        "gemini" = Sys.getenv("GEMINI_API_KEY")
      )
    }

    if (!nzchar(api_key)) {
      return(list(
        success = FALSE,
        error = .missing_api_key_message(provider, "package"),
        answer = "",
        confidence = 0.0,
        sources = c()
      ))
    }
  } else {
    # Check Ollama availability
    if (!check_ollama(verbose = FALSE)) {
      return(list(
        success = FALSE,
        error = "Ollama is not running. Please start Ollama and try again.",
        answer = "",
        confidence = 0.0,
        sources = c()
      ))
    }
  }

  if (length(documents) < 1) {
    return(list(
      success = FALSE,
      error = "No documents provided",
      answer = "",
      confidence = 0.0,
      sources = c()
    ))
  }

  # Default models per provider
  if (is.null(embedding_model)) {
    embedding_model <- switch(provider,
      "ollama" = "nomic-embed-text",
      "openai" = "text-embedding-3-small",
      "gemini" = "gemini-embedding-001"
    )
  }

  if (is.null(chat_model)) {
    chat_model <- switch(provider,
      "ollama" = "llama3.2",
      "openai" = "gpt-4.1-mini",
      "gemini" = "gemini-2.5-flash-lite"
    )
  }

  # Cap top_k at available documents
  top_k <- min(top_k, length(documents))

  # Step 1: Generate embeddings for all documents
  doc_embeddings <- tryCatch({
    get_api_embeddings(
      texts = documents,
      provider = provider,
      model = embedding_model,
      api_key = if (provider == "ollama") NULL else api_key
    )
  }, error = function(e) {
    return(list(error = e$message))
  })

  if (is.list(doc_embeddings) && !is.null(doc_embeddings$error)) {
    return(list(
      success = FALSE,
      error = paste("Failed to generate document embeddings:", doc_embeddings$error),
      answer = "",
      confidence = 0.0,
      sources = c()
    ))
  }

  # Step 2: Generate embedding for the query
  query_embedding <- tryCatch({
    get_api_embeddings(
      texts = query,
      provider = provider,
      model = embedding_model,
      api_key = if (provider == "ollama") NULL else api_key
    )
  }, error = function(e) {
    return(list(error = e$message))
  })

  if (is.list(query_embedding) && !is.null(query_embedding$error)) {
    return(list(
      success = FALSE,
      error = paste("Failed to generate query embedding:", query_embedding$error),
      answer = "",
      confidence = 0.0,
      sources = c()
    ))
  }

  # Step 3: Calculate cosine similarity between query and all documents
  query_vec <- as.numeric(query_embedding[1, ])
  doc_matrix <- as.matrix(doc_embeddings)

  similarities <- apply(doc_matrix, 1, function(doc_vec) {
    .cosine_sim(query_vec, doc_vec)
  })

  # Step 4: Get top-k most similar documents
  top_indices <- order(similarities, decreasing = TRUE)[1:top_k]
  top_scores <- similarities[top_indices]
  retrieved_docs <- documents[top_indices]

  query <- sanitize_llm_input(query)

  # Step 5: Generate answer using LLM with retrieved context
  context <- paste(
    sapply(seq_along(retrieved_docs), function(i) {
      sprintf("[Document %d] %s", top_indices[i], retrieved_docs[i])
    }),
    collapse = "\n\n"
  )

  system_prompt <- "You are a helpful research assistant. Answer the user's question based on the provided context documents. Be concise and accurate. If the context doesn't contain relevant information, say so."

  user_prompt <- sprintf(
    "Context Documents:\n%s\n\nQuestion: %s\n\nPlease provide a concise answer based on the context above.",
    context,
    query
  )

  answer <- tryCatch({
    call_llm_api(
      provider = provider,
      system_prompt = system_prompt,
      user_prompt = user_prompt,
      model = chat_model,
      temperature = 0.3,
      max_tokens = 500,
      api_key = api_key
    )
  }, error = function(e) {
    return(NULL)
  })

  if (is.null(answer)) {
    return(list(
      success = FALSE,
      error = "Failed to generate answer from LLM",
      answer = "",
      confidence = 0.0,
      sources = top_indices,
      retrieved_docs = retrieved_docs,
      scores = top_scores
    ))
  }

  # Confidence from top similarity scores
  avg_similarity <- mean(top_scores)
  confidence <- min(1.0, max(0.0, avg_similarity))

  return(list(
    success = TRUE,
    answer = answer,
    confidence = round(confidence, 3),
    sources = top_indices,
    retrieved_docs = retrieved_docs,
    scores = round(top_scores, 4),
    models = list(
      embedding = embedding_model,
      chat = chat_model
    ),
    provider = provider
  ))
}


# Network Analysis Functions

.network_dt <- function(df, group_label) {
  df <- df %>% dplyr::mutate(dplyr::across(dplyr::where(is.numeric), ~ round(., 3)))
  tbl <- DT::datatable(df, rownames = FALSE, extensions = "Buttons",
                       options = list(scrollX = TRUE, width = "80%", dom = "Bfrtip",
                                      buttons = c("copy", "csv", "excel", "pdf", "print"))) %>%
    DT::formatStyle(columns = colnames(df), `font-size` = "16px")
  htmltools::tagList(
    htmltools::tags$div(style = "margin-bottom: 20px;",
                        htmltools::tags$p(group_label,
                                          style = "font-weight: bold; text-align: center; font-size: 16px;")),
    tbl
  )
}

.network_summary_html <- function(graph, group_label) {
  df <- data.frame(
    Metric = c("Nodes", "Edges", "Density", "Diameter",
               "Global Clustering", "Mean Local Clustering",
               "Modularity", "Assortativity", "Mean Geodesic Distance"),
    Value = round(c(
      igraph::vcount(graph), igraph::ecount(graph), igraph::edge_density(graph),
      igraph::diameter(graph), igraph::transitivity(graph, type = "global"),
      mean(igraph::transitivity(graph, type = "local"), na.rm = TRUE),
      igraph::modularity(graph, membership = igraph::V(graph)$community),
      igraph::assortativity_degree(graph),
      mean(igraph::distances(graph)[igraph::distances(graph) != Inf], na.rm = TRUE)
    ), 3)
  )
  .network_dt(df, group_label)
}

.network_centrality <- function(graph, community_method = "leiden") {
  igraph::V(graph)$degree      <- igraph::degree(graph)
  igraph::V(graph)$betweenness <- igraph::betweenness(graph)
  igraph::V(graph)$closeness   <- igraph::closeness(graph)
  igraph::V(graph)$eigenvector <- igraph::eigen_centrality(graph)$vector
  membership <- switch(community_method,
    "louvain" = igraph::cluster_louvain(graph)$membership,
    igraph::cluster_leiden(graph, objective_function = "modularity")$membership
  )
  igraph::V(graph)$community <- membership
  graph
}

.node_metric_vec <- function(graph, by) {
  switch(by,
    "betweenness" = igraph::V(graph)$betweenness,
    "closeness"   = igraph::V(graph)$closeness,
    "eigenvector" = igraph::V(graph)$eigenvector,
    "frequency"   = igraph::V(graph)$frequency,
    igraph::V(graph)$degree)
}

.community_palette <- function(n) {
  if (n <= 8) RColorBrewer::brewer.pal(max(3, n), "Set2")[seq_len(n)]
  else grDevices::colorRampPalette(RColorBrewer::brewer.pal(8, "Set2"))(n)
}

.safe_rescale <- function(vals, to, fallback) {
  if (length(unique(vals)) <= 1) rep(fallback, length(vals))
  else scales::rescale(vals, to = to)
}

.visnet_widget <- function(nodes, edges, group_level, width, height, node_label_size,
                           physics_gravity, physics_spring_length, physics_avoid_overlap,
                           seed, showlegend, node_color_by, community_colors) {
  main_cfg <- if (is.null(group_level)) NULL else list(
    text = as.character(group_level),
    style = "font-family:Roboto, sans-serif; font-size:16px; color:#0c1f4a; font-weight:bold;"
  )
  widget <- visNetwork::visNetwork(nodes, edges, width = width, height = height, main = main_cfg) %>%
    visNetwork::visNodes(font = list(color = "black", size = node_label_size, vadjust = 0,
                                     strokeWidth = 3, strokeColor = "#ffffff")) %>%
    visNetwork::visOptions(
      highlightNearest = list(enabled = TRUE, degree = 1, hover = TRUE, algorithm = "hierarchical"),
      nodesIdSelection = TRUE,
      selectedBy = list(variable = "group", multiple = FALSE, style = "width: 150px; height: 26px;")
    ) %>%
    visNetwork::visPhysics(
      solver = "barnesHut",
      barnesHut = list(gravitationalConstant = physics_gravity, centralGravity = 0.4,
                       springLength = physics_spring_length, springConstant = 0.05,
                       avoidOverlap = physics_avoid_overlap),
      stabilization = list(enabled = TRUE, iterations = 150, updateInterval = 50)
    ) %>%
    visNetwork::visInteraction(hover = TRUE, tooltipDelay = 0, tooltipStay = 1000,
                               zoomView = TRUE, dragView = TRUE,
                               navigationButtons = TRUE, keyboard = TRUE) %>%
    visNetwork::visLayout(randomSeed = seed %||% 2025)

  n_communities <- length(community_colors)
  if (isTRUE(showlegend) && node_color_by == "community" && n_communities > 0) {
    legend_df <- data.frame(
      label = paste0("Community ", seq_len(n_communities),
                     " (", tabulate(nodes$group, nbins = n_communities), ")"),
      color = community_colors[as.character(seq_len(n_communities))],
      shape = "dot",
      stringsAsFactors = FALSE
    )
    widget <- widget %>% visNetwork::visLegend(addNodes = legend_df, useGroups = FALSE,
                                               position = "right", width = 0.2, zoom = FALSE)
  }

  scale_js <- sprintf(
    "function(el, x) {
      var network = this;
      var baseSize = %d;
      var refWidth = 1200;
      var minSize = 16;
      var maxSize = 36;
      function compute() {
        var w = el && el.clientWidth ? el.clientWidth : refWidth;
        var s = Math.round(baseSize * (w / refWidth));
        if (s < minSize) s = minSize;
        if (s > maxSize) s = maxSize;
        return s;
      }
      function apply() {
        var s = compute();
        try {
          network.setOptions({ nodes: { font: { size: s } } });
        } catch (e) {}
      }
      setTimeout(apply, 80);
      if (typeof ResizeObserver !== 'undefined') {
        var ro = new ResizeObserver(function() { apply(); });
        ro.observe(el);
      } else {
        window.addEventListener('resize', apply);
      }
    }",
    as.integer(node_label_size)
  )

  htmlwidgets::onRender(widget, scale_js)
}

.facet_widgets_html <- function(per_level, nrows) {
  cols_per_row <- max(ceiling(length(per_level) / max(nrows, 1)), 1)
  panel_pct <- floor(100 / cols_per_row)
  widgets <- lapply(names(per_level), function(nm) {
    htmltools::tags$div(
      style = sprintf("flex: 1 1 %d%%; min-width: 350px; padding: 10px; box-sizing: border-box;", panel_pct),
      per_level[[nm]]$plot
    )
  })
  htmltools::tags$div(
    style = "display: flex; flex-wrap: wrap; gap: 20px; justify-content: center;",
    widgets
  )
}

#' @title Analyze and Visualize Word Co-occurrence Networks
#'
#' @description
#' This function creates a word co-occurrence network based on a document-feature matrix (dfm).
#'
#' @param dfm_object A quanteda document-feature matrix (dfm).
#' @param doc_var A document-level metadata variable (default: NULL).
#' @param co_occur_n Minimum co-occurrence count (default: 50).
#' @param top_node_n Number of top nodes to display (default: 30).
#' @param nrows Number of rows to display in the table (default: 1).
#' @param height The height of the resulting Plotly plot, in pixels (default: 800).
#' @param width The width of the resulting Plotly plot, in pixels (default: 900).
#' @param node_label_size Maximum font size for node labels in pixels (default: 22).
#' @param community_method Community detection method: "leiden" (default) or "louvain".
#' @param node_size_by Node sizing method: "degree", "betweenness", "closeness", "eigenvector", or "fixed" (default: "degree").
#' @param node_color_by Node coloring method: "community" or "centrality" (default: "community").
#' @param category_params Optional named list of category-specific parameters. Each element should be a list with `co_occur_n` and `top_node_n` values for that category (default: NULL).
#' @param pattern Optional regex (case-insensitive). If provided, keeps only edges whose `item1` or `item2` matches (default: NULL).
#' @param seed Integer RNG seed for reproducible layout (default: 2025).
#' @param physics_gravity barnesHut `gravitationalConstant`. More negative = nodes spread further apart (default: -1500).
#' @param physics_spring_length barnesHut spring length. Higher = longer edges (default: 100).
#' @param physics_avoid_overlap barnesHut overlap avoidance, 0 to 1. Higher = more node separation (default: 0.3).
#' @param showlegend Whether to display the community legend (default: TRUE).
#'
#' @return A list containing the visNetwork widget (or flex-layout HTML for faceted), a DT table, and a summary.
#'
#' @importFrom igraph graph_from_data_frame V vcount ecount degree betweenness closeness eigen_centrality cluster_leiden cluster_louvain edge_density diameter transitivity modularity assortativity_degree distances
#' @importFrom dplyr count filter mutate group_by summarise ungroup left_join arrange desc group_map pull
#' @importFrom tidytext tidy
#' @importFrom widyr pairwise_count
#' @importFrom scales rescale alpha col_numeric
#' @importFrom stats quantile setNames
#' @importFrom DT datatable formatStyle
#' @importFrom rlang sym
#' @importFrom grDevices colorRampPalette
#' @importFrom htmltools tagList tags browsable
#' @importFrom RColorBrewer brewer.pal
#'
#' @concept semantic
#' @seealso [word_correlation_network()] for correlation-based edges instead of co-occurrence counts; [plot_cluster_terms()] for bar-style cluster terms
#' @export
#'
#' @examples
#' if (interactive()) {
#'   df <- TextAnalysisR::SpecialEduTech
#'
#'   united_tbl <- TextAnalysisR::unite_cols(df, listed_vars = c("title", "abstract"))
#'
#'   tokens <- TextAnalysisR::prep_texts(united_tbl, text_field = "united_texts")
#'
#'   dfm_object <- quanteda::dfm(tokens)
#'
#'   word_co_occurrence_network_results <- TextAnalysisR::word_co_occurrence_network(
#'                                         dfm_object,
#'                                         doc_var = NULL,
#'                                         co_occur_n = 50,
#'                                         top_node_n = 30,
#'                                         nrows = 1,
#'                                         height = 800,
#'                                         width = 900,
#'                                         community_method = "leiden")
#'   print(word_co_occurrence_network_results$plot)
#'   print(word_co_occurrence_network_results$table)
#'   print(word_co_occurrence_network_results$summary)
#' }
word_co_occurrence_network <- function(dfm_object,
                                       doc_var = NULL,
                                       co_occur_n = 50,
                                       top_node_n = 30,
                                       nrows = 1,
                                       height = 800,
                                       width = 900,
                                       node_label_size = 22,
                                       community_method = "leiden",
                                       node_size_by = "degree",
                                       node_color_by = "community",
                                       category_params = NULL,
                                       pattern = NULL,
                                       seed = 2025,
                                       physics_gravity = -1500,
                                       physics_spring_length = 100,
                                       physics_avoid_overlap = 0.3,
                                       showlegend = TRUE) {

  if (!requireNamespace("visNetwork", quietly = TRUE) ||
      !requireNamespace("htmltools", quietly = TRUE) ||
      !requireNamespace("RColorBrewer", quietly = TRUE)) {
    stop(
      "Packages 'visNetwork', 'htmltools', and 'RColorBrewer' are required. ",
      "Install with install.packages(c('visNetwork', 'htmltools', 'RColorBrewer'))."
    )
  }

  dfm_td <- tidytext::tidy(dfm_object)
  docvars_df <- quanteda::docvars(dfm_object)
  docvars_df$document <- quanteda::docnames(dfm_object)
  dfm_td <- dplyr::left_join(dfm_td, docvars_df, by = "document")

  if (!is.null(doc_var) && doc_var != "" && !doc_var %in% colnames(dfm_td)) {
    message("Document-level metadata variable '", doc_var, "' was not selected or not found.")
    doc_var <- NULL
  }

  if (!is.null(doc_var) && doc_var %in% colnames(dfm_td)) {
    docvar_levels <- unique(dfm_td[[doc_var]])
    message(paste("doc_var has", length(docvar_levels), "levels:", paste(docvar_levels, collapse = ", ")))
  } else {
    docvar_levels <- NULL
  }

  build_network_plot <- function(data, group_level = NULL, local_co_occur_n = NULL,
                                 local_top_node_n = NULL, panel_width = NULL, panel_height = NULL) {
    if (!is.null(seed)) set.seed(seed)
    eff_co_occur_n <- local_co_occur_n %||% co_occur_n
    eff_top_node_n <- local_top_node_n %||% top_node_n

    data <- data %>%
      dplyr::group_by(term) %>%
      dplyr::filter(dplyr::n_distinct(document) >= eff_co_occur_n) %>%
      dplyr::ungroup()

    if (nrow(data) == 0) {
      message("No terms meet the co-occurrence threshold.")
      return(NULL)
    }

    term_co_occur <- data %>%
      widyr::pairwise_count(term, document, sort = TRUE) %>%
      dplyr::filter(n >= eff_co_occur_n)

    if (!is.null(pattern) && nzchar(pattern)) {
      term_co_occur <- term_co_occur %>%
        dplyr::filter(grepl(pattern, item1, ignore.case = TRUE) |
                      grepl(pattern, item2, ignore.case = TRUE))
    }

    if (nrow(term_co_occur) == 0) {
      message("No co-occurrence relationships meet the threshold.")
      return(NULL)
    }

    graph <- igraph::graph_from_data_frame(term_co_occur, directed = FALSE)
    if (igraph::vcount(graph) == 0) return(NULL)

    graph <- .network_centrality(graph, community_method)

    freq_df <- data %>%
      dplyr::group_by(term) %>%
      dplyr::summarise(frequency = sum(count, na.rm = TRUE), .groups = "drop")
    igraph::V(graph)$frequency <- freq_df$frequency[match(igraph::V(graph)$name, freq_df$term)]
    igraph::V(graph)$frequency[is.na(igraph::V(graph)$frequency)] <- 1

    sort_metric <- .node_metric_vec(graph, node_size_by)
    names(sort_metric) <- igraph::V(graph)$name
    top_n <- min(eff_top_node_n, length(sort_metric))
    top_nodes <- names(sort(sort_metric, decreasing = TRUE))[seq_len(top_n)]

    size_vals <- if (node_size_by == "fixed") rep(10, igraph::vcount(graph)) else sort_metric
    cap_vals <- pmin(size_vals, stats::quantile(size_vals, 0.95, na.rm = TRUE))
    node_values <- .safe_rescale(sqrt(cap_vals), to = c(10, 40), fallback = 20)

    communities <- igraph::V(graph)$community
    unique_communities <- sort(unique(communities))
    n_communities <- length(unique_communities)
    palette <- .community_palette(n_communities)
    community_map <- stats::setNames(seq_along(unique_communities), as.character(unique_communities))
    community_colors <- stats::setNames(palette, as.character(seq_len(n_communities)))

    freq_vec <- igraph::V(graph)$frequency
    node_colors <- if (node_color_by == "frequency") {
      scales::col_numeric("viridis", domain = range(freq_vec, na.rm = TRUE))(freq_vec)
    } else {
      community_colors[as.character(community_map[as.character(communities)])]
    }

    node_names_esc <- htmltools::htmlEscape(igraph::V(graph)$name)
    nodes <- data.frame(
      id = igraph::V(graph)$name,
      label = ifelse(igraph::V(graph)$name %in% top_nodes, igraph::V(graph)$name, ""),
      group = community_map[as.character(communities)],
      value = node_values,
      color = node_colors,
      title = paste0(
        "<b style='color:black;'>", node_names_esc, "</b><br>",
        "<span style='color:black;'>Degree: ", igraph::V(graph)$degree, "<br>",
        "Betweenness: ", round(igraph::V(graph)$betweenness, 2), "<br>",
        "Closeness: ", round(igraph::V(graph)$closeness, 3), "<br>",
        "Eigenvector: ", round(igraph::V(graph)$eigenvector, 3), "<br>",
        "Frequency: ", freq_vec, "<br>",
        "Community: ", communities, "</span>"
      ),
      stringsAsFactors = FALSE
    )

    edges <- igraph::as_data_frame(graph, what = "edges")
    edges$weight <- term_co_occur$n[match(paste(edges$from, edges$to),
                                          paste(term_co_occur$item1, term_co_occur$item2))]
    edges$width <- .safe_rescale(edges$weight, to = c(1, 8), fallback = 2)
    edge_alphas <- .safe_rescale(edges$weight, to = c(0.3, 1), fallback = 0.7)
    edges$color <- scales::alpha("#5C5CFF", edge_alphas)
    edges$title <- paste0("<span style='color:black;'>Co-occurrences: ", edges$weight,
                          "<br>From: ", htmltools::htmlEscape(edges$from),
                          "<br>To: ", htmltools::htmlEscape(edges$to), "</span>")

    widget <- .visnet_widget(nodes, edges, group_level,
                             width = panel_width %||% width,
                             height = panel_height %||% height,
                             node_label_size = node_label_size,
                             physics_gravity = physics_gravity,
                             physics_spring_length = physics_spring_length,
                             physics_avoid_overlap = physics_avoid_overlap,
                             seed = seed, showlegend = showlegend,
                             node_color_by = node_color_by,
                             community_colors = community_colors)

    layout_df <- data.frame(
      term = igraph::V(graph)$name,
      frequency = freq_vec,
      degree = igraph::V(graph)$degree,
      betweenness = igraph::V(graph)$betweenness,
      closeness = igraph::V(graph)$closeness,
      eigenvector = igraph::V(graph)$eigenvector,
      community = communities,
      stringsAsFactors = FALSE
    )

    list(plot = widget, graph = graph, layout_df = layout_df, top_nodes = top_nodes)
  }

  if (!is.null(doc_var) && length(docvar_levels) > 1) {
    width_num <- if (is.numeric(width)) width else 1200
    cols_per_row <- max(ceiling(length(docvar_levels) / max(nrows, 1)), 1)
    panel_w <- floor(width_num / cols_per_row)
    panel_h <- floor(height / max(nrows, 1))

    per_level <- lapply(docvar_levels, function(level) {
      group_data <- dplyr::filter(dfm_td, !!rlang::sym(doc_var) == level)
      cat_p <- category_params[[level]] %||% list()
      build_network_plot(group_data, level, cat_p$co_occur_n, cat_p$top_node_n,
                         panel_width = panel_w, panel_height = panel_h)
    })
    names(per_level) <- as.character(docvar_levels)
    per_level <- Filter(Negate(is.null), per_level)

    tables <- htmltools::tagList(lapply(names(per_level), function(nm) {
      .network_dt(per_level[[nm]]$layout_df, nm)
    })) %>% htmltools::browsable()

    summaries <- htmltools::tagList(lapply(names(per_level), function(nm) {
      .network_summary_html(per_level[[nm]]$graph, paste("Network Summary:", nm))
    })) %>% htmltools::browsable()

    list(plot = .facet_widgets_html(per_level, nrows), table = tables, summary = summaries)
  } else {
    net <- build_network_plot(dfm_td, panel_width = "100%")
    if (is.null(net)) return(NULL)
    table_label <- if (is.null(doc_var)) "Network Centrality Table"
                   else paste("Network Centrality Table for", doc_var)
    summary_label <- if (is.null(doc_var)) "Network Summary"
                     else paste("Network Summary for", doc_var)
    list(
      plot = net$plot,
      top_nodes = net$top_nodes,
      table = .network_dt(net$layout_df, table_label) %>% htmltools::browsable(),
      summary = .network_summary_html(net$graph, summary_label) %>% htmltools::browsable()
    )
  }
}


#' @title Analyze and Visualize Word Correlation Networks
#'
#' @description
#' This function creates a word correlation network based on a document-feature matrix (dfm).
#'
#' @param dfm_object A quanteda document-feature matrix (dfm).
#' @param doc_var A document-level metadata variable (default: NULL).
#' @param common_term_n Minimum number of common terms for filtering terms (default: 130).
#' @param corr_n Minimum correlation value for filtering terms (default: 0.4).
#' @param top_node_n Number of top nodes to display (default: 40).
#' @param nrows Number of rows to display in the table (default: 1).
#' @param height The height of the resulting Plotly plot, in pixels (default: 1000).
#' @param width The width of the resulting Plotly plot, in pixels (default: 900).
#' @param node_label_size Maximum font size for node labels in pixels (default: 22).
#' @param community_method Community detection method: "leiden" (default) or "louvain".
#' @param node_size_by Node sizing method: "degree", "betweenness", "closeness", "eigenvector", or "fixed" (default: "degree").
#' @param node_color_by Node coloring method: "community" or "centrality" (default: "community").
#' @param category_params Optional named list of category-specific parameters. Each element should be a list with `common_term_n`, `corr_n`, and `top_node_n` values for that category (default: NULL).
#' @param pattern Optional regex (case-insensitive). If provided, keeps only edges whose `item1` or `item2` matches (default: NULL).
#' @param seed Integer RNG seed for reproducible layout (default: 2025).
#' @param physics_gravity barnesHut `gravitationalConstant`. More negative = nodes spread further apart (default: -1500).
#' @param physics_spring_length barnesHut spring length. Higher = longer edges (default: 100).
#' @param physics_avoid_overlap barnesHut overlap avoidance, 0 to 1. Higher = more node separation (default: 0.3).
#' @param showlegend Whether to display the community legend (default: TRUE).
#'
#' @return A list containing the visNetwork widget (or flex-layout HTML for faceted), a DT table, and a summary.
#'
#' @importFrom igraph graph_from_data_frame V vcount ecount degree betweenness closeness eigen_centrality cluster_leiden cluster_louvain edge_density diameter transitivity modularity assortativity_degree distances
#' @importFrom dplyr count filter mutate group_by summarise ungroup left_join arrange desc group_map pull
#' @importFrom tidytext tidy
#' @importFrom widyr pairwise_cor
#' @importFrom scales rescale alpha col_numeric
#' @importFrom stats quantile setNames
#' @importFrom DT datatable formatStyle
#' @importFrom rlang sym
#' @importFrom grDevices colorRampPalette
#' @importFrom htmltools tagList tags browsable
#' @importFrom RColorBrewer brewer.pal
#'
#' @concept semantic
#' @export
#'
#' @examples
#' if (interactive()) {
#'   df <- TextAnalysisR::SpecialEduTech
#'
#'   united_tbl <- TextAnalysisR::unite_cols(df, listed_vars = c("title", "abstract"))
#'
#'   tokens <- TextAnalysisR::prep_texts(united_tbl, text_field = "united_texts")
#'
#'   dfm_object <- quanteda::dfm(tokens)
#'
#'   word_correlation_network_results <- TextAnalysisR::word_correlation_network(
#'                                       dfm_object,
#'                                       doc_var = NULL,
#'                                       common_term_n = 30,
#'                                       corr_n = 0.4,
#'                                       top_node_n = 40,
#'                                       nrows = 1,
#'                                       height = 1000,
#'                                       width = 900,
#'                                       community_method = "leiden")
#'   print(word_correlation_network_results$plot)
#'   print(word_correlation_network_results$table)
#'   print(word_correlation_network_results$summary)
#' }
word_correlation_network <- function(dfm_object,
                                     doc_var = NULL,
                                     common_term_n = 130,
                                     corr_n = 0.4,
                                     top_node_n = 40,
                                     nrows = 1,
                                     height = 1000,
                                     width = 900,
                                     node_label_size = 22,
                                     community_method = "leiden",
                                     node_size_by = "degree",
                                     node_color_by = "community",
                                     category_params = NULL,
                                     pattern = NULL,
                                     seed = 2025,
                                     physics_gravity = -1500,
                                     physics_spring_length = 100,
                                     physics_avoid_overlap = 0.3,
                                     showlegend = TRUE) {

  if (!requireNamespace("visNetwork", quietly = TRUE) ||
      !requireNamespace("htmltools", quietly = TRUE) ||
      !requireNamespace("RColorBrewer", quietly = TRUE)) {
    stop(
      "Packages 'visNetwork', 'htmltools', and 'RColorBrewer' are required. ",
      "Install with install.packages(c('visNetwork', 'htmltools', 'RColorBrewer'))."
    )
  }

  dfm_td <- tidytext::tidy(dfm_object)
  docvars_df <- quanteda::docvars(dfm_object)
  docvars_df$document <- quanteda::docnames(dfm_object)
  dfm_td <- dplyr::left_join(dfm_td, docvars_df, by = "document")

  if (!is.null(doc_var) && doc_var != "" && !doc_var %in% colnames(dfm_td)) {
    message("Document-level metadata variable '", doc_var, "' was not selected or not found.")
    doc_var <- NULL
  }

  if (!is.null(doc_var) && doc_var %in% colnames(dfm_td)) {
    docvar_levels <- unique(dfm_td[[doc_var]])
    message(paste("doc_var has", length(docvar_levels), "levels:", paste(docvar_levels, collapse = ", ")))
  } else {
    docvar_levels <- NULL
  }

  build_network_plot <- function(data, group_level = NULL, local_common_term_n = NULL,
                                 local_corr_n = NULL, local_top_node_n = NULL,
                                 panel_width = NULL, panel_height = NULL) {
    if (!is.null(seed)) set.seed(seed)
    eff_common_term_n <- local_common_term_n %||% common_term_n
    eff_corr_n <- local_corr_n %||% corr_n
    eff_top_node_n <- local_top_node_n %||% top_node_n

    term_cor <- data %>%
      dplyr::group_by(term) %>%
      dplyr::filter(dplyr::n() >= eff_common_term_n) %>%
      widyr::pairwise_cor(term, document, sort = TRUE) %>%
      dplyr::ungroup() %>%
      dplyr::filter(correlation > eff_corr_n)

    if (!is.null(pattern) && nzchar(pattern)) {
      term_cor <- term_cor %>%
        dplyr::filter(grepl(pattern, item1, ignore.case = TRUE) |
                      grepl(pattern, item2, ignore.case = TRUE))
    }

    if (nrow(term_cor) == 0) {
      message("No correlation relationships meet the threshold.")
      return(NULL)
    }

    graph <- igraph::graph_from_data_frame(term_cor, directed = FALSE)
    if (igraph::vcount(graph) == 0) return(NULL)

    graph <- .network_centrality(graph, community_method)

    freq_df <- data %>%
      dplyr::group_by(term) %>%
      dplyr::summarise(frequency = sum(count, na.rm = TRUE), .groups = "drop")
    igraph::V(graph)$frequency <- freq_df$frequency[match(igraph::V(graph)$name, freq_df$term)]
    igraph::V(graph)$frequency[is.na(igraph::V(graph)$frequency)] <- 1

    sort_metric <- .node_metric_vec(graph, node_size_by)
    names(sort_metric) <- igraph::V(graph)$name
    top_n <- min(eff_top_node_n, length(sort_metric))
    top_nodes <- names(sort(sort_metric, decreasing = TRUE))[seq_len(top_n)]

    size_vals <- if (node_size_by == "fixed") rep(10, igraph::vcount(graph)) else sort_metric
    cap_vals <- pmin(size_vals, stats::quantile(size_vals, 0.95, na.rm = TRUE))
    node_values <- .safe_rescale(sqrt(cap_vals), to = c(10, 40), fallback = 20)

    communities <- igraph::V(graph)$community
    unique_communities <- sort(unique(communities))
    n_communities <- length(unique_communities)
    palette <- .community_palette(n_communities)
    community_map <- stats::setNames(seq_along(unique_communities), as.character(unique_communities))
    community_colors <- stats::setNames(palette, as.character(seq_len(n_communities)))

    freq_vec <- igraph::V(graph)$frequency
    node_colors <- if (node_color_by == "frequency") {
      scales::col_numeric("viridis", domain = range(freq_vec, na.rm = TRUE))(freq_vec)
    } else {
      community_colors[as.character(community_map[as.character(communities)])]
    }

    node_names_esc <- htmltools::htmlEscape(igraph::V(graph)$name)
    nodes <- data.frame(
      id = igraph::V(graph)$name,
      label = ifelse(igraph::V(graph)$name %in% top_nodes, igraph::V(graph)$name, ""),
      group = community_map[as.character(communities)],
      value = node_values,
      color = node_colors,
      title = paste0(
        "<b style='color:black;'>", node_names_esc, "</b><br>",
        "<span style='color:black;'>Degree: ", igraph::V(graph)$degree, "<br>",
        "Betweenness: ", round(igraph::V(graph)$betweenness, 2), "<br>",
        "Closeness: ", round(igraph::V(graph)$closeness, 3), "<br>",
        "Eigenvector: ", round(igraph::V(graph)$eigenvector, 3), "<br>",
        "Frequency: ", freq_vec, "<br>",
        "Community: ", communities, "</span>"
      ),
      stringsAsFactors = FALSE
    )

    edges <- igraph::as_data_frame(graph, what = "edges")
    edges$correlation <- term_cor$correlation[match(paste(edges$from, edges$to),
                                                    paste(term_cor$item1, term_cor$item2))]
    edges$width <- .safe_rescale(edges$correlation, to = c(1, 8), fallback = 2)
    edge_alphas <- .safe_rescale(abs(edges$correlation), to = c(0.3, 1), fallback = 0.7)
    edges$color <- scales::alpha("#5C5CFF", edge_alphas)
    edges$title <- paste0("<span style='color:black;'>Correlation: ", round(edges$correlation, 3),
                          "<br>From: ", htmltools::htmlEscape(edges$from),
                          "<br>To: ", htmltools::htmlEscape(edges$to), "</span>")

    widget <- .visnet_widget(nodes, edges, group_level,
                             width = panel_width %||% width,
                             height = panel_height %||% height,
                             node_label_size = node_label_size,
                             physics_gravity = physics_gravity,
                             physics_spring_length = physics_spring_length,
                             physics_avoid_overlap = physics_avoid_overlap,
                             seed = seed, showlegend = showlegend,
                             node_color_by = node_color_by,
                             community_colors = community_colors)

    layout_df <- data.frame(
      term = igraph::V(graph)$name,
      frequency = freq_vec,
      degree = igraph::V(graph)$degree,
      betweenness = igraph::V(graph)$betweenness,
      closeness = igraph::V(graph)$closeness,
      eigenvector = igraph::V(graph)$eigenvector,
      community = communities,
      stringsAsFactors = FALSE
    )

    list(plot = widget, graph = graph, layout_df = layout_df, top_nodes = top_nodes)
  }

  if (!is.null(doc_var) && length(docvar_levels) > 1) {
    width_num <- if (is.numeric(width)) width else 1200
    cols_per_row <- max(ceiling(length(docvar_levels) / max(nrows, 1)), 1)
    panel_w <- floor(width_num / cols_per_row)
    panel_h <- floor(height / max(nrows, 1))

    per_level <- lapply(docvar_levels, function(level) {
      group_data <- dplyr::filter(dfm_td, !!rlang::sym(doc_var) == level)
      cat_p <- category_params[[level]] %||% list()
      build_network_plot(group_data, level, cat_p$common_term_n, cat_p$corr_n, cat_p$top_node_n,
                         panel_width = panel_w, panel_height = panel_h)
    })
    names(per_level) <- as.character(docvar_levels)
    per_level <- Filter(Negate(is.null), per_level)

    tables <- htmltools::tagList(lapply(names(per_level), function(nm) {
      .network_dt(per_level[[nm]]$layout_df, nm)
    })) %>% htmltools::browsable()

    summaries <- htmltools::tagList(lapply(names(per_level), function(nm) {
      .network_summary_html(per_level[[nm]]$graph, paste("Network Summary:", nm))
    })) %>% htmltools::browsable()

    list(plot = .facet_widgets_html(per_level, nrows), table = tables, summary = summaries)
  } else {
    net <- build_network_plot(dfm_td, panel_width = "100%")
    if (is.null(net)) return(NULL)
    table_label <- if (is.null(doc_var)) "Network Centrality Table"
                   else paste("Network Centrality Table for", doc_var)
    summary_label <- if (is.null(doc_var)) "Network Summary"
                     else paste("Network Summary for", doc_var)
    list(
      plot = net$plot,
      top_nodes = net$top_nodes,
      table = .network_dt(net$layout_df, table_label) %>% htmltools::browsable(),
      summary = .network_summary_html(net$graph, summary_label) %>% htmltools::browsable()
    )
  }
}
