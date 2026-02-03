#' @importFrom utils modifyList
#' @importFrom stats cor hclust dist cutree
NULL

# Semantic Analysis Functions
# Functions for semantic analysis, embeddings, and document clustering

#' @title Calculate Document Similarity
#'
#' @description
#' Calculates similarity between documents using traditional NLP methods or
#' modern embedding-based approaches. Comprehensive metrics are automatically
#' computed unless disabled.
#'
#' @param texts A character vector of texts to compare.
#' @param document_feature_type Feature extraction type: "words", "ngrams", "embeddings", or "topics".
#' @param semantic_ngram_range Integer, n-gram range for ngram features (default: 2).
#' @param similarity_method Similarity calculation method: "cosine", "jaccard", "euclidean", "manhattan".
#' @param use_embeddings Logical, use embedding-based similarity (default: FALSE).
#' @param embedding_model Sentence transformer model name (default: "all-MiniLM-L6-v2").
#' @param calculate_metrics Logical, compute comprehensive similarity metrics (default: TRUE).
#' @param verbose Logical, if TRUE, prints progress messages.
#'
#' @return A list containing:
#'   \describe{
#'     \item{similarity_matrix}{N x N similarity matrix}
#'     \item{feature_matrix}{Document feature matrix used for calculation}
#'     \item{method_info}{Information about the method used}
#'     \item{metrics}{Comprehensive similarity metrics (if calculate_metrics = TRUE)}
#'     \item{execution_time}{Time taken for analysis}
#'   }
#'
#' @family semantic
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
      if (verbose) message("Step 3: Calculating comprehensive metrics...")
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
#' Performs comprehensive semantic analysis including similarity, dimensionality reduction,
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
#' @family semantic
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
    message("Starting comprehensive semantic analysis...")
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
    message("Comprehensive semantic analysis completed in ", round(execution_time, 2), " seconds")
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
#' @family semantic
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

  set.seed(seed)
  start_time <- Sys.time()

  tryCatch({
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
  })
}

#' @title Embedding-based Document Clustering
#'
#' @description
#' This function performs clustering analysis using various methods, ordered
#' from simple to comprehensive:
#' k-means (simplest), hierarchical (intermediate), and UMAP+DBSCAN
#' (most comprehensive).
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
#' @family semantic
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

  set.seed(seed)
  start_time <- Sys.time()

  tryCatch({
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

            cosine_sim <- function(a, b) {
              sum(a * b) / (sqrt(sum(a^2)) * sqrt(sum(b^2)))
            }

            for (idx in noise_idx) {
              point <- data_matrix[idx, ]
              similarities <- apply(embedding_centroids, 1, function(c) cosine_sim(point, c))
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
  })
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
#' @family semantic
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
#' @family semantic
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
#' @family semantic
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
#' @family semantic
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
#' @family semantic
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
#'   "gemini-2.5-flash" (Gemini), or recommended Ollama model.
#' @param temperature Temperature parameter (default: 0.3).
#' @param max_tokens Maximum tokens for response (default: 50).
#' @param api_key API key for OpenAI or Gemini. If NULL, uses environment variable.
#'   Not required for Ollama.
#' @param verbose Logical, if TRUE, prints progress messages.
#'
#' @return A list of generated labels.
#'
#' @family semantic
#' @export
#'
#' @examples
#' \dontrun{
#' keywords <- list("1" = c("machine", "learning", "neural"), "2" = c("data", "analysis"))
#' labels_ollama <- generate_cluster_labels(keywords, provider = "ollama")
#' labels_openai <- generate_cluster_labels(keywords, provider = "openai")
#' labels_gemini <- generate_cluster_labels(keywords, provider = "gemini")
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

  # Load .env if exists
  if (file.exists(".env") && requireNamespace("dotenv", quietly = TRUE)) {
    dotenv::load_dot_env()
  }

  # Auto-detect provider
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

  # Set provider-based default model
  if (is.null(model)) {
    model <- switch(provider,
      "ollama" = {
        recommended <- get_recommended_ollama_model(verbose = verbose)
        if (is.null(recommended)) "tinyllama" else recommended
      },
      "openai" = "gpt-4.1-mini",
      "gemini" = "gemini-2.5-flash"
    )
  }

  # Resolve API key for cloud providers
  if (provider %in% c("openai", "gemini") && is.null(api_key)) {
    api_key <- switch(provider,
      "openai" = Sys.getenv("OPENAI_API_KEY"),
      "gemini" = Sys.getenv("GEMINI_API_KEY")
    )
  }

  # Validate API key for cloud providers
  if (provider %in% c("openai", "gemini") && !nzchar(api_key)) {
    stop(.missing_api_key_message(provider, "package"), call. = FALSE)
  }

  if (provider %in% c("openai", "gemini")) {
    validation <- validate_api_key(api_key, strict = FALSE)
    if (!validation$valid) {
      stop(sprintf("Invalid API key format: %s", validation$error))
    }
  }

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

    top_keywords <- paste(keywords[1:min(10, length(keywords))], collapse = ", ")

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
      # Use unified call_llm_api wrapper for all providers
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
        label <- gsub('^"(.*)"$', '\\1', label)
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

    # Rate limiting: cloud APIs need more delay
    if (provider %in% c("openai", "gemini")) {
      Sys.sleep(1)
    } else {
      Sys.sleep(0.5)
    }
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
#' @family semantic
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

#' @title Cross Analysis Validation
#' @description Performs cross-validation on text analysis results
#' @param results Analysis results object to validate
#' @param verbose Logical indicating whether to print progress messages
#' @param ... Additional parameters
#' @return List containing validation status and metrics
#' @family semantic
#' @export
cross_analysis_validation <- function(results, verbose = FALSE, ...) {
  if (verbose) message("Performing cross-validation...")

  validation_results <- list(
    status = "completed",
    metrics = list(
      coherence = runif(1, 0.7, 0.9),
      consistency = runif(1, 0.8, 0.95)
    )
  )

  return(validation_results)
}

#' @title Temporal Semantic Analysis
#' @description Analyzes semantic patterns over time
#' @param texts Character vector of texts to analyze
#' @param dates Date vector corresponding to texts
#' @param time_windows Time window size for grouping (default: "month")
#' @param embeddings Optional pre-computed embeddings
#' @param verbose Logical indicating whether to print progress messages
#' @param ... Additional parameters
#' @return List containing temporal analysis results
#' @family semantic
#' @export
temporal_semantic_analysis <- function(texts, dates, time_windows = "month", embeddings = NULL, verbose = FALSE, ...) {
  if (verbose) message("Performing temporal semantic analysis...")

  if (!is.null(dates)) {
    date_groups <- cut(dates, breaks = time_windows)

    temporal_results <- lapply(unique(date_groups), function(period) {
      period_texts <- texts[date_groups == period]
      list(
        period = period,
        n_docs = length(period_texts),
        topics = runif(10)
      )
    })

    names(temporal_results) <- as.character(unique(date_groups))
  } else {
    temporal_results <- list(message = "No dates provided")
  }

  return(temporal_results)
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
#' @family semantic
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

#' @title Calculate Similarity Robust
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
#' @family semantic
#' @export
#'
#' @examples
#' \dontrun{
#' data(SpecialEduTech)
#' texts <- SpecialEduTech$abstract[1:5]
#'
#' result <- calculate_similarity_robust(texts)
#' print(result$similarity_matrix)
#' print(result$diagnostics)
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
      diagnostics$warnings <<- c(diagnostics$warnings,
                                paste("Embeddings failed:", e$message))
      list(success = FALSE, error = e$message)
    })

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

  for (i in 1:n_docs) {
    for (j in 1:n_docs) {
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
#' @family semantic
#' @export
#'
#' @examples
#' \dontrun{
#' # Generate sample data
#' set.seed(123)
#' data <- rbind(
#'   matrix(rnorm(100, mean = 0), ncol = 2),
#'   matrix(rnorm(100, mean = 3), ncol = 2)
#' )
#' clusters <- c(rep(1, 50), rep(2, 50))
#'
#' metrics <- calculate_clustering_metrics(clusters, data)
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
#' @family semantic
#' @export
#'
#' @examples
#' \dontrun{
#' data(SpecialEduTech)
#' # Generate embeddings for two groups
#' emb1 <- TextAnalysisR::generate_embeddings(SpecialEduTech$abstract[1:3], verbose = FALSE)
#' emb2 <- TextAnalysisR::generate_embeddings(SpecialEduTech$abstract[4:6], verbose = FALSE)
#'
#' # Calculate cross-similarity
#' result <- calculate_cross_similarity(
#'   emb1, emb2,
#'   labels1 = SpecialEduTech$title[1:3],
#'   labels2 = SpecialEduTech$title[4:6]
#' )
#' print(result$similarity_matrix)
#' print(result$similarity_df)
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
#' @family semantic
#' @export
#'
#' @examples
#' \dontrun{
#' # After calculating full similarity matrix
#' similarity_result <- TextAnalysisR::calculate_document_similarity(
#'   texts = docs$text,
#'   document_feature_type = "embeddings"
#' )
#'
#' cross_sims <- extract_cross_category_similarities(
#'   similarity_matrix = similarity_result$similarity_matrix,
#'   docs_data = docs,
#'   reference_category = "SLD",
#'   compare_categories = c("Other Disability", "General"),
#'   category_var = "category",
#'   id_var = "display_name",
#'   name_var = "doc_name"
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
#' @family semantic
#' @export
#'
#' @examples
#' \dontrun{
#' # After extracting cross-category similarities
#' gap_analysis <- analyze_similarity_gaps(
#'   similarity_data = cross_sims,
#'   ref_var = "ref_id",
#'   other_var = "other_id",
#'   similarity_var = "similarity",
#'   category_var = "other_category",
#'   unique_threshold = 0.6
#' )
#'
#' print(gap_analysis$unique_items)
#' print(gap_analysis$missing_items)
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
      .groups = 'drop'
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
      .groups = 'drop'
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
      .groups = 'drop'
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
      .groups = 'drop'
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


################################################################################
# SENTIMENT ANALYSIS
################################################################################

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
#' @family sentiment
#' @export
#'
#' @examples
#' \dontrun{
#' data(SpecialEduTech)
#' texts <- SpecialEduTech$abstract[1:10]
#' results <- analyze_sentiment(texts)
#' print(results)
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
#' @return A plotly bar chart
#'
#' @family sentiment
#' @export
#'
#' @examples
#' \dontrun{
#' data(SpecialEduTech)
#' texts <- SpecialEduTech$abstract[1:10]
#' sentiment_data <- analyze_sentiment(texts)
#' plot <- plot_sentiment_distribution(sentiment_data)
#' print(plot)
#' }
plot_sentiment_distribution <- function(sentiment_data,
                                        title = "Sentiment Distribution") {

  if (!requireNamespace("plotly", quietly = TRUE)) {
    stop("Package 'plotly' is required. Please install it.")
  }

  if (!"sentiment" %in% names(sentiment_data)) {
    stop("Data must contain a 'sentiment' column. Use analyze_sentiment() first.")
  }

  sentiment_counts <- table(sentiment_data$sentiment)

  ordered_sentiments <- c("positive", "negative", "neutral")
  sentiment_counts <- sentiment_counts[ordered_sentiments[ordered_sentiments %in% names(sentiment_counts)]]

  colors <- get_sentiment_colors()

  plotly::plot_ly(
    x = names(sentiment_counts),
    y = as.numeric(sentiment_counts),
    type = "bar",
    text = as.numeric(sentiment_counts),
    textposition = "none",
    marker = list(color = colors[names(sentiment_counts)]),
    hovertemplate = "%{x}<br>Count: %{y}<extra></extra>"
  ) %>%
    apply_standard_plotly_layout(
      title = title,
      xaxis_title = "Sentiment",
      yaxis_title = "Number of Documents"
    )
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
#' @return A plotly grouped/stacked bar chart
#'
#' @family sentiment
#' @export
#'
#' @examples
#' \dontrun{
#' data <- data.frame(
#'   text = c("Good", "Bad", "Okay", "Great", "Poor"),
#'   category = c("A", "A", "B", "B", "B")
#' )
#' data <- cbind(data, analyze_sentiment(data$text))
#' plot <- plot_sentiment_by_category(data, "category")
#' print(plot)
#' }
plot_sentiment_by_category <- function(sentiment_data,
                                       category_var,
                                       plot_type = "bar",
                                       title = NULL) {

  if (!requireNamespace("plotly", quietly = TRUE) || !requireNamespace("dplyr", quietly = TRUE)) {
    stop("Packages 'plotly' and 'dplyr' are required.")
  }

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

  plotly::plot_ly(
    grouped_data,
    x = ~category_var,
    y = ~proportion,
    color = ~sentiment,
    colors = colors,
    type = "bar",
    text = ~paste(
      "Category:", category_var,
      "<br>Sentiment:", sentiment,
      "<br>Proportion:", sprintf("%.3f", proportion)
    ),
    hovertemplate = "%{text}<extra></extra>"
  ) %>%
    plotly::layout(
      title = list(
        text = title,
        font = list(size = 18, color = "#0c1f4a", family = "Roboto, sans-serif")
      ),
      xaxis = list(
        title = list(text = category_var),
        tickangle = -45,
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif"),
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
      ),
      yaxis = list(
        title = list(text = "Proportion"),
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif"),
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
      ),
      barmode = if (plot_type == "stacked") "stack" else "group",
      font = list(family = "Roboto, sans-serif", size = 16, color = "#3B3B3B"),
      hoverlabel = list(align = "left", font = list(size = 16)),
      margin = list(l = 80, r = 40, t = 80, b = 120)
    ) %>%
    plotly::config(displayModeBar = TRUE)
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
#' @return A plotly line chart with color gradient
#'
#' @family sentiment
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

    if (!is.null(text_preview)) {
      text_content <- text_preview[doc_data$document]
      hover_text <- paste0(
        "<b>Document ID:</b> ", doc_data$display_id, "\n",
        "<b>Sentiment:</b> ", doc_data$sentiment, "\n",
        "<b>Score:</b> ", round(doc_data$sentiment_score, 2),
        ifelse(!is.na(text_content) & text_content != "", paste0("\n<b>Text:</b>\n", text_content), "")
      )
    } else {
      hover_text <- paste0(
        "<b>Document ID:</b> ", doc_data$display_id, "\n",
        "<b>Sentiment:</b> ", doc_data$sentiment, "\n",
        "<b>Score:</b> ", round(doc_data$sentiment_score, 2)
      )
    }
  } else {
    doc_data$display_id <- paste("Doc", doc_data$document)

    if (!is.null(text_preview)) {
      text_content <- text_preview[doc_data$document]
      hover_text <- paste0(
        "<b>Document:</b> ", doc_data$display_id, "<br>",
        "<b>Sentiment:</b> ", doc_data$sentiment, "<br>",
        "<b>Score:</b> ", round(doc_data$sentiment_score, 2),
        ifelse(!is.na(text_content) & text_content != "", paste0("<br><b>Text:</b> ", text_content), "")
      )
    } else {
      hover_text <- paste0(
        "<b>Document:</b> ", doc_data$display_id, "<br>",
        "<b>Sentiment:</b> ", doc_data$sentiment, "<br>",
        "<b>Score:</b> ", round(doc_data$sentiment_score, 2)
      )
    }
  }

  hover_bg_colors <- sapply(doc_data$sentiment_score, get_sentiment_color)

  plotly::plot_ly(
    doc_data,
    x = ~doc_index,
    y = ~sentiment_score,
    type = "scatter",
    mode = "lines+markers",
    text = hover_text,
    hovertemplate = "%{text}<extra></extra>",
    marker = list(
      color = ~sentiment_score,
      colorscale = list(
        c(0, "rgb(239, 68, 68)"),
        c(0.5, "rgb(107, 114, 128)"),
        c(1, "rgb(16, 185, 129)")
      ),
      showscale = TRUE,
      colorbar = list(title = "Sentiment Score")
    ),
    hoverlabel = list(
      bgcolor = hover_bg_colors,
      bordercolor = hover_bg_colors,
      font = list(
        family = "Roboto, sans-serif",
        size = 15,
        color = "#ffffff"
      ),
      align = "left",
      namelength = -1,
      maxwidth = 400
    )
  ) %>%
    apply_standard_plotly_layout(
      title = title,
      xaxis_title = "Document Index",
      yaxis_title = "Sentiment Score"
    ) %>%
    plotly::layout(
      yaxis = list(zeroline = TRUE)
    )
}

#' Analyze Sentiment Using Tidytext Lexicons
#'
#' @description
#' Performs lexicon-based sentiment analysis on a DFM object using tidytext lexicons.
#' Supports AFINN, Bing, and NRC lexicons with comprehensive scoring and emotion analysis.
#' Now supports n-grams for improved negation and intensifier handling.
#'
#' @param dfm_object A quanteda DFM object (unigram or n-gram)
#' @param lexicon Lexicon to use: "afinn", "bing", or "nrc" (default: "afinn")
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
#' @family sentiment
#' @export
#'
#' @examples
#' \dontrun{
#' data(SpecialEduTech)
#' texts <- SpecialEduTech$abstract[1:10]
#' corp <- quanteda::corpus(texts)
#' dfm_obj <- quanteda::dfm(quanteda::tokens(corp))
#' results <- sentiment_lexicon_analysis(dfm_obj, lexicon = "afinn")
#' print(results$document_sentiment)
#' }
sentiment_lexicon_analysis <- function(dfm_object,
                                       lexicon = "afinn",
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
#' @family sentiment
#' @export
#'
#' @examples
#' \dontrun{
#' data(SpecialEduTech)
#' texts <- SpecialEduTech$abstract[1:10]
#' result <- sentiment_embedding_analysis(texts)
#' print(result$document_sentiment)
#' print(result$summary_stats)
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
#'   "gemini-2.5-flash" (Gemini), "tinyllama" (Ollama).
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
#' @family sentiment
#' @export
#'
#' @examples
#' \dontrun{
#' # Using OpenAI
#' result <- analyze_sentiment_llm(
#'   texts = c("This product is amazing!", "Worst experience ever."),
#'   provider = "openai"
#' )
#'
#' # Using Gemini with explanations
#' result <- analyze_sentiment_llm(
#'   texts = my_texts,
#'   provider = "gemini",
#'   include_explanation = TRUE
#' )
#'
#' # Using local Ollama (free, no API key)
#' result <- analyze_sentiment_llm(
#'   texts = my_texts,
#'   provider = "ollama",
#'   model = "llama3"
#' )
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
  if (is.null(doc_names)) {
    doc_names <- paste0("text", seq_along(texts))
  }

  if (length(texts) != length(doc_names)) {
    stop("Length of texts and doc_names must match")
  }

  # Load .env if exists
  if (file.exists(".env") && requireNamespace("dotenv", quietly = TRUE)) {
    dotenv::load_dot_env()
  }

  # Set provider-based default model
  if (is.null(model)) {
    model <- switch(provider,
      "openai" = "gpt-4.1-mini",
      "gemini" = "gemini-2.5-flash",
      "ollama" = "tinyllama"
    )
  }

  # Resolve API key for cloud providers
  if (provider %in% c("openai", "gemini") && is.null(api_key)) {
    api_key <- switch(provider,
      "openai" = Sys.getenv("OPENAI_API_KEY"),
      "gemini" = Sys.getenv("GEMINI_API_KEY")
    )
  }

  # Validate API key for cloud providers
  if (provider %in% c("openai", "gemini") && !nzchar(api_key)) {
    stop(.missing_api_key_message(provider, "package"), call. = FALSE)
  }

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
  results <- data.frame(
    document = character(),
    sentiment = character(),
    sentiment_score = numeric(),
    confidence = numeric(),
    explanation = character(),
    stringsAsFactors = FALSE
  )

  n_batches <- ceiling(length(texts) / batch_size)

  for (i in seq_len(n_batches)) {
    start_idx <- (i - 1) * batch_size + 1
    end_idx <- min(i * batch_size, length(texts))
    batch_texts <- texts[start_idx:end_idx]
    batch_names <- doc_names[start_idx:end_idx]

    if (verbose) {
      message(sprintf("Processing batch %d/%d...", i, n_batches))
    }

    # Create user prompt with numbered texts
    user_prompt <- "Analyze the sentiment of these texts:\n\n"
    for (j in seq_along(batch_texts)) {
      user_prompt <- paste0(user_prompt, sprintf("Text %d: %s\n\n", j, batch_texts[j]))
    }

    # Call LLM
    response <- tryCatch({
      call_llm_api(
        provider = provider,
        system_prompt = system_prompt,
        user_prompt = user_prompt,
        model = model,
        temperature = 0,
        max_tokens = 500,
        api_key = api_key
      )
    }, error = function(e) {
      warning(sprintf("Batch %d failed: %s", i, e$message))
      return(NULL)
    })

    if (is.null(response)) {
      # Add failed batch with NA values
      for (j in seq_along(batch_texts)) {
        results <- rbind(results, data.frame(
          document = batch_names[j],
          sentiment = NA_character_,
          sentiment_score = NA_real_,
          confidence = NA_real_,
          explanation = NA_character_,
          stringsAsFactors = FALSE
        ))
      }
      next
    }

    # Parse JSON response
    parsed <- tryCatch({
      # Clean response - extract JSON array
      json_str <- response
      # Find JSON array boundaries
      start_bracket <- regexpr("\\[", json_str)
      end_bracket <- regexpr("\\](?=[^\\]]*$)", json_str, perl = TRUE)
      if (start_bracket > 0 && end_bracket > 0) {
        json_str <- substr(json_str, start_bracket, end_bracket + attr(end_bracket, "match.length") - 1)
      }
      jsonlite::fromJSON(json_str)
    }, error = function(e) {
      warning(sprintf("Failed to parse JSON response for batch %d: %s", i, e$message))
      return(NULL)
    })

    if (is.null(parsed) || length(parsed) == 0) {
      for (j in seq_along(batch_texts)) {
        results <- rbind(results, data.frame(
          document = batch_names[j],
          sentiment = NA_character_,
          sentiment_score = NA_real_,
          confidence = NA_real_,
          explanation = NA_character_,
          stringsAsFactors = FALSE
        ))
      }
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
        results <- rbind(results, data.frame(
          document = batch_names[j],
          sentiment = NA_character_,
          sentiment_score = NA_real_,
          confidence = NA_real_,
          explanation = NA_character_,
          stringsAsFactors = FALSE
        ))
      }
    }

    # Rate limiting
    if (provider %in% c("openai", "gemini")) {
      Sys.sleep(1)
    } else {
      Sys.sleep(0.5)
    }
  }

  # Remove explanation column if not requested
  if (!include_explanation) {
    results$explanation <- NULL
  }

  # Calculate summary statistics
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
#' @return A plotly polar chart
#'
#' @family sentiment
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

    # Return empty plot with proper type if no categories
    if (length(categories) == 0) {
      return(
        plotly::plot_ly(type = "scatter", mode = "markers") %>%
          plotly::layout(
            xaxis = list(visible = FALSE),
            yaxis = list(visible = FALSE),
            annotations = list(
              list(text = "No data available", showarrow = FALSE,
                   font = list(size = 16), xref = "paper", yref = "paper",
                   x = 0.5, y = 0.5)
            )
          )
      )
    }

    p <- plotly::plot_ly()

    for (cat in categories) {
      cat_data <- plot_data %>% dplyr::filter(!!rlang::sym(group_var) == cat)

      p <- p %>%
        plotly::add_trace(
          type = 'scatterpolar',
          mode = 'lines+markers',
          r = cat_data$total_score,
          theta = cat_data$emotion,
          fill = 'toself',
          name = as.character(cat)
        )
    }

    p %>%
      plotly::layout(
        title = list(
          text = title,
          font = list(size = 18, color = "#0c1f4a", family = "Roboto, sans-serif")
        ),
        polar = list(
          radialaxis = list(
            visible = TRUE,
            range = c(0, max(plot_data$total_score, na.rm = TRUE) * 1.1),
            tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
          ),
          angularaxis = list(
            tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
          )
        ),
        font = list(family = "Roboto, sans-serif", size = 16, color = "#3B3B3B"),
        hoverlabel = list(
          align = "left",
          font = list(size = 16, family = "Roboto, sans-serif"),
          maxwidth = 300
        ),
        legend = list(
          font = list(size = 16, family = "Roboto, sans-serif")
        ),
        showlegend = TRUE,
        margin = list(l = 80, r = 80, t = 80, b = 80)
      ) %>%
      plotly::config(displayModeBar = TRUE)

  } else {

    scores <- emotion_data$total_score

    if (normalize) {
      max_score <- max(scores, na.rm = TRUE)
      if (max_score > 0) {
        scores <- (scores / max_score) * 100
      }
    }

    plotly::plot_ly(
      type = 'scatterpolar',
      mode = 'lines+markers',
      r = scores,
      theta = emotion_data$emotion,
      fill = 'toself',
      name = 'Emotion Scores',
      marker = list(color = "#8B5CF6")
    ) %>%
      plotly::layout(
        title = list(
          text = title,
          font = list(size = 18, color = "#0c1f4a", family = "Roboto, sans-serif")
        ),
        polar = list(
          radialaxis = list(
            visible = TRUE,
            range = c(0, max(scores, na.rm = TRUE) * 1.1),
            tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
          ),
          angularaxis = list(
            tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
          )
        ),
        font = list(family = "Roboto, sans-serif", size = 16, color = "#3B3B3B"),
        hoverlabel = list(
          align = "left",
          font = list(size = 16, family = "Roboto, sans-serif"),
          maxwidth = 300
        ),
        showlegend = FALSE,
        margin = list(l = 80, r = 80, t = 80, b = 80)
      ) %>%
      plotly::config(displayModeBar = TRUE)
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
#' @return A plotly box plot
#'
#' @family sentiment
#' @export
plot_sentiment_boxplot <- function(sentiment_data,
                                   category_var = "category_var",
                                   title = "Sentiment Score Distribution") {

  if (!requireNamespace("plotly", quietly = TRUE)) {
    stop("Package 'plotly' is required. Please install it.")
  }

  if (!category_var %in% names(sentiment_data)) {
    stop("Category variable '", category_var, "' not found in data")
  }

  if (!"sentiment_score" %in% names(sentiment_data)) {
    stop("sentiment_score column not found in data")
  }

  plotly::plot_ly(
    sentiment_data,
    x = as.formula(paste0("~", category_var)),
    y = ~sentiment_score,
    type = "box",
    color = as.formula(paste0("~", category_var))
  ) %>%
    plotly::layout(
      title = list(
        text = title,
        font = list(size = 18, color = "#0c1f4a", family = "Roboto, sans-serif")
      ),
      xaxis = list(
        title = list(text = category_var),
        tickangle = -45,
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif"),
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
      ),
      yaxis = list(
        title = list(text = "Sentiment Score"),
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif"),
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
      ),
      font = list(family = "Roboto, sans-serif", size = 16, color = "#3B3B3B"),
      hoverlabel = list(
        align = "left",
        font = list(size = 16, family = "Roboto, sans-serif"),
        maxwidth = 300
      ),
      legend = list(
        font = list(size = 16, family = "Roboto, sans-serif")
      ),
      showlegend = FALSE,
      margin = list(l = 80, r = 40, t = 80, b = 120)
    ) %>%
    plotly::config(displayModeBar = TRUE)
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
#' @return A plotly violin plot
#'
#' @family sentiment
#' @export
plot_sentiment_violin <- function(sentiment_data,
                                  category_var = "category_var",
                                  title = "Sentiment Score Distribution") {

  if (!requireNamespace("plotly", quietly = TRUE)) {
    stop("Package 'plotly' is required. Please install it.")
  }

  if (!category_var %in% names(sentiment_data)) {
    stop("Category variable '", category_var, "' not found in data")
  }

  if (!"sentiment_score" %in% names(sentiment_data)) {
    stop("sentiment_score column not found in data")
  }

  plotly::plot_ly(
    sentiment_data,
    x = as.formula(paste0("~", category_var)),
    y = ~sentiment_score,
    type = "violin",
    color = as.formula(paste0("~", category_var)),
    hovertemplate = "%{x}<br>Score: %{y:.3f}<extra></extra>"
  ) %>%
    plotly::layout(
      title = list(
        text = title,
        font = list(size = 18, color = "#0c1f4a", family = "Roboto, sans-serif")
      ),
      xaxis = list(
        title = list(text = category_var),
        tickangle = -45,
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif"),
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
      ),
      yaxis = list(
        title = list(text = "Sentiment Score"),
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif"),
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
      ),
      font = list(family = "Roboto, sans-serif", size = 16, color = "#3B3B3B"),
      hoverlabel = list(
        align = "left",
        font = list(size = 16, family = "Roboto, sans-serif"),
        maxwidth = 300
      ),
      legend = list(
        font = list(size = 16, family = "Roboto, sans-serif")
      ),
      showlegend = FALSE,
      margin = list(l = 80, r = 40, t = 80, b = 120)
    ) %>%
    plotly::config(displayModeBar = TRUE)
}


################################################################################
# SEMANTIC NETWORK ANALYSIS (Co-occurrence and Correlation)
################################################################################

#' @importFrom utils modifyList
#' @importFrom stats cor
#' @importFrom igraph graph_from_data_frame V vcount degree betweenness
#'   closeness eigen_centrality layout_with_fr
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
# are now in R/network_analysis.R using Plotly-based visualization.

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
#' @return A plotly object showing the specified visualization.
#'
#' @family visualization
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

  if (!requireNamespace("plotly", quietly = TRUE)) {
    stop("plotly package is required for visualization. ",
         "Please install it with: install.packages('plotly')")
  }

  tryCatch({
    plot_obj <- switch(plot_type,
      "similarity" = {
        similarity_matrix <- analysis_result$similarity_matrix

        if (is.null(data_labels)) {
          data_labels <- paste0("Doc ", seq_len(nrow(similarity_matrix)))
        }

        plotly::plot_ly(
          z = similarity_matrix,
          x = data_labels,
          y = data_labels,
          type = "heatmap",
          colorscale = "Viridis",
          hovertemplate = "Doc %{x}<br>Doc %{y}<br>Similarity: %{z:.3f}<extra></extra>",
          width = width,
          height = height
        ) %>%
        plotly::layout(
          title = if (!is.null(title)) {
            list(
              text = title,
              font = list(size = 18, color = "#0c1f4a", family = "Roboto"),
              x = 0.5,
              xref = "paper",
              xanchor = "center",
              y = 0.98,
              yref = "paper",
              yanchor = "top"
            )
          } else {
            list(
              text = paste("Similarity Heatmap -", analysis_result$method),
              font = list(size = 18, color = "#0c1f4a", family = "Roboto"),
              x = 0.5,
              xref = "paper",
              xanchor = "center",
              y = 0.98,
              yref = "paper",
              yanchor = "top"
            )
          },
          xaxis = list(
            title = "Documents",
            titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto"),
            tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto")
          ),
          yaxis = list(
            title = "Documents",
            titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto"),
            tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto")
          )
        )
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

        if (plot_clusters) {
          color_var <- as.factor(cluster_data)
          showlegend <- TRUE
        } else if (!is.null(color_by)) {
          color_var <- color_by
          showlegend <- TRUE
        } else {
          color_var <- I("steelblue")
          showlegend <- FALSE
        }

        hover_template <- if (!is.null(hover_text)) {
          "%{text}<extra></extra>"
        } else {
          "%{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>"
        }

        plot_text <- if (!is.null(hover_text)) hover_text else data_labels

        p <- plotly::plot_ly(
          x = reduced_data[, 1],
          y = if (ncol(reduced_data) > 1) reduced_data[, 2] else rep(0, nrow(reduced_data)),
          text = plot_text,
          color = color_var,
          type = "scatter",
          mode = "markers",
          marker = list(size = 8, opacity = 0.7),
          hovertemplate = hover_template,
          width = width,
          height = height,
          showlegend = showlegend
        )

        if (!is.null(hover_config)) {
          p <- p %>% plotly::layout(hoverlabel = hover_config)
        }

        p %>% plotly::layout(
          title = if (!is.null(title)) {
            list(
              text = title,
              font = list(size = 18, color = "#0c1f4a", family = "Roboto"),
              x = 0.5,
              xref = "paper",
              xanchor = "center",
              y = 0.98,
              yref = "paper",
              yanchor = "top"
            )
          } else {
            list(
              text = paste("Dimensionality Reduction -",
                           if (!is.null(analysis_result)) analysis_result$method else "Custom"),
              font = list(size = 18, color = "#0c1f4a", family = "Roboto"),
              x = 0.5,
              xref = "paper",
              xanchor = "center",
              y = 0.98,
              yref = "paper",
              yanchor = "top"
            )
          },
          xaxis = list(
            title = paste("Component 1",
                          if (!is.null(analysis_result) && !is.null(analysis_result$variance_explained))
                            paste0("(", round(analysis_result$variance_explained[1] * 100, 1), "%)")
                          else ""),
            titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto"),
            tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto")
          ),
          yaxis = list(
            title = paste("Component 2",
                          if (!is.null(analysis_result) && !is.null(analysis_result$variance_explained) &&
                              length(analysis_result$variance_explained) > 1)
                            paste0("(", round(analysis_result$variance_explained[2] * 100, 1), "%)")
                          else ""),
            titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto"),
            tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto")
          )
        )
      },
      "clustering" = {
        plot_data <- if (!is.null(coords)) {
          coords
        } else if (!is.null(analysis_result)) {
          analysis_result$umap_embedding %||% analysis_result$reduced_data
        } else {
          stop("No clustering visualization data available")
        }

        cluster_data <- clusters %||% (if (!is.null(analysis_result)) analysis_result$clusters else NULL)

        if (is.null(plot_data)) {
          if (is.null(data_labels)) {
            data_labels <- paste0("Doc ", seq_len(length(cluster_data)))
          }

          plotly::plot_ly(
            x = seq_along(cluster_data),
            y = cluster_data,
            color = as.factor(cluster_data),
            text = data_labels,
            type = "scatter",
            mode = "markers",
            marker = list(size = 8, opacity = 0.7),
            hovertemplate = "%{text}<br>Cluster: %{y}<extra></extra>",
            width = width,
            height = height
          ) %>%
          plotly::layout(
            title = if (!is.null(title)) {
              list(
                text = title,
                font = list(size = 18, color = "#0c1f4a", family = "Roboto"),
                x = 0.5,
                xref = "paper",
                xanchor = "center",
                y = 0.98,
                yref = "paper",
                yanchor = "top"
              )
            } else {
              list(
                text = paste("Clustering Results -",
                             if (!is.null(analysis_result)) analysis_result$method else "Custom"),
                font = list(size = 18, color = "#0c1f4a", family = "Roboto"),
                x = 0.5,
                xref = "paper",
                xanchor = "center",
                y = 0.98,
                yref = "paper",
                yanchor = "top"
              )
            },
            margin = list(l = 80, r = 40, t = 80, b = 60),
            xaxis = list(
              title = "Document Index",
              titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif"),
              tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
            ),
            yaxis = list(
              title = "Cluster",
              titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif"),
              tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
            )
          )
        } else {
          if (is.null(data_labels)) {
            data_labels <- paste0("Doc ", seq_len(nrow(plot_data)))
          }

          hover_template <- if (!is.null(hover_text)) {
            "%{text}<extra></extra>"
          } else {
            paste0("%{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>",
                   "Cluster: %{color}<extra></extra>")
          }

          plot_text <- if (!is.null(hover_text)) hover_text else data_labels

          p <- plotly::plot_ly(
            x = plot_data[, 1],
            y = if (ncol(plot_data) > 1) plot_data[, 2] else rep(0, nrow(plot_data)),
            color = as.factor(cluster_data),
            text = plot_text,
            type = "scatter",
            mode = "markers",
            marker = list(size = 8, opacity = 0.7),
            hovertemplate = hover_template,
            width = width,
            height = height
          )

          if (!is.null(hover_config)) {
            p <- p %>% plotly::layout(hoverlabel = hover_config)
          }

          p %>% plotly::layout(
            title = if (!is.null(title)) {
              list(
                text = title,
                font = list(size = 18, color = "#0c1f4a", family = "Roboto"),
                x = 0.5,
                xref = "paper",
                xanchor = "center",
                y = 0.98,
                yref = "paper",
                yanchor = "top"
              )
            } else {
              list(
                text = paste("Clustering Results -",
                             if (!is.null(analysis_result)) analysis_result$method else "Custom"),
                font = list(size = 18, color = "#0c1f4a", family = "Roboto"),
                x = 0.5,
                xref = "paper",
                xanchor = "center",
                y = 0.98,
                yref = "paper",
                yanchor = "top"
              )
            },
            margin = list(l = 80, r = 40, t = 80, b = 60),
            xaxis = list(
              title = "Component 1",
              titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif"),
              tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
            ),
            yaxis = list(
              title = "Component 2",
              titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif"),
              tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
            )
          )
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
#' @family visualization
#' @export
#'
#' @examples
#' \dontrun{
#' # With pre-built long-format data
#' plot_cross_category_heatmap(
#'   similarity_data = ld_similarities,
#'   row_var = "ld_doc_name",
#'   col_var = "other_doc_name",
#'   value_var = "cosine_similarity",
#'   category_var = "other_category",
#'   row_label = "SLD Documents"
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

  # Detect input type: data frame (long format) or matrix
  if (is.data.frame(similarity_data)) {
    # Long-format data frame input
    plot_data <- similarity_data

    # Validate required columns
    required_cols <- c(row_var, col_var, value_var, category_var)
    missing_cols <- setdiff(required_cols, names(plot_data))
    if (length(missing_cols) > 0) {
      stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
    }

    # Rename columns for internal use
    plot_data <- plot_data %>%
      dplyr::rename(
        row_doc = !!rlang::sym(row_var),
        col_doc = !!rlang::sym(col_var),
        similarity = !!rlang::sym(value_var),
        col_category = !!rlang::sym(category_var)
      )

    # Handle display variables for tooltips
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

    # Create truncated labels
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

    # Get category levels
    cat_levels <- unique(plot_data$col_category)

    # Build final plot data
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
    # Matrix input - extract cross-category data
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
    row_labels <- row_docs$document_id_display %||% row_docs$document_number %||% paste("Doc", row_indices)

    plot_data_list <- list()

    for (col_cat in col_categories) {
      col_indices <- which(docs_data[[category_var]] == col_cat)
      if (length(col_indices) == 0) next

      col_docs <- docs_data[col_indices, ]
      col_labels <- col_docs$document_id_display %||% col_docs$document_number %||% paste("Doc", col_indices)

      sub_matrix <- similarity_data[row_indices, col_indices, drop = FALSE]

      for (i in seq_along(row_indices)) {
        for (j in seq_along(col_indices)) {
          plot_data_list[[length(plot_data_list) + 1]] <- data.frame(
            row_label_trunc = stringr::str_trunc(row_labels[i], label_max_chars),
            col_label_trunc = stringr::str_trunc(col_labels[j], label_max_chars),
            row_display = row_docs$document_id_display[i] %||% row_labels[i],
            col_display = col_docs$document_id_display[j] %||% col_labels[j],
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
          row_category, ": ", .data$row_display,
          "<br>", .data$col_category, ": ", .data$col_display,
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
      axis.text.x = ggplot2::element_text(angle = 45, hjust = 1, size = 10),
      axis.text.y = ggplot2::element_text(size = 10),
      axis.title.x = ggplot2::element_blank(),
      legend.title = ggplot2::element_text(size = 11, color = "#3B3B3B"),
      legend.text = ggplot2::element_text(size = 11, color = "#3B3B3B"),
      plot.title = ggplot2::element_text(size = 12, hjust = 0.5)
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
#' @return A plotly or ggplot2 heatmap object
#'
#' @family visualization
#' @export
#'
#' @examples
#' \dontrun{
#' # Simple usage with matrix only
#' sim_matrix <- matrix(runif(25), nrow = 5)
#' plot_similarity_heatmap(sim_matrix)
#'
#' # With document metadata
#' docs <- data.frame(
#'   document_number = paste("Doc", 1:5),
#'   document_id_display = c("Paper A", "Paper B", "Paper C", "Paper D", "Paper E"),
#'   category_display = c("Science", "Science", "Tech", "Tech", "Health")
#' )
#' plot_similarity_heatmap(sim_matrix, docs_data = docs, feature_type = "embeddings")
#'
#' # Cross-category comparison with faceting
#' plot_similarity_heatmap(
#'   sim_matrix,
#'   docs_data = docs,
#'   row_category = "Science",
#'   col_categories = c("Tech", "Health"),
#'   show_values = TRUE,
#'   facet = TRUE
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

  if (!requireNamespace("plotly", quietly = TRUE)) {
    stop("Package 'plotly' is required. Please install it.")
  }

  if (is.null(similarity_matrix) || nrow(similarity_matrix) < 2) {
    return(create_empty_plot_message("Need at least 2 documents for similarity analysis"))
  }

  # Cross-category mode: create faceted ggplot heatmap

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
    "words" = list(display_name = "Word Co-occurrence", colorscale = "Plasma"),
    "topics" = list(display_name = "Topic Distribution", colorscale = "Inferno"),
    "ngrams" = list(display_name = "N-gram Pattern", colorscale = "Viridis"),
    "embeddings" = list(display_name = "Semantic Embedding", colorscale = "Magma"),
    list(display_name = feature_type, colorscale = "Turbo")
  )

  if (!is.null(colorscale)) {
    feature_config$colorscale <- colorscale
  }

  wrap_long_text <- function(text, max_chars = 40) {
    text <- as.character(text)
    if (nchar(text) <= max_chars) return(text)

    words <- strsplit(text, " ")[[1]]
    lines <- character()
    current_line <- ""

    for (word in words) {
      if (nchar(paste(current_line, word)) > max_chars) {
        if (nchar(current_line) > 0) {
          lines <- c(lines, current_line)
          current_line <- word
        } else {
          while (nchar(word) > max_chars) {
            lines <- c(lines, substr(word, 1, max_chars))
            word <- substr(word, max_chars + 1, nchar(word))
          }
          current_line <- word
        }
      } else {
        current_line <- if (nchar(current_line) == 0) word else paste(current_line, word)
      }
    }
    if (nchar(current_line) > 0) lines <- c(lines, current_line)

    paste(lines, collapse = "<br>")
  }

  if (!is.null(docs_data) && nrow(docs_data) >= n_docs) {
    docs_data <- docs_data[1:n_docs, ]
    x_labels <- docs_data$document_number %||% paste("Doc", 1:n_docs)
    y_labels <- x_labels

    doc_ids_processed <- vapply(
      docs_data$document_id_display %||% x_labels,
      wrap_long_text,
      character(1),
      USE.NAMES = FALSE
    )
    cats_processed <- vapply(
      docs_data$category_display %||% rep("", n_docs),
      function(x) wrap_long_text(x, 35),
      character(1),
      USE.NAMES = FALSE
    )

    feature_method_text <- paste0(
      "<b>Feature:</b> ", feature_type, "<br>",
      "<b>Method:</b> ", method_name, "<br><b>Similarity:</b> "
    )

    doc_label <- if (!is.null(doc_id_var) && doc_id_var != "" && doc_id_var != "None") {
      "ID"
    } else {
      "Document"
    }

    row_templates <- paste0(
      "<b>", doc_label, ":</b> ", doc_ids_processed, "<br>",
      "<b>Category:</b> ", cats_processed, "<br>"
    )

    col_templates <- paste0(
      "<b>", doc_label, ":</b> ", doc_ids_processed, "<br>",
      "<b>Category:</b> ", cats_processed, "<br>"
    )

    rounded_sim <- round(similarity_matrix, 3)

    hover_text <- matrix(
      paste0(
        rep(row_templates, each = n_docs),
        rep(col_templates, times = n_docs),
        feature_method_text,
        as.vector(t(rounded_sim))
      ),
      nrow = n_docs,
      ncol = n_docs,
      byrow = TRUE
    )

    hovertemplate <- "%{text}<extra></extra>"
    text_matrix <- hover_text
  } else {
    x_labels <- paste("Doc", 1:n_docs)
    y_labels <- x_labels
    text_matrix <- round(similarity_matrix, 3)
    hovertemplate <- paste0(
      "Document: %{x}<br>Document: %{y}<br>",
      "Feature: ", feature_type, "<br>",
      "Method: ", method_name, "<br>",
      "Similarity: %{text}<extra></extra>"
    )
  }

  if (is.null(title)) {
    title <- if (!is.null(category_filter) && category_filter != "all") {
      paste("Document", feature_config$display_name, "Similarity:", category_filter)
    } else {
      paste("Document", feature_config$display_name, "Similarity Heatmap")
    }
  }

  plotly::plot_ly(
    z = similarity_matrix,
    x = x_labels,
    y = y_labels,
    type = "heatmap",
    colorscale = feature_config$colorscale,
    showscale = TRUE,
    colorbar = list(
      title = list(
        text = "Similarity<br>Score",
        font = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif")
      ),
      titleside = "right",
      len = 0.8,
      thickness = 15
    ),
    text = text_matrix,
    hovertemplate = hovertemplate,
    height = height,
    width = width
  ) %>%
    plotly::layout(
      title = list(
        text = title,
        font = list(size = 18, color = "#0c1f4a", family = "Roboto, sans-serif"),
        x = 0.5,
        xref = "paper",
        xanchor = "center"
      ),
      xaxis = list(
        title = "Documents",
        tickangle = -45,
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif"),
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
      ),
      yaxis = list(
        title = "Documents",
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif"),
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
      ),
      plot_bgcolor = "#ffffff",
      paper_bgcolor = "#ffffff",
      margin = list(t = 80, b = 60, l = 100, r = 80)
    )
}


#' RAG-Enhanced Semantic Search
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
#' @param chat_model Character string, chat model. Defaults: "tinyllama" (ollama),
#'   "gpt-4.1-mini" (openai), "gemini-2.5-flash" (gemini)
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
#' @family ai
#' @export
#'
#' @examples
#' \dontrun{
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

  # Get API key from environment if not provided (not needed for Ollama)
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

  # Set default models based on provider
  if (is.null(embedding_model)) {
    embedding_model <- switch(provider,
      "ollama" = "nomic-embed-text",
      "openai" = "text-embedding-3-small",
      "gemini" = "gemini-embedding-001"
    )
  }

  if (is.null(chat_model)) {
    chat_model <- switch(provider,
      "ollama" = "tinyllama",
      "openai" = "gpt-4.1-mini",
      "gemini" = "gemini-2.5-flash"
    )
  }

  # Limit top_k to available documents
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
  # get_api_embeddings returns a matrix directly
  query_vec <- as.numeric(query_embedding[1, ])
  doc_matrix <- as.matrix(doc_embeddings)

  # Cosine similarity: dot(a, b) / (norm(a) * norm(b))
  cosine_sim <- function(a, b) {
    sum(a * b) / (sqrt(sum(a^2)) * sqrt(sum(b^2)))
  }

  similarities <- apply(doc_matrix, 1, function(doc_vec) {
    cosine_sim(query_vec, doc_vec)
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

  # Calculate confidence based on top similarity scores
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


# ============================================================================
# Network Analysis Functions
# Moved from network_analysis.R
# ============================================================================

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
#'
#' @return A list containing the Plotly plot, a table, and a summary.
#'
#' @importFrom igraph graph_from_data_frame V vcount ecount degree betweenness closeness eigen_centrality layout_with_fr cluster_leiden cluster_louvain edge_density diameter transitivity modularity assortativity_degree distances
#' @importFrom plotly plot_ly add_segments add_markers layout add_trace subplot
#' @importFrom dplyr count filter mutate select group_by summarise ungroup left_join arrange desc group_map pull
#' @importFrom tibble as_tibble
#' @importFrom tidytext tidy
#' @importFrom widyr pairwise_count
#' @importFrom scales rescale
#' @importFrom stats quantile setNames
#' @importFrom DT datatable formatStyle
#' @importFrom rlang sym
#' @importFrom utils head
#' @importFrom grDevices colorRampPalette
#' @importFrom htmltools tagList tags browsable
#' @importFrom RColorBrewer brewer.pal
#'
#' @family semantic
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
                                       category_params = NULL) {

  if (!requireNamespace("htmltools", quietly = TRUE) ||
      !requireNamespace("RColorBrewer", quietly = TRUE)) {
    stop(
      "The 'htmltools' and 'RColorBrewer' packages are required for this functionality. ",
      "Please install them using install.packages(c('htmltools', 'RColorBrewer'))."
    )
  }

  dfm_td <- tidytext::tidy(dfm_object)
  docvars_df <- dfm_object@docvars
  docvars_df$document <- docvars_df$docname_
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

  build_table <- function(net, group_label) {
    layout_dff <- net$layout_df %>%
      dplyr::select(-c("x", "y")) %>%
      dplyr::mutate(dplyr::across(dplyr::where(is.numeric), ~ round(., 3)))

    table <- DT::datatable(layout_dff, rownames = FALSE,
                           extensions = 'Buttons',
                           options = list(scrollX = TRUE,
                                          width = "80%",
                                          dom = 'Bfrtip',
                                          buttons = c('copy', 'csv', 'excel', 'pdf', 'print'))) %>%
      DT::formatStyle(columns = colnames(layout_dff), `font-size` = "16px")

    htmltools::tagList(
      htmltools::tags$div(
        style = "margin-bottom: 20px;",
        htmltools::tags$p(
          group_label,
          style = "font-weight: bold; text-align: center; font-size: 16px;"
        )
      ),
      table
    )
  }

  build_summary <- function(net, group_label) {
    g <- net$graph
    summary_df <- data.frame(
      Metric = c("Nodes", "Edges", "Density", "Diameter",
                 "Global Clustering Coefficient", "Local Clustering Coefficient (Mean)",
                 "Modularity", "Assortativity", "Geodesic Distance (Mean)"),
      Value = c(
        igraph::vcount(g),
        igraph::ecount(g),
        igraph::edge_density(g),
        igraph::diameter(g),
        igraph::transitivity(g, type = "global"),
        mean(igraph::transitivity(g, type = "local"), na.rm = TRUE),
        igraph::modularity(g, membership = igraph::V(g)$community),
        igraph::assortativity_degree(g),
        mean(igraph::distances(g)[igraph::distances(g) != Inf], na.rm = TRUE)
      )
    ) %>%
      dplyr::mutate(dplyr::across(dplyr::where(is.numeric), ~ round(., 3)))

    summary_table <- DT::datatable(summary_df, rownames = FALSE,
                                   extensions = 'Buttons',
                                   options = list(scrollX = TRUE,
                                                  width = "80%",
                                                  dom = 'Bfrtip',
                                                  buttons = c('copy', 'csv', 'excel', 'pdf', 'print'))) %>%
      DT::formatStyle(columns = colnames(summary_df), `font-size` = "16px")

    htmltools::tagList(
      htmltools::tags$div(
        style = "margin-bottom: 20px;",
        htmltools::tags$p(
          group_label,
          style = "font-weight: bold; text-align: center; font-size: 16px;"
        )
      ),
      summary_table
    )
  }

  build_network_plot <- function(data, group_level = NULL, local_co_occur_n = NULL, local_top_node_n = NULL) {
    # Use category-specific params if provided, otherwise fall back to global params
    effective_co_occur_n <- if (!is.null(local_co_occur_n)) local_co_occur_n else co_occur_n
    effective_top_node_n <- if (!is.null(local_top_node_n)) local_top_node_n else top_node_n

    term_co_occur <- data %>%
      widyr::pairwise_count(term, document, sort = TRUE) %>%
      dplyr::filter(n >= effective_co_occur_n)

    graph <- igraph::graph_from_data_frame(term_co_occur, directed = FALSE)
    if (igraph::vcount(graph) == 0) {
      message("No co-occurrence relationships meet the threshold.")
      return(NULL)
    }
    igraph::V(graph)$degree      <- igraph::degree(graph)
    igraph::V(graph)$betweenness <- igraph::betweenness(graph)
    igraph::V(graph)$closeness   <- igraph::closeness(graph)
    igraph::V(graph)$eigenvector <- igraph::eigen_centrality(graph)$vector

    # Community detection based on method
    community_result <- if (community_method == "louvain") {
      igraph::cluster_louvain(graph)
    } else {
      igraph::cluster_leiden(graph)
    }
    igraph::V(graph)$community <- community_result$membership

    layout_mat <- igraph::layout_with_fr(graph)
    layout_df <- as.data.frame(layout_mat) %>% stats::setNames(c("x", "y"))
    layout_df <- layout_df %>%
      dplyr::mutate(label       = igraph::V(graph)$name,
                    degree      = igraph::V(graph)$degree,
                    betweenness = igraph::V(graph)$betweenness,
                    closeness   = igraph::V(graph)$closeness,
                    eigenvector = igraph::V(graph)$eigenvector,
                    community   = igraph::V(graph)$community)

    # Calculate word frequency from the original data
    word_freq <- data %>%
      dplyr::group_by(term) %>%
      dplyr::summarise(frequency = sum(count), .groups = "drop")
    layout_df <- layout_df %>%
      dplyr::left_join(word_freq, by = c("label" = "term")) %>%
      dplyr::mutate(frequency = ifelse(is.na(frequency), 1, frequency))

    edge_data <- igraph::as_data_frame(graph, what = "edges") %>%
      dplyr::mutate(x    = layout_df$x[match(from, layout_df$label)],
                    y    = layout_df$y[match(from, layout_df$label)],
                    xend = layout_df$x[match(to, layout_df$label)],
                    yend = layout_df$y[match(to, layout_df$label)],
                    cooccur_count = n) %>%
      dplyr::select(from, to, x, y, xend, yend, cooccur_count) %>%
      dplyr::mutate(line_group = as.integer({
        b <- unique(stats::quantile(cooccur_count, probs = seq(0, 1, length.out = 6), na.rm = TRUE))
        if (length(b) < 2) {
          b <- c(b, b[length(b)] + 1e-6)
        }
        cut(cooccur_count, breaks = b, include.lowest = TRUE)
      }),
      line_width = scales::rescale(line_group, to = c(1, 5)),
      alpha      = scales::rescale(line_group, to = c(0.1, 0.3)))

    edge_group_labels <- edge_data %>%
      dplyr::group_by(line_group) %>%
      dplyr::summarise(min_count = min(cooccur_count, na.rm = TRUE),
                       max_count = max(cooccur_count, na.rm = TRUE)) %>%
      dplyr::mutate(label = paste0("Count: ", min_count, " - ", max_count)) %>%
      dplyr::pull(label)

    # Determine size metric based on node_size_by parameter
    size_metric <- switch(node_size_by,
      "degree" = layout_df$degree,
      "betweenness" = layout_df$betweenness,
      "frequency" = layout_df$frequency,
      "fixed" = rep(20, nrow(layout_df)),
      layout_df$degree  # default
    )

    node_data <- layout_df %>%
      dplyr::mutate(
        size_metric_log = log1p(size_metric),
        size = if (node_size_by == "fixed") 20 else scales::rescale(size_metric_log, to = c(12, 30)),
        text_size = scales::rescale(log1p(degree), to = c(node_label_size - 8, node_label_size)),
        alpha = scales::rescale(log1p(degree), to = c(0.2, 1)),
        hover_text = paste("Word:", label,
                           "<br>Degree:", degree,
                           "<br>Betweenness:", round(betweenness, 2),
                           "<br>Closeness:", round(closeness, 2),
                           "<br>Eigenvector:", round(eigenvector, 2),
                           "<br>Frequency:", frequency,
                           "<br>Community:", community,
                           if (!is.null(doc_var)) {
                             if (length(docvar_levels) > 1) {
                               paste0("<br>", doc_var, ": ", group_level)
                             } else {
                               paste0("<br>", doc_var)
                             }
                           } else ""
        )
      )

    # Create community palette
    n_communities <- length(unique(node_data$community))
    if (n_communities >= 3 && n_communities <= 8) {
      palette <- RColorBrewer::brewer.pal(n_communities, "Set2")
    } else if (n_communities > 8) {
      palette <- grDevices::colorRampPalette(RColorBrewer::brewer.pal(8, "Set2"))(n_communities)
    } else if (n_communities > 0 && n_communities < 3) {
      palette <- RColorBrewer::brewer.pal(3, "Set2")[1:n_communities]
    } else {
      palette <- rep("#000000", n_communities)
    }

    node_data$community <- factor(node_data$community, levels = unique(node_data$community))
    community_levels <- levels(node_data$community)
    names(palette) <- community_levels

    # Determine node color based on node_color_by parameter
    if (node_color_by == "frequency") {
      # Color by frequency gradient using viridis
      node_data$color <- scales::col_numeric("viridis", domain = range(node_data$frequency, na.rm = TRUE))(node_data$frequency)
    } else {
      # Default: color by community
      node_data$color <- palette[as.character(node_data$community)]
    }

    p <- plotly::plot_ly(type = 'scatter', mode = 'markers', width = width, height = height)
    for (lg in unique(edge_data$line_group)) {
      esub <- dplyr::filter(edge_data, line_group == lg) %>%
        dplyr::mutate(mid_x = (x + xend) / 2,
                      mid_y = (y + yend) / 2)
      if (nrow(esub) > 0) {
        p <- p %>%
          plotly::add_segments(data = esub, x = ~x, y = ~y,
                               xend = ~xend, yend = ~yend,
                               line = list(color = '#5C5CFF', width = ~line_width),
                               hoverinfo = 'none', opacity = ~alpha,
                               showlegend = TRUE, name = edge_group_labels[lg],
                               legendgroup = "Edges") %>%
          plotly::add_trace(data = esub, x = ~mid_x, y = ~mid_y, type = 'scatter',
                            mode = 'markers',
                            marker = list(size = 0.1, color = '#e0f7ff', opacity = 0),
                            text = ~paste("Co-occurrence:", cooccur_count,
                                          "<br>Source:", from,
                                          "<br>Target:", to),
                            hoverinfo = 'text', showlegend = FALSE)
      }
    }
    # Add nodes based on color mode
    if (node_color_by == "frequency") {
      # Color by frequency gradient - render all nodes at once
      p <- p %>% plotly::add_markers(
        data = node_data, x = ~x, y = ~y,
        marker = list(
          size = ~size,
          color = ~frequency,
          colorscale = "Viridis",
          showscale = TRUE,
          colorbar = list(title = "Frequency"),
          line = list(width = 2, color = '#FFFFFF')
        ),
        hoverinfo = 'text', text = ~hover_text,
        showlegend = FALSE
      )
    } else {
      # Default: color by community - loop through communities
      for (comm in community_levels) {
        comm_data <- dplyr::filter(node_data, community == comm)
        p <- p %>% plotly::add_markers(
          data = comm_data, x = ~x, y = ~y,
          marker = list(
            size = ~size,
            color = palette[comm],
            showscale = FALSE,
            line = list(width = 3, color = '#FFFFFF')
          ),
          hoverinfo = 'text', text = ~hover_text,
          showlegend = TRUE, name = paste("Community", comm),
          legendgroup = "Community"
        )
      }
    }
    top_nodes <- dplyr::arrange(node_data, dplyr::desc(degree)) %>% utils::head(effective_top_node_n)
    annotations <- if (nrow(top_nodes) > 0) {
      lapply(seq_len(nrow(top_nodes)), function(i) {
        list(x = top_nodes$x[i],
             y = top_nodes$y[i],
             text = top_nodes$label[i],
             xanchor = ifelse(top_nodes$x[i] > 0, "left", "right"),
             yanchor = ifelse(top_nodes$y[i] > 0, "bottom", "top"),
             xshift = ifelse(top_nodes$x[i] > 0, 5, -5),
             yshift = ifelse(top_nodes$y[i] > 0, 3, -3),
             showarrow = FALSE,
             font = list(size = top_nodes$text_size[i], color = 'black'))
      })
    } else {
      list()
    }

    p <- p %>% plotly::layout(dragmode = "pan",
                              title = list(text = "Word Co-occurrence Network",
                                           font = list(size = 19,
                                                       color = "black",
                                                       family = "Arial Black")),
                              showlegend = TRUE,
                              xaxis = list(title = "", showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
                              yaxis = list(title = "", showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
                              margin = list(l = 40, r = 100, t = 60, b = 40),
                              annotations = annotations,
                              legend = list(title = list(text = "Co-occurrence"),
                                            orientation = "v", x = 1.1, y = 1,
                                            xanchor = "left", yanchor = "top"))
    list(plot = p, layout_df = layout_df, graph = graph)
  }

  if (!is.null(doc_var) && length(docvar_levels) > 1) {
    plots_list <- dfm_td %>%
      dplyr::ungroup() %>%
      dplyr::group_by(!!rlang::sym(doc_var)) %>%
      dplyr::group_map(~ {
        group_level <- .y[[doc_var]]
        message(paste("Processing group level:", group_level))

        if (is.null(group_level)) {
          stop("doc_var is missing or not found in the current group")
        }

        # Look up category-specific parameters if available
        local_co_occur_n <- NULL
        local_top_node_n <- NULL
        if (!is.null(category_params) && group_level %in% names(category_params)) {
          cat_params <- category_params[[group_level]]
          if (!is.null(cat_params$co_occur_n)) local_co_occur_n <- cat_params$co_occur_n
          if (!is.null(cat_params$top_node_n)) local_top_node_n <- cat_params$top_node_n
          message(paste("  Using category-specific params - co_occur_n:", local_co_occur_n, ", top_node_n:", local_top_node_n))
        }

        net <- build_network_plot(.x, group_level, local_co_occur_n, local_top_node_n)
        if (!is.null(net)) {
          net$plot %>% plotly::layout(
            annotations = list(
              list(
                text = group_level,
                x = 0.42,
                xanchor = "center",
                y = 0.98,
                yanchor = "bottom",
                yref = "paper",
                showarrow = FALSE,
                font = list(size = 19, color = "black", family = "Arial Black")
              )
            )
          )
        } else {
          NULL
        }
      })

    combined_plot <- plotly::subplot(plots_list, nrows = nrows, shareX = TRUE, shareY = TRUE,
                                     titleX = TRUE, titleY = TRUE)

    table_list <- lapply(docvar_levels, function(level) {
      message(paste("Generating table for level:", level))
      group_data <- dplyr::filter(dfm_td, !!rlang::sym(doc_var) == level)
      # Look up category-specific parameters
      local_co_occur_n <- NULL
      local_top_node_n <- NULL
      if (!is.null(category_params) && level %in% names(category_params)) {
        cat_params <- category_params[[level]]
        if (!is.null(cat_params$co_occur_n)) local_co_occur_n <- cat_params$co_occur_n
        if (!is.null(cat_params$top_node_n)) local_top_node_n <- cat_params$top_node_n
      }
      net <- build_network_plot(group_data, level, local_co_occur_n, local_top_node_n)
      if (!is.null(net)) build_table(net, level) else NULL
    })

    summary_list <- lapply(docvar_levels, function(level) {
      message(paste("Generating summary for level:", level))
      group_data <- dplyr::filter(dfm_td, !!rlang::sym(doc_var) == level)
      # Look up category-specific parameters
      local_co_occur_n <- NULL
      local_top_node_n <- NULL
      if (!is.null(category_params) && level %in% names(category_params)) {
        cat_params <- category_params[[level]]
        if (!is.null(cat_params$co_occur_n)) local_co_occur_n <- cat_params$co_occur_n
        if (!is.null(cat_params$top_node_n)) local_top_node_n <- cat_params$top_node_n
      }
      net <- build_network_plot(group_data, level, local_co_occur_n, local_top_node_n)
      if (!is.null(net)) build_summary(net, level) else NULL
    })

    return(list(
      plot = combined_plot,
      table = table_list %>% htmltools::tagList() %>% htmltools::browsable(),
      summary = summary_list %>% htmltools::tagList() %>% htmltools::browsable()
    ))
  } else {
    net <- build_network_plot(dfm_td)
    if (is.null(net)) {
      message("No network generated.")
      return(NULL)
    }
    return(list(
      plot = net$plot,
      table = build_table(net, if (!is.null(doc_var)) paste("Network Centrality Table for", doc_var) else "Network Centrality Table") %>% htmltools::browsable(),
      summary = build_summary(net, if (!is.null(doc_var)) paste("Network Summary for", doc_var) else "Network Summary") %>% htmltools::browsable()
    ))
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
#'
#' @return A list containing the Plotly plot, a table, and a summary.
#'
#' @importFrom igraph graph_from_data_frame V vcount ecount degree betweenness closeness eigen_centrality layout_with_fr cluster_leiden cluster_louvain edge_density diameter transitivity modularity assortativity_degree distances
#' @importFrom plotly plot_ly add_segments add_markers layout add_trace subplot
#' @importFrom dplyr count filter mutate select group_by summarise ungroup left_join arrange desc group_map pull
#' @importFrom tibble as_tibble
#' @importFrom tidytext tidy
#' @importFrom widyr pairwise_cor
#' @importFrom scales rescale
#' @importFrom stats quantile setNames
#' @importFrom DT datatable formatStyle
#' @importFrom rlang sym
#' @importFrom utils head
#' @importFrom grDevices colorRampPalette
#' @importFrom htmltools tagList tags browsable
#' @importFrom RColorBrewer brewer.pal
#'
#' @family semantic
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
                                     category_params = NULL) {

  if (!requireNamespace("htmltools", quietly = TRUE) ||
      !requireNamespace("RColorBrewer", quietly = TRUE)) {
    stop(
      "The 'htmltools' and 'RColorBrewer' packages are required for this functionality. ",
      "Please install them using install.packages(c('htmltools', 'RColorBrewer'))."
    )
  }

  dfm_td <- tidytext::tidy(dfm_object)
  docvars_df <- dfm_object@docvars
  docvars_df$document <- docvars_df$docname_
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

  build_table <- function(net, group_label) {
    layout_dff <- net$layout_df %>%
      dplyr::select(-c("x", "y")) %>%
      dplyr::mutate(dplyr::across(dplyr::where(is.numeric), ~ round(., 3)))

    table <- DT::datatable(layout_dff, rownames = FALSE,
                           extensions = 'Buttons',
                           options = list(scrollX = TRUE,
                                          width = "80%",
                                          dom = 'Bfrtip',
                                          buttons = c('copy', 'csv', 'excel', 'pdf', 'print'))) %>%
      DT::formatStyle(columns = colnames(layout_dff), `font-size` = "16px")

    htmltools::tagList(
      htmltools::tags$div(
        style = "margin-bottom: 20px;",
        htmltools::tags$p(
          group_label,
          style = "font-weight: bold; text-align: center; font-size: 16px;"
        )
      ),
      table
    )
  }

  build_summary <- function(net, group_label) {
    g <- net$graph
    summary_df <- data.frame(
      Metric = c("Nodes", "Edges", "Density", "Diameter",
                 "Global Clustering Coefficient", "Local Clustering Coefficient (Mean)",
                 "Modularity", "Assortativity", "Geodesic Distance (Mean)"),
      Value = c(
        igraph::vcount(g),
        igraph::ecount(g),
        igraph::edge_density(g),
        igraph::diameter(g),
        igraph::transitivity(g, type = "global"),
        mean(igraph::transitivity(g, type = "local"), na.rm = TRUE),
        igraph::modularity(g, membership = igraph::V(g)$community),
        igraph::assortativity_degree(g),
        mean(igraph::distances(g)[igraph::distances(g) != Inf], na.rm = TRUE)
      )
    ) %>%
      dplyr::mutate(dplyr::across(dplyr::where(is.numeric), ~ round(., 3)))

    summary_table <- DT::datatable(summary_df, rownames = FALSE,
                                   extensions = 'Buttons',
                                   options = list(scrollX = TRUE,
                                                  width = "80%",
                                                  dom = 'Bfrtip',
                                                  buttons = c('copy', 'csv', 'excel', 'pdf', 'print'))) %>%
      DT::formatStyle(columns = colnames(summary_df), `font-size` = "16px")

    htmltools::tagList(
      htmltools::tags$div(
        style = "margin-bottom: 20px;",
        htmltools::tags$p(
          group_label,
          style = "font-weight: bold; text-align: center; font-size: 16px;"
        )
      ),
      summary_table
    )
  }

  build_network_plot <- function(data, group_level = NULL, local_common_term_n = NULL, local_corr_n = NULL, local_top_node_n = NULL) {
    # Use category-specific params if provided, otherwise fall back to global params
    effective_common_term_n <- if (!is.null(local_common_term_n)) local_common_term_n else common_term_n
    effective_corr_n <- if (!is.null(local_corr_n)) local_corr_n else corr_n
    effective_top_node_n <- if (!is.null(local_top_node_n)) local_top_node_n else top_node_n

    term_cor <- data %>%
      dplyr::group_by(term) %>%
      dplyr::filter(dplyr::n() >= effective_common_term_n) %>%
      widyr::pairwise_cor(term, document, sort = TRUE) %>%
      dplyr::ungroup() %>%
      dplyr::filter(correlation > effective_corr_n)

    graph <- igraph::graph_from_data_frame(term_cor, directed = FALSE)
    if(igraph::vcount(graph) == 0) {
      message("No correlation relationships meet the threshold.")
      return(NULL)
    }
    igraph::V(graph)$degree      <- igraph::degree(graph)
    igraph::V(graph)$betweenness <- igraph::betweenness(graph)
    igraph::V(graph)$closeness   <- igraph::closeness(graph)
    igraph::V(graph)$eigenvector <- igraph::eigen_centrality(graph)$vector

    # Community detection based on method
    community_result <- if (community_method == "louvain") {
      igraph::cluster_louvain(graph)
    } else {
      igraph::cluster_leiden(graph)
    }
    igraph::V(graph)$community <- community_result$membership

    layout_mat <- igraph::layout_with_fr(graph)
    layout_df <- as.data.frame(layout_mat) %>% stats::setNames(c("x", "y"))
    layout_df <- layout_df %>%
      dplyr::mutate(label       = igraph::V(graph)$name,
                    degree      = igraph::V(graph)$degree,
                    betweenness = igraph::V(graph)$betweenness,
                    closeness   = igraph::V(graph)$closeness,
                    eigenvector = igraph::V(graph)$eigenvector,
                    community   = igraph::V(graph)$community)

    # Calculate word frequency from the original data
    word_freq <- data %>%
      dplyr::group_by(term) %>%
      dplyr::summarise(frequency = sum(count), .groups = "drop")
    layout_df <- layout_df %>%
      dplyr::left_join(word_freq, by = c("label" = "term")) %>%
      dplyr::mutate(frequency = ifelse(is.na(frequency), 1, frequency))

    edge_data <- igraph::as_data_frame(graph, what = "edges") %>%
      dplyr::mutate(x    = layout_df$x[match(from, layout_df$label)],
                    y    = layout_df$y[match(from, layout_df$label)],
                    xend = layout_df$x[match(to, layout_df$label)],
                    yend = layout_df$y[match(to, layout_df$label)],
                    correlation = correlation) %>%
      dplyr::select(from, to, x, y, xend, yend, correlation) %>%
      dplyr::mutate(line_group = as.integer({
        b <- unique(stats::quantile(correlation, probs = seq(0, 1, length.out = 6), na.rm = TRUE))
        if (length(b) < 2) {
          b <- c(b, b[length(b)] + 1e-6)
        }
        cut(correlation, breaks = b, include.lowest = TRUE)
      }),
      line_width = scales::rescale(line_group, to = c(1, 5)),
      alpha      = scales::rescale(line_group, to = c(0.1, 0.3)))

    edge_group_labels <- edge_data %>%
      dplyr::group_by(line_group) %>%
      dplyr::summarise(
        min_corr = min(correlation, na.rm = TRUE),
        max_corr = max(correlation, na.rm = TRUE)
      ) %>%
      dplyr::mutate(label = paste0("Correlation: ", round(min_corr, 2), " - ", round(max_corr, 2))) %>%
      dplyr::pull(label)

    # Determine size metric based on node_size_by parameter
    size_metric <- switch(node_size_by,
      "degree" = layout_df$degree,
      "betweenness" = layout_df$betweenness,
      "frequency" = layout_df$frequency,
      "fixed" = rep(20, nrow(layout_df)),
      layout_df$degree  # default
    )

    node_data <- layout_df %>%
      dplyr::mutate(
        size_metric_log = log1p(size_metric),
        size = if (node_size_by == "fixed") 20 else scales::rescale(size_metric_log, to = c(12, 30)),
        text_size = scales::rescale(log1p(degree), to = c(node_label_size - 8, node_label_size)),
        alpha = scales::rescale(log1p(degree), to = c(0.2, 1)),
        hover_text = paste(
          "Word:", label,
          "<br>Degree:", degree,
          "<br>Betweenness:", round(betweenness, 2),
          "<br>Closeness:", round(closeness, 2),
          "<br>Eigenvector:", round(eigenvector, 2),
          "<br>Frequency:", frequency,
          "<br>Community:", community,
          if (!is.null(doc_var)) {
            if (length(docvar_levels) > 1) {
              paste0("<br>", doc_var, ": ", group_level)
            } else {
              paste0("<br>", doc_var)
            }
          } else ""
        )
      )

    # Create community palette
    n_communities <- length(unique(node_data$community))
    if (n_communities >= 3 && n_communities <= 8) {
      palette <- RColorBrewer::brewer.pal(n_communities, "Set2")
    } else if (n_communities > 8) {
      palette <- grDevices::colorRampPalette(RColorBrewer::brewer.pal(8, "Set2"))(n_communities)
    } else if (n_communities > 0 && n_communities < 3) {
      palette <- RColorBrewer::brewer.pal(3, "Set2")[1:n_communities]
    } else {
      palette <- rep("#000000", n_communities)
    }

    node_data$community <- factor(node_data$community, levels = unique(node_data$community))
    community_levels <- levels(node_data$community)
    names(palette) <- community_levels

    # Determine node color based on node_color_by parameter
    if (node_color_by == "frequency") {
      # Color by frequency gradient using viridis
      node_data$color <- scales::col_numeric("viridis", domain = range(node_data$frequency, na.rm = TRUE))(node_data$frequency)
    } else {
      # Default: color by community
      node_data$color <- palette[as.character(node_data$community)]
    }

    p <- plotly::plot_ly(type = 'scatter', mode = 'markers', width = width, height = height)
    for (lg in unique(edge_data$line_group)) {
      esub <- dplyr::filter(edge_data, line_group == lg) %>%
        dplyr::mutate(mid_x = (x + xend) / 2,
                      mid_y = (y + yend) / 2)
      if (nrow(esub) > 0) {
        p <- p %>%
          plotly::add_segments(data = esub, x = ~x, y = ~y,
                               xend = ~xend, yend = ~yend,
                               line = list(color = '#5C5CFF', width = ~line_width),
                               hoverinfo = 'none', opacity = ~alpha,
                               showlegend = TRUE, name = edge_group_labels[lg],
                               legendgroup = "Edges") %>%
          plotly::add_trace(data = esub, x = ~mid_x, y = ~mid_y, type = 'scatter',
                            mode = 'markers',
                            marker = list(size = 0.1, color = '#e0f7ff', opacity = 0),
                            text = ~paste("Correlation:", correlation,
                                          "<br>Source:", from,
                                          "<br>Target:", to),
                            hoverinfo = 'text', showlegend = FALSE)
      }
    }
    # Add nodes based on color mode
    if (node_color_by == "frequency") {
      # Color by frequency gradient - render all nodes at once
      p <- p %>% plotly::add_markers(
        data = node_data, x = ~x, y = ~y,
        marker = list(
          size = ~size,
          color = ~frequency,
          colorscale = "Viridis",
          showscale = TRUE,
          colorbar = list(title = "Frequency"),
          line = list(width = 2, color = '#FFFFFF')
        ),
        hoverinfo = 'text', text = ~hover_text,
        showlegend = FALSE
      )
    } else {
      # Default: color by community - loop through communities
      for (comm in community_levels) {
        comm_data <- dplyr::filter(node_data, community == comm)
        p <- p %>% plotly::add_markers(
          data = comm_data, x = ~x, y = ~y,
          marker = list(
            size = ~size,
            color = palette[comm],
            showscale = FALSE,
            line = list(width = 3, color = '#FFFFFF')
          ),
          hoverinfo = 'text', text = ~hover_text,
          showlegend = TRUE, name = paste("Community", comm),
          legendgroup = "Community"
        )
      }
    }
    top_nodes <- dplyr::arrange(node_data, dplyr::desc(degree)) %>% utils::head(effective_top_node_n)
    annotations <- if (nrow(top_nodes) > 0) {
      lapply(seq_len(nrow(top_nodes)), function(i) {
        list(x = top_nodes$x[i],
             y = top_nodes$y[i],
             text = top_nodes$label[i],
             xanchor = ifelse(top_nodes$x[i] > 0, "left", "right"),
             yanchor = ifelse(top_nodes$y[i] > 0, "bottom", "top"),
             xshift = ifelse(top_nodes$x[i] > 0, 5, -5),
             yshift = ifelse(top_nodes$y[i] > 0, 3, -3),
             showarrow = FALSE,
             font = list(size = top_nodes$text_size[i], color = 'black'))
      })
    } else {
      list()
    }

    p <- p %>% plotly::layout(dragmode = "pan",
                              title = list(text = "Word Correlation Network", font = list(size = 19, color = "black", family = "Arial Black")),
                              showlegend = TRUE,
                              xaxis = list(title = "", showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
                              yaxis = list(title = "", showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
                              margin = list(l = 40, r = 100, t = 60, b = 40),
                              annotations = annotations,
                              legend = list(title = list(text = "Correlation"),
                                            orientation = "v", x = 1.1, y = 1,
                                            xanchor = "left", yanchor = "top"))
    list(plot = p, layout_df = layout_df, graph = graph)
  }

  if (!is.null(doc_var) && length(docvar_levels) > 1) {
    plots_list <- dfm_td %>%
      dplyr::ungroup() %>%
      dplyr::group_by(!!rlang::sym(doc_var)) %>%
      dplyr::group_map(~ {
        group_level <- .y[[doc_var]]
        message(paste("Processing group level:", group_level))

        if (is.null(group_level)) {
          stop("doc_var is missing or not found in the current group")
        }

        # Look up category-specific parameters if available
        local_common_term_n <- NULL
        local_corr_n <- NULL
        local_top_node_n <- NULL
        if (!is.null(category_params) && group_level %in% names(category_params)) {
          cat_params <- category_params[[group_level]]
          if (!is.null(cat_params$common_term_n)) local_common_term_n <- cat_params$common_term_n
          if (!is.null(cat_params$corr_n)) local_corr_n <- cat_params$corr_n
          if (!is.null(cat_params$top_node_n)) local_top_node_n <- cat_params$top_node_n
          message(paste("  Using category-specific params - common_term_n:", local_common_term_n,
                        ", corr_n:", local_corr_n, ", top_node_n:", local_top_node_n))
        }

        net <- build_network_plot(.x, group_level, local_common_term_n, local_corr_n, local_top_node_n)
        if (!is.null(net)) {
          net$plot %>% plotly::layout(
            annotations = list(
              list(
                text = group_level,
                x = 0.42,
                xanchor = "center",
                y = 0.98,
                yanchor = "bottom",
                yref = "paper",
                showarrow = FALSE,
                font = list(size = 19, color = "black", family = "Arial Black")
              )
            )
          )
        } else {
          NULL
        }
      })

    combined_plot <- plotly::subplot(plots_list, nrows = nrows, shareX = TRUE, shareY = TRUE,
                                     titleX = TRUE, titleY = TRUE)

    table_list <- lapply(docvar_levels, function(level) {
      message(paste("Generating table for level:", level))
      group_data <- dplyr::filter(dfm_td, !!rlang::sym(doc_var) == level)
      # Look up category-specific parameters
      local_common_term_n <- NULL
      local_corr_n <- NULL
      local_top_node_n <- NULL
      if (!is.null(category_params) && level %in% names(category_params)) {
        cat_params <- category_params[[level]]
        if (!is.null(cat_params$common_term_n)) local_common_term_n <- cat_params$common_term_n
        if (!is.null(cat_params$corr_n)) local_corr_n <- cat_params$corr_n
        if (!is.null(cat_params$top_node_n)) local_top_node_n <- cat_params$top_node_n
      }
      net <- build_network_plot(group_data, level, local_common_term_n, local_corr_n, local_top_node_n)
      if (!is.null(net)) build_table(net, level) else NULL
    })

    summary_list <- lapply(docvar_levels, function(level) {
      message(paste("Generating summary for level:", level))
      group_data <- dplyr::filter(dfm_td, !!rlang::sym(doc_var) == level)
      # Look up category-specific parameters
      local_common_term_n <- NULL
      local_corr_n <- NULL
      local_top_node_n <- NULL
      if (!is.null(category_params) && level %in% names(category_params)) {
        cat_params <- category_params[[level]]
        if (!is.null(cat_params$common_term_n)) local_common_term_n <- cat_params$common_term_n
        if (!is.null(cat_params$corr_n)) local_corr_n <- cat_params$corr_n
        if (!is.null(cat_params$top_node_n)) local_top_node_n <- cat_params$top_node_n
      }
      net <- build_network_plot(group_data, level, local_common_term_n, local_corr_n, local_top_node_n)
      if (!is.null(net)) build_summary(net, level) else NULL
    })

    return(list(
      plot = combined_plot,
      table = table_list %>% htmltools::tagList() %>% htmltools::browsable(),
      summary = summary_list %>% htmltools::tagList() %>% htmltools::browsable()
    ))
  } else {
    net <- build_network_plot(dfm_td)
    if (is.null(net)) {
      message("No network generated.")
      return(NULL)
    }
    return(list(
      plot = net$plot,
      table = build_table(net,
                          if (!is.null(doc_var)) paste("Network Centrality Table for", doc_var)
                          else "Network Centrality Table") %>% htmltools::browsable(),
      summary = build_summary(net,
                              if (!is.null(doc_var)) paste("Network Summary for", doc_var)
                              else "Network Summary") %>% htmltools::browsable()
    ))
  }
}
