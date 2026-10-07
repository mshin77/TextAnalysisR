blob_matrix <- function(seed = 6) {
  withr::with_seed(seed, rbind(
    matrix(rnorm(60, 0, 0.3), 20),
    matrix(rnorm(60, 4, 0.3), 20)
  ))
}

test_that("silhouette excludes noise points", {
  skip_if_not_installed("cluster")
  coords <- rbind(blob_matrix(), withr::with_seed(7, matrix(runif(30, -10, 15), 10)))
  labels <- c(rep(1, 20), rep(2, 20), rep(0, 10))
  scored <- labels > 0
  expected <- mean(cluster::silhouette(labels[scored], stats::dist(coords[scored, ]))[, 3])

  expect_equal(TextAnalysisR:::.silhouette_no_noise(labels, coords), expected)
  expect_equal(TextAnalysisR:::.silhouette_no_noise(labels, stats::dist(coords)), expected)
  expect_true(is.na(TextAnalysisR:::.silhouette_no_noise(c(rep(1, 20), rep(0, 5)), coords[1:25, ])))
  expect_equal(calculate_clustering_metrics(labels, coords)$silhouette, expected)
})

test_that("cluster_embeddings sets aside empty documents with label 0", {
  m <- rbind(blob_matrix(), matrix(0, 4, 3))
  res <- cluster_embeddings(m, method = "kmeans", n_clusters = 2, verbose = FALSE)

  expect_length(res$clusters, 44)
  expect_equal(res$clusters[41:44], rep(0L, 4))
  expect_equal(res$empty_documents, 41:44)
  expect_setequal(unique(res$clusters[1:40]), 1:2)
})

test_that("cluster_embeddings recovers direction-only clusters with unit-length rows", {
  dirs <- withr::with_seed(1, matrix(rnorm(60), 3))
  truth <- rep(1:3, each = 40)
  x <- withr::with_seed(2, t(sapply(truth, function(k) dirs[k, ] + rnorm(20, sd = 0.3))) * rexp(120, 0.2))
  res <- cluster_embeddings(x, method = "kmeans", n_clusters = 3, verbose = FALSE)
  expect_equal(max(table(res$clusters, truth)) * 3, 120)
})

test_that("automatic K does not fail on three documents", {
  tiny <- withr::with_seed(3, matrix(rnorm(12), 3))
  expect_equal(cluster_embeddings(tiny, method = "kmeans", verbose = FALSE)$n_clusters, 2)
  expect_equal(cluster_embeddings(tiny, method = "hierarchical", verbose = FALSE)$n_clusters, 2)
})

test_that("UMAP neighbors are capped at n - 1 and never below 2", {
  skip_if_not_installed("umap")
  x <- withr::with_seed(4, matrix(rnorm(400), 40))
  wide <- suppressMessages(reduce_dimensions(x, method = "UMAP", n_components = 2, umap_neighbors = 30, verbose = FALSE))
  small <- suppressMessages(reduce_dimensions(x[1:10, ], method = "UMAP", n_components = 2, umap_neighbors = 50, verbose = FALSE))
  expect_equal(wide$umap_params$n_neighbors, 30L)
  expect_equal(small$umap_params$n_neighbors, 9L)
})

test_that("edge style bins stay within five levels and the contrast floor", {
  bins <- TextAnalysisR:::.edge_style_bins(c(rep(2, 50), rep(3, 30), 4:10, 500))
  expect_lte(dplyr::n_distinct(bins$line_width), 5)
  expect_gte(min(bins$alpha), 0.75)
  expect_equal(nrow(TextAnalysisR:::.edge_style_bins(rep(3, 5))), 5)
})

test_that("network summary reports mean geodesic over node pairs", {
  g <- igraph::make_ring(5)
  pairs <- igraph::distances(g)[upper.tri(diag(5))]
  expect_equal(mean(Filter(is.finite, pairs)), 1.5)
})

test_that("correlation network ignores negative thresholds and labels harmonic closeness", {
  texts <- rep(c("reading fluency intervention students", "math word problems students",
                 "reading comprehension intervention", "math fluency problems"), 10)
  dfm <- quanteda::dfm(quanteda::tokens(texts))
  net <- suppressMessages(suppressWarnings(
    word_correlation_network(dfm, common_term_n = 1, corr_n = -0.5, top_node_n = 10)
  ))
  expect_false(is.null(net))
  expect_true(all(grepl("Harmonic closeness", net$top_nodes$hover_text)))
})

test_that("calculate_metrics fills silhouette and modularity for supplied labels", {
  skip_if_not_installed("cluster")
  a <- withr::with_seed(3, matrix(rexp(200), 10)); a[, 1] <- a[, 1] + 8
  b <- withr::with_seed(4, matrix(rexp(200), 10)); b[, 20] <- b[, 20] + 8
  sim <- calculate_cosine_similarity(rbind(a, b))
  m <- calculate_metrics(sim, labels = rep(c("a", "b"), each = 10))
  expect_gt(m$silhouette_score, 0.5)
  expect_gt(m$modularity, 0.3)
})

test_that("sentiment_lexicon_analysis returns token counts for every document", {
  skip_if_not_installed("textdata")
  skip_if(inherits(try(tidytext::get_sentiments("nrc"), silent = TRUE), "try-error"), "NRC lexicon not downloaded")
  texts <- c("joy and trust and hope", "plain words here", "fear and anger rising today")
  dfm <- quanteda::dfm(quanteda::tokens(texts))
  res <- sentiment_lexicon_analysis(dfm, lexicon = "nrc")
  expect_equal(res$document_tokens$document, quanteda::docnames(dfm))
  expect_equal(res$document_tokens$n_tokens, as.numeric(quanteda::ntoken(dfm)))
})

test_that("co-occurrence closeness follows the normalized argument", {
  skip_if_not_installed("widyr")
  docs <- c("cat dog bird", "cat dog fish", "dog bird fish", "cat bird fish", "cat dog bird fish")
  d <- quanteda::dfm(quanteda::tokens(docs))
  raw <- suppressWarnings(suppressMessages(word_co_occurrence_network(d, co_occur_n = 1, top_node_n = 10)))
  norm <- suppressWarnings(suppressMessages(word_co_occurrence_network(d, co_occur_n = 1, top_node_n = 10, normalized = TRUE)))
  ratio <- raw$table$closeness / norm$table$closeness
  expect_true(all(abs(ratio - 3) < 0.01))
})
