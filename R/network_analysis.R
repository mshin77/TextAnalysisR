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
# Semantic network analysis functions for word co-occurrence and correlation
# NOTE: plot_cooccurrence_network and plot_correlation_network were removed
# as they were not used in the Shiny app. Use semantic_cooccurrence_network
# and semantic_correlation_network instead.



#' Compute Word Co-occurrence Network
#'
#' @description
#' Computes word co-occurrence networks with community detection and network metrics.
#' Supports multiple feature spaces: unigrams, n-grams, and embeddings.
#' Based on proven implementation for intuitive network visualization.
#'
#' @param dfm_object A quanteda document-feature matrix (dfm).
#' @param doc_var A document-level metadata variable for categories (default: NULL).
#' @param co_occur_n Minimum co-occurrence count (default: 10).
#' @param top_node_n Number of top nodes to display based on degree centrality (default: 30).
#' @param node_label_size Font size for node labels (default: 14).
#' @param pattern Regex pattern to filter specific words (default: NULL).
#' @param showlegend Whether to show community legend (default: TRUE).
#' @param seed Random seed for reproducible layout (default: NULL).
#' @param feature_type Feature space: "words", "ngrams", or "embeddings" (default: "words").
#' @param ngram_range N-gram size when feature_type = "ngrams" (default: 2).
#' @param texts Optional character vector of texts for n-gram creation (default: NULL).
#' @param embeddings Optional embedding matrix for embedding-based networks (default: NULL).
#' @param embedding_sim_threshold Similarity threshold for embedding-based networks (default: 0.5).
#' @param community_method Community detection method: "leiden" (default), "louvain", "label_prop", or "fast_greedy".
#'
#' @return A list containing plot, table, nodes, edges, and stats
#' @family network
#' @export
#'
semantic_cooccurrence_network <- function(dfm_object,
                                        doc_var = NULL,
                                        co_occur_n = 10,
                                        top_node_n = 30,
                                        node_label_size = 22,
                                        pattern = NULL,
                                        showlegend = TRUE,
                                        seed = NULL,
                                        feature_type = "words",
                                        ngram_range = 2,
                                        texts = NULL,
                                        embeddings = NULL,
                                        embedding_sim_threshold = 0.5,
                                        community_method = "leiden") {

  if (!is.null(seed)) set.seed(seed)

  if (feature_type == "ngrams" && !is.null(texts)) {
    if (!requireNamespace("quanteda", quietly = TRUE)) {
      stop("Package 'quanteda' is required for n-gram analysis.")
    }
    message("Creating ", ngram_range, "-gram co-occurrence network")
    corp <- quanteda::corpus(texts)
    toks <- quanteda::tokens(corp, remove_punct = TRUE, remove_symbols = TRUE)
    toks_ngrams <- quanteda::tokens_ngrams(toks, n = ngram_range)
    dfm_object <- quanteda::dfm(toks_ngrams)

    if (co_occur_n > 2) {
      original_threshold <- co_occur_n
      co_occur_n <- max(2, floor(co_occur_n * 0.3))
      message("Automatically adjusting co-occurrence threshold for ngrams: ",
              original_threshold, " -> ", co_occur_n)
    }
  } else if (feature_type == "embeddings" && !is.null(embeddings)) {
    message("Creating embedding-based document similarity network")

    # For embeddings, create document-document similarity network
    # Nodes = Documents, Edges = Similarity > threshold

    if (!is.matrix(embeddings) || nrow(embeddings) < 2) {
      message("Invalid embeddings matrix. Falling back to word-based network.")
    } else {
      # Compute cosine similarity matrix
      norm_embeddings <- embeddings / sqrt(rowSums(embeddings^2))
      sim_matrix <- tcrossprod(norm_embeddings)
      diag(sim_matrix) <- 0  # Remove self-similarity

      # Get document names
      doc_names <- if (!is.null(rownames(embeddings))) {
        rownames(embeddings)
      } else {
        paste0("doc_", seq_len(nrow(embeddings)))
      }
      rownames(sim_matrix) <- doc_names
      colnames(sim_matrix) <- doc_names

      # Create edges from similarity threshold
      edge_list <- which(sim_matrix > embedding_sim_threshold & upper.tri(sim_matrix), arr.ind = TRUE)

      if (nrow(edge_list) == 0) {
        message("No document pairs exceed similarity threshold. Try lowering the threshold.")
        return(NULL)
      }

      edge_df <- data.frame(
        from = doc_names[edge_list[, 1]],
        to = doc_names[edge_list[, 2]],
        weight = sim_matrix[edge_list],
        stringsAsFactors = FALSE
      )

      # Create graph
      graph <- igraph::graph_from_data_frame(edge_df, directed = FALSE)
      if (igraph::vcount(graph) == 0) return(NULL)

      # Compute centrality measures
      igraph::V(graph)$degree <- igraph::degree(graph)
      igraph::V(graph)$eigenvector <- igraph::eigen_centrality(graph)$vector

      # Community detection
      community_result <- switch(community_method,
        "louvain" = igraph::cluster_louvain(graph),
        "label_prop" = igraph::cluster_label_prop(graph),
        "fast_greedy" = igraph::cluster_fast_greedy(igraph::as.undirected(graph)),
        igraph::cluster_leiden(graph)
      )
      igraph::V(graph)$community <- community_result$membership

      # Create layout table
      layout_df <- data.frame(
        Document = igraph::V(graph)$name,
        Degree = igraph::V(graph)$degree,
        Eigenvector = round(igraph::V(graph)$eigenvector, 3),
        Community = igraph::V(graph)$community,
        stringsAsFactors = FALSE
      )

      # Limit nodes by degree
      node_degrees <- igraph::degree(graph)
      sorted_indices <- order(node_degrees, decreasing = TRUE)
      top_n <- min(top_node_n, length(sorted_indices))
      top_indices <- sorted_indices[1:top_n]
      top_nodes <- igraph::V(graph)$name[top_indices]
      graph <- igraph::induced_subgraph(graph, top_nodes)

      # Prepare visNetwork data
      communities <- unique(igraph::V(graph)$community)
      n_communities <- length(communities)
      colors <- colorRampPalette(c("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                                   "#9467bd", "#8c564b", "#e377c2"))(n_communities)
      community_colors <- setNames(colors, communities)

      nodes <- data.frame(
        id = igraph::V(graph)$name,
        label = igraph::V(graph)$name,
        value = igraph::V(graph)$degree,
        group = paste("Community", igraph::V(graph)$community),
        color = community_colors[as.character(igraph::V(graph)$community)],
        font.size = node_label_size,
        title = paste0("<b>", igraph::V(graph)$name, "</b><br>",
                       "Degree: ", igraph::V(graph)$degree, "<br>",
                       "Community: ", igraph::V(graph)$community),
        stringsAsFactors = FALSE
      )

      edge_data <- igraph::as_data_frame(graph, what = "edges")
      edges <- data.frame(
        from = edge_data$from,
        to = edge_data$to,
        value = if ("weight" %in% names(edge_data)) edge_data$weight else 1,
        color = "#888888",
        title = paste0("Similarity: ", round(edge_data$weight, 3)),
        stringsAsFactors = FALSE
      )

      # Create legend
      legend_labels <- lapply(seq_len(n_communities), function(i) {
        list(
          label = paste("Community", communities[i]),
          shape = "dot",
          color = colors[i],
          size = 15
        )
      })

      # Create plot
      plot <- visNetwork::visNetwork(nodes, edges) %>%
        visNetwork::visNodes(
          shape = "dot",
          scaling = list(min = 10, max = 40)
        ) %>%
        visNetwork::visEdges(
          smooth = FALSE,
          width = 1,
          color = list(color = "#888888", opacity = 0.6)
        ) %>%
        visNetwork::visLayout(randomSeed = seed %||% 2025) %>%
        visNetwork::visOptions(
          highlightNearest = TRUE,
          nodesIdSelection = TRUE,
          manipulation = FALSE,
          selectedBy = list(
            variable = "group",
            multiple = FALSE,
            style = "width: 150px; height: 26px;"
          )
        ) %>%
        visNetwork::visPhysics(enabled = FALSE) %>%
        visNetwork::visInteraction(
          hover = TRUE,
          tooltipDelay = 0,
          tooltipStay = 1000,
          zoomView = TRUE,
          dragView = TRUE
        ) %>%
        {if (showlegend) visNetwork::visLegend(.,
                                                addNodes = do.call(rbind, lapply(legend_labels, as.data.frame)),
                                                useGroups = FALSE,
                                                position = "right",
                                                width = 0.2,
                                                zoom = FALSE
        ) else .}

      # Compute stats
      safe_round <- function(x, digits = 4) {
        if (is.null(x) || is.na(x) || is.nan(x) || !is.numeric(x)) return(NA_real_)
        round(x, digits)
      }

      stats_list <- list(
        nodes = igraph::vcount(graph),
        edges = igraph::ecount(graph),
        density = safe_round(igraph::edge_density(graph)),
        diameter = tryCatch(igraph::diameter(graph), error = function(e) NA_integer_),
        global_clustering = safe_round(igraph::transitivity(graph, type = "global")),
        avg_local_clustering = safe_round(igraph::transitivity(graph, type = "average")),
        modularity = safe_round(igraph::modularity(community_result)),
        assortativity = safe_round(igraph::assortativity_degree(graph)),
        avg_path_length = safe_round(igraph::mean_distance(graph, directed = FALSE))
      )

      return(list(
        plot = plot,
        table = layout_df,
        nodes = nodes,
        edges = edges,
        stats = stats_list
      ))
    }
  }

  dfm_td <- tidytext::tidy(dfm_object)
  docvars_df <- dfm_object@docvars
  docvars_df$document <- docvars_df$docname_
  dfm_td <- dplyr::left_join(dfm_td, docvars_df, by = "document")

  if (!is.null(doc_var) && doc_var %in% colnames(dfm_td)) {
    available_levels <- unique(dfm_td[[doc_var]])
    available_levels <- available_levels[!is.na(available_levels)]
    message("Analyzing network for ", doc_var, ": ", paste(available_levels, collapse = ", "))
  }

  term_co_occur <- dfm_td %>%
    widyr::pairwise_count(term, document, sort = TRUE) %>%
    dplyr::filter(n >= co_occur_n)

  if (!is.null(pattern)) {
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

  igraph::V(graph)$degree <- igraph::degree(graph)
  igraph::V(graph)$eigenvector <- igraph::eigen_centrality(graph)$vector
  # Apply selected community detection method
  community_result <- switch(community_method,
    "louvain" = igraph::cluster_louvain(graph),
    "label_prop" = igraph::cluster_label_prop(graph),
    "fast_greedy" = igraph::cluster_fast_greedy(igraph::as.undirected(graph)),
    igraph::cluster_leiden(graph)  # default: leiden
  )
  igraph::V(graph)$community <- community_result$membership

  layout_df <- data.frame(
    Term = igraph::V(graph)$name,
    Degree = igraph::V(graph)$degree,
    Eigenvector = round(igraph::V(graph)$eigenvector, 3),
    Community = igraph::V(graph)$community,
    stringsAsFactors = FALSE
  )

  node_degrees <- igraph::degree(graph)
  sorted_indices <- order(node_degrees, decreasing = TRUE)
  top_n <- min(top_node_n, length(sorted_indices))
  top_nodes <- names(node_degrees)[sorted_indices[1:top_n]]

  nodes <- data.frame(
    id = igraph::V(graph)$name,
    label = ifelse(igraph::V(graph)$name %in% top_nodes, igraph::V(graph)$name, ""),
    group = igraph::V(graph)$community,
    value = igraph::V(graph)$degree,
    title = paste0(
      "<b style='color:black;'>", igraph::V(graph)$name, "</b><br>",
      "<span style='color:black;'>Degree: ", igraph::V(graph)$degree, "<br>",
      "Eigenvector: ", round(igraph::V(graph)$eigenvector, 2), "<br>",
      "Community: ", igraph::V(graph)$community, "</span>"
    ),
    stringsAsFactors = FALSE
  )

  edges <- igraph::as_data_frame(graph, what = "edges")
  edges$width <- scales::rescale(edges$n, to = c(1, 8))

  edge_color_base <- "#5C5CFF"
  edges$color <- mapply(function(n_val) {
    alpha_val <- scales::rescale(n_val, to = c(0.3, 1))
    scales::alpha(edge_color_base, alpha_val)
  }, edges$n)

  edges$title <- paste0(
    "<span style='color:black;'>Co-occurrence: ", edges$n,
    "<br>From: ", edges$from,
    "<br>To: ", edges$to, "</span>"
  )

  unique_communities <- sort(unique(nodes$group))
  community_map <- setNames(seq_along(unique_communities), unique_communities)
  nodes$group <- community_map[as.character(nodes$group)]

  n_communities <- length(unique(nodes$group))
  if (n_communities <= 8) {
    palette <- RColorBrewer::brewer.pal(max(3, n_communities), "Set2")
  } else {
    palette <- grDevices::colorRampPalette(RColorBrewer::brewer.pal(8, "Set2"))(n_communities)
  }
  community_colors <- setNames(palette, as.character(seq_len(n_communities)))
  nodes$color <- community_colors[as.character(nodes$group)]

  legend_labels <- lapply(seq_len(n_communities), function(i) {
    community_size <- sum(nodes$group == i)
    list(
      label = paste0("Community ", i, " (", community_size, ")"),
      color = community_colors[as.character(i)],
      shape = "dot"
    )
  })

  plot <- visNetwork::visNetwork(nodes, edges) %>%
    visNetwork::visNodes(font = list(color = "black", size = node_label_size, vadjust = 0)) %>%
    visNetwork::visEdges(smooth = FALSE) %>%
    visNetwork::visIgraphLayout(layout = "layout_with_fr", randomSeed = ifelse(is.null(seed), 2025, seed)) %>%
    visNetwork::visOptions(
      highlightNearest = list(
        enabled = TRUE,
        degree = 1,
        hover = TRUE,
        algorithm = "hierarchical"
      ),
      nodesIdSelection = TRUE,
      manipulation = FALSE,
      selectedBy = list(
        variable = "group",
        multiple = FALSE,
        style = "width: 150px; height: 26px;"
      )
    ) %>%
    visNetwork::visPhysics(enabled = FALSE) %>%
    visNetwork::visInteraction(
      hover = TRUE,
      tooltipDelay = 0,
      tooltipStay = 1000,
      zoomView = TRUE,
      dragView = TRUE
    ) %>%
    {if (showlegend) visNetwork::visLegend(.,
                                            addNodes = do.call(rbind, lapply(legend_labels, as.data.frame)),
                                            useGroups = FALSE,
                                            position = "right",
                                            width = 0.2,
                                            zoom = FALSE
    ) else .}

  # Compute comprehensive network statistics
  safe_round <- function(x, digits = 4) {
    if (is.null(x) || is.na(x) || is.nan(x) || !is.numeric(x)) return(NA_real_)
    round(x, digits)
  }

  stats_list <- list(
    nodes = igraph::vcount(graph),
    edges = igraph::ecount(graph),
    density = safe_round(igraph::edge_density(graph)),
    diameter = tryCatch(igraph::diameter(graph), error = function(e) NA_integer_),
    global_clustering = safe_round(igraph::transitivity(graph, type = "global")),
    avg_local_clustering = safe_round(igraph::transitivity(graph, type = "average")),
    modularity = safe_round(igraph::modularity(community_result)),
    assortativity = safe_round(igraph::assortativity_degree(graph)),
    avg_path_length = safe_round(igraph::mean_distance(graph, directed = FALSE))
  )

  list(
    plot = plot,
    table = layout_df,
    nodes = nodes,
    edges = edges,
    stats = stats_list
  )
}

#' Compute Word Correlation Network
#'
#' @description
#' Computes word correlation networks with community detection and network metrics.
#' Supports multiple feature spaces: unigrams, n-grams, and embeddings.
#' Based on proven implementation for intuitive network visualization.
#'
#' @param dfm_object A quanteda document-feature matrix (dfm).
#' @param doc_var A document-level metadata variable for categories (default: NULL).
#' @param common_term_n Minimum term frequency to include (default: 20).
#' @param corr_n Minimum correlation threshold (default: 0.4).
#' @param top_node_n Number of top nodes to display (default: 30).
#' @param node_label_size Font size for node labels (default: 14).
#' @param pattern Regex pattern to filter specific words (default: NULL).
#' @param showlegend Whether to show community legend (default: TRUE).
#' @param seed Random seed for reproducible layout (default: NULL).
#' @param feature_type Feature space: "words", "ngrams", or "embeddings" (default: "words").
#' @param ngram_range N-gram size when feature_type = "ngrams" (default: 2).
#' @param texts Optional character vector of texts for n-gram creation (default: NULL).
#' @param embeddings Optional embedding matrix for embedding-based networks (default: NULL).
#' @param embedding_sim_threshold Similarity threshold for embedding-based networks (default: 0.5).
#' @param community_method Community detection method: "leiden" (default), "louvain", "label_prop", or "fast_greedy".
#'
#' @return A list containing plot, table, nodes, edges, and stats
#' @family network
#' @export
#'
semantic_correlation_network <- function(dfm_object,
                                       doc_var = NULL,
                                       common_term_n = 20,
                                       corr_n = 0.4,
                                       top_node_n = 30,
                                       node_label_size = 22,
                                       pattern = NULL,
                                       showlegend = TRUE,
                                       seed = NULL,
                                       feature_type = "words",
                                       ngram_range = 2,
                                       texts = NULL,
                                       embeddings = NULL,
                                       embedding_sim_threshold = 0.5,
                                       community_method = "leiden") {

  if (!is.null(seed)) set.seed(seed)

  if (feature_type == "ngrams" && !is.null(texts)) {
    if (!requireNamespace("quanteda", quietly = TRUE)) {
      stop("Package 'quanteda' is required for n-gram analysis.")
    }
    message("Creating ", ngram_range, "-gram correlation network")
    corp <- quanteda::corpus(texts)
    toks <- quanteda::tokens(corp, remove_punct = TRUE, remove_symbols = TRUE)
    toks_ngrams <- quanteda::tokens_ngrams(toks, n = ngram_range)
    dfm_object <- quanteda::dfm(toks_ngrams)
  } else if (feature_type == "embeddings" && !is.null(embeddings)) {
    message("Creating embedding-based document similarity network")

    # For embeddings, create document-document similarity network
    # Nodes = Documents, Edges = Similarity > threshold

    if (!is.matrix(embeddings) || nrow(embeddings) < 2) {
      message("Invalid embeddings matrix. Falling back to word-based network.")
    } else {
      # Compute cosine similarity matrix
      norm_embeddings <- embeddings / sqrt(rowSums(embeddings^2))
      sim_matrix <- tcrossprod(norm_embeddings)
      diag(sim_matrix) <- 0  # Remove self-similarity

      # Get document names
      doc_names <- if (!is.null(rownames(embeddings))) {
        rownames(embeddings)
      } else {
        paste0("doc_", seq_len(nrow(embeddings)))
      }
      rownames(sim_matrix) <- doc_names
      colnames(sim_matrix) <- doc_names

      # Create edges from similarity threshold
      edge_list <- which(sim_matrix > embedding_sim_threshold & upper.tri(sim_matrix), arr.ind = TRUE)

      if (nrow(edge_list) == 0) {
        message("No document pairs exceed similarity threshold. Try lowering the threshold.")
        return(NULL)
      }

      edge_df <- data.frame(
        from = doc_names[edge_list[, 1]],
        to = doc_names[edge_list[, 2]],
        weight = sim_matrix[edge_list],
        stringsAsFactors = FALSE
      )

      # Create graph
      graph <- igraph::graph_from_data_frame(edge_df, directed = FALSE)
      if (igraph::vcount(graph) == 0) return(NULL)

      # Compute centrality measures
      igraph::V(graph)$degree <- igraph::degree(graph)
      igraph::V(graph)$eigenvector <- igraph::eigen_centrality(graph)$vector

      # Community detection
      community_result <- switch(community_method,
        "louvain" = igraph::cluster_louvain(graph),
        "label_prop" = igraph::cluster_label_prop(graph),
        "fast_greedy" = igraph::cluster_fast_greedy(igraph::as.undirected(graph)),
        igraph::cluster_leiden(graph)
      )
      igraph::V(graph)$community <- community_result$membership

      # Create layout table
      layout_df <- data.frame(
        Document = igraph::V(graph)$name,
        Degree = igraph::V(graph)$degree,
        Eigenvector = round(igraph::V(graph)$eigenvector, 3),
        Community = igraph::V(graph)$community,
        stringsAsFactors = FALSE
      )

      # Limit nodes by degree
      node_degrees <- igraph::degree(graph)
      sorted_indices <- order(node_degrees, decreasing = TRUE)
      top_n <- min(top_node_n, length(sorted_indices))
      top_indices <- sorted_indices[1:top_n]
      top_nodes <- igraph::V(graph)$name[top_indices]
      graph <- igraph::induced_subgraph(graph, top_nodes)

      # Prepare visNetwork data
      communities <- unique(igraph::V(graph)$community)
      n_communities <- length(communities)
      colors <- colorRampPalette(c("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                                   "#9467bd", "#8c564b", "#e377c2"))(n_communities)
      community_colors <- setNames(colors, communities)

      nodes <- data.frame(
        id = igraph::V(graph)$name,
        label = igraph::V(graph)$name,
        value = igraph::V(graph)$degree,
        group = paste("Community", igraph::V(graph)$community),
        color = community_colors[as.character(igraph::V(graph)$community)],
        font.size = node_label_size,
        title = paste0("<b>", igraph::V(graph)$name, "</b><br>",
                       "Degree: ", igraph::V(graph)$degree, "<br>",
                       "Community: ", igraph::V(graph)$community),
        stringsAsFactors = FALSE
      )

      edge_data <- igraph::as_data_frame(graph, what = "edges")
      edges <- data.frame(
        from = edge_data$from,
        to = edge_data$to,
        value = if ("weight" %in% names(edge_data)) edge_data$weight else 1,
        color = "#888888",
        title = paste0("Similarity: ", round(edge_data$weight, 3)),
        stringsAsFactors = FALSE
      )

      # Create legend
      legend_labels <- lapply(seq_len(n_communities), function(i) {
        list(
          label = paste("Community", communities[i]),
          shape = "dot",
          color = colors[i],
          size = 15
        )
      })

      # Create plot
      plot <- visNetwork::visNetwork(nodes, edges) %>%
        visNetwork::visNodes(
          shape = "dot",
          scaling = list(min = 10, max = 40)
        ) %>%
        visNetwork::visEdges(
          smooth = FALSE,
          width = 1,
          color = list(color = "#888888", opacity = 0.6)
        ) %>%
        visNetwork::visLayout(randomSeed = seed %||% 2025) %>%
        visNetwork::visOptions(
          highlightNearest = TRUE,
          nodesIdSelection = TRUE,
          manipulation = FALSE,
          selectedBy = list(
            variable = "group",
            multiple = FALSE,
            style = "width: 150px; height: 26px;"
          )
        ) %>%
        visNetwork::visPhysics(enabled = FALSE) %>%
        visNetwork::visInteraction(
          hover = TRUE,
          tooltipDelay = 0,
          tooltipStay = 1000,
          zoomView = TRUE,
          dragView = TRUE
        ) %>%
        {if (showlegend) visNetwork::visLegend(.,
                                                addNodes = do.call(rbind, lapply(legend_labels, as.data.frame)),
                                                useGroups = FALSE,
                                                position = "right",
                                                width = 0.2,
                                                zoom = FALSE
        ) else .}

      # Compute stats
      safe_round <- function(x, digits = 4) {
        if (is.null(x) || is.na(x) || is.nan(x) || !is.numeric(x)) return(NA_real_)
        round(x, digits)
      }

      stats_list <- list(
        nodes = igraph::vcount(graph),
        edges = igraph::ecount(graph),
        density = safe_round(igraph::edge_density(graph)),
        diameter = tryCatch(igraph::diameter(graph), error = function(e) NA_integer_),
        global_clustering = safe_round(igraph::transitivity(graph, type = "global")),
        avg_local_clustering = safe_round(igraph::transitivity(graph, type = "average")),
        modularity = safe_round(igraph::modularity(community_result)),
        assortativity = safe_round(igraph::assortativity_degree(graph)),
        avg_path_length = safe_round(igraph::mean_distance(graph, directed = FALSE))
      )

      return(list(
        plot = plot,
        table = layout_df,
        nodes = nodes,
        edges = edges,
        stats = stats_list
      ))
    }
  }

  dfm_td <- tidytext::tidy(dfm_object)
  docvars_df <- dfm_object@docvars
  docvars_df$document <- docvars_df$docname_
  dfm_td <- dplyr::left_join(dfm_td, docvars_df, by = "document")

  if (!is.null(doc_var) && doc_var %in% colnames(dfm_td)) {
    available_levels <- unique(dfm_td[[doc_var]])
    available_levels <- available_levels[!is.na(available_levels)]
    message("Analyzing network for ", doc_var, ": ", paste(available_levels, collapse = ", "))
  }

  term_cor <- dfm_td %>%
    dplyr::group_by(term) %>%
    dplyr::filter(dplyr::n() >= common_term_n) %>%
    widyr::pairwise_cor(term, document, count, sort = TRUE) %>%
    dplyr::ungroup() %>%
    dplyr::filter(correlation > corr_n)

  if (!is.null(pattern)) {
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

  igraph::V(graph)$degree <- igraph::degree(graph)
  igraph::V(graph)$eigenvector <- igraph::eigen_centrality(graph)$vector
  # Apply selected community detection method
  community_result <- switch(community_method,
    "louvain" = igraph::cluster_louvain(graph),
    "label_prop" = igraph::cluster_label_prop(graph),
    "fast_greedy" = igraph::cluster_fast_greedy(igraph::as.undirected(graph)),
    igraph::cluster_leiden(graph)  # default: leiden
  )
  igraph::V(graph)$community <- community_result$membership

  layout_df <- data.frame(
    Term = igraph::V(graph)$name,
    Degree = igraph::V(graph)$degree,
    Eigenvector = round(igraph::V(graph)$eigenvector, 3),
    Community = igraph::V(graph)$community,
    stringsAsFactors = FALSE
  )

  node_degrees <- igraph::degree(graph)
  sorted_indices <- order(node_degrees, decreasing = TRUE)
  top_n <- min(top_node_n, length(sorted_indices))
  top_nodes <- names(node_degrees)[sorted_indices[1:top_n]]

  nodes <- data.frame(
    id = igraph::V(graph)$name,
    label = ifelse(igraph::V(graph)$name %in% top_nodes, igraph::V(graph)$name, ""),
    group = igraph::V(graph)$community,
    value = igraph::V(graph)$degree,
    title = paste0(
      "<b style='color:black;'>", igraph::V(graph)$name, "</b><br>",
      "<span style='color:black;'>Degree: ", igraph::V(graph)$degree, "<br>",
      "Eigenvector: ", round(igraph::V(graph)$eigenvector, 2), "<br>",
      "Community: ", igraph::V(graph)$community, "</span>"
    ),
    stringsAsFactors = FALSE
  )

  edges <- igraph::as_data_frame(graph, what = "edges")
  edges$correlation <- term_cor$correlation[match(paste(edges$from, edges$to), paste(term_cor$item1, term_cor$item2))]
  edges$width <- scales::rescale(abs(edges$correlation), to = c(1, 8))

  edge_color_base <- "#5C5CFF"
  edges$color <- mapply(function(corr) {
    alpha_val <- scales::rescale(abs(corr), to = c(0.3, 1))
    scales::alpha(edge_color_base, alpha_val)
  }, edges$correlation)

  edges$title <- paste0(
    "<span style='color:black;'>Correlation: ", round(edges$correlation, 3),
    "<br>From: ", edges$from,
    "<br>To: ", edges$to, "</span>"
  )

  unique_communities <- sort(unique(nodes$group))
  community_map <- setNames(seq_along(unique_communities), unique_communities)
  nodes$group <- community_map[as.character(nodes$group)]

  n_communities <- length(unique(nodes$group))
  if (n_communities <= 8) {
    palette <- RColorBrewer::brewer.pal(max(3, n_communities), "Set2")
  } else {
    palette <- grDevices::colorRampPalette(RColorBrewer::brewer.pal(8, "Set2"))(n_communities)
  }
  community_colors <- setNames(palette, as.character(seq_len(n_communities)))
  nodes$color <- community_colors[as.character(nodes$group)]

  legend_labels <- lapply(seq_len(n_communities), function(i) {
    community_size <- sum(nodes$group == i)
    list(
      label = paste0("Community ", i, " (", community_size, ")"),
      color = community_colors[as.character(i)],
      shape = "dot"
    )
  })

  plot <- visNetwork::visNetwork(nodes, edges) %>%
    visNetwork::visNodes(font = list(color = "black", size = node_label_size, vadjust = 0)) %>%
    visNetwork::visEdges(smooth = FALSE) %>%
    visNetwork::visIgraphLayout(layout = "layout_with_fr", randomSeed = ifelse(is.null(seed), 2025, seed)) %>%
    visNetwork::visOptions(
      highlightNearest = list(
        enabled = TRUE,
        degree = 1,
        hover = TRUE,
        algorithm = "hierarchical"
      ),
      nodesIdSelection = TRUE,
      manipulation = FALSE,
      selectedBy = list(
        variable = "group",
        multiple = FALSE,
        style = "width: 150px; height: 26px;"
      )
    ) %>%
    visNetwork::visPhysics(enabled = FALSE) %>%
    visNetwork::visInteraction(
      hover = TRUE,
      tooltipDelay = 0,
      tooltipStay = 1000,
      zoomView = TRUE,
      dragView = TRUE
    ) %>%
    {if (showlegend) visNetwork::visLegend(.,
                                            addNodes = do.call(rbind, lapply(legend_labels, as.data.frame)),
                                            useGroups = FALSE,
                                            position = "right",
                                            width = 0.2,
                                            zoom = FALSE
    ) else .}

  # Compute comprehensive network statistics
  safe_round <- function(x, digits = 4) {
    if (is.null(x) || is.na(x) || is.nan(x) || !is.numeric(x)) return(NA_real_)
    round(x, digits)
  }

  stats_list <- list(
    nodes = igraph::vcount(graph),
    edges = igraph::ecount(graph),
    density = safe_round(igraph::edge_density(graph)),
    diameter = tryCatch(igraph::diameter(graph), error = function(e) NA_integer_),
    global_clustering = safe_round(igraph::transitivity(graph, type = "global")),
    avg_local_clustering = safe_round(igraph::transitivity(graph, type = "average")),
    modularity = safe_round(igraph::modularity(community_result)),
    assortativity = safe_round(igraph::assortativity_degree(graph)),
    avg_path_length = safe_round(igraph::mean_distance(graph, directed = FALSE))
  )
  list(
    plot = plot,
    table = layout_df,
    nodes = nodes,
    edges = edges,
    stats = stats_list
  )
}
