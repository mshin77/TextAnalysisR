#' @importFrom utils modifyList
#' @importFrom stats cor
NULL

# Network Analysis Functions
# Functions for word co-occurrence and correlation network analysis

#' @title Analyze and Visualize Word Co-occurrence Networks
#'
#' @description
#' This function creates a word co-occurrence network based on a document-feature matrix (dfm).
#'
#' @param dfm_object A quanteda document-feature matrix (dfm).
#' @param doc_var A document-level metadata variable (default: NULL).
#' @param co_occur_n Minimum number of co-occurrences for filtering terms (default: 50).
#' @param top_node_n Number of top nodes to display (default: 30).
#' @param nrows Number of rows to display in the table (default: 1).
#' @param height The height of the resulting Plotly plot, in pixels (default: 800).
#' @param width The width of the resulting Plotly plot, in pixels (default: 900).
#' @param category An optional category to filter the data (default: NULL).
#' @param use_category_specific Logical; if TRUE, uses category-specific
#'   parameters (default: FALSE).
#' @param category_params A named list of parameters for each category level (default: NULL).
#'
#' @return A list containing the Plotly plot, a data frame of the network
#'   layout, and the igraph graph object.
#'
#' @importFrom igraph graph_from_data_frame V vcount degree betweenness
#'   closeness eigen_centrality layout_with_fr
#' @importFrom dplyr count filter mutate select group_by summarise
#' @importFrom tibble as_tibble
#' @importFrom tidytext tidy
#' @importFrom widyr pairwise_count
#' @importFrom stats quantile
#' @importFrom shiny showNotification
#' @importFrom rlang sym %||%
#' @importFrom utils head
#' @importFrom grDevices colorRampPalette
#'
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
#'   # Overall
#'   word_co_occurrence_network_results <- TextAnalysisR::plot_cooccurrence_network(
#'                                         dfm_object,
#'                                         doc_var = "reference_type",
#'                                         co_occur_n = 30,
#'                                         top_node_n = 0,
#'                                         nrows = 1,
#'                                         height = 800,
#'                                         width = 900)
#'
#'   print(word_co_occurrence_network_results$plot)
#'   print(word_co_occurrence_network_results$table)
#'   print(word_co_occurrence_network_results$summary)
#'
#'   # Journal article
#'  category_params <- list(
#'   "journal_article" = list(co_occur_n = 80, top_node_n = 20),
#'   "thesis" = list(co_occur_n = 30, top_node_n = 20)
#' )
#'
#'  word_co_occurrence_category <- TextAnalysisR::plot_cooccurrence_network(
#'   dfm_object,
#'   doc_var = "reference_type",
#'   use_category_specific = TRUE,
#'   category_params = category_params)
#'
#'  print(word_co_occurrence_category$journal_article$plot)
#'  print(word_co_occurrence_category$journal_article$table)
#'  print(word_co_occurrence_category$journal_article$summary)
#'
#'  # Thesis
#'  print(word_co_occurrence_category$thesis$plot)
#'  print(word_co_occurrence_category$thesis$table)
#'  print(word_co_occurrence_category$thesis$summary)
#'
#' }
plot_cooccurrence_network <- function(dfm_object,
                                      doc_var = NULL,
                                      co_occur_n = 50,
                                      top_node_n = 30,
                                      nrows = 1,
                                      height = 800,
                                      width = 900,
                                      category = NULL,
                                      use_category_specific = FALSE,
                                      category_params = NULL) {

  if (!requireNamespace("htmltools", quietly = TRUE) ||
      !requireNamespace("RColorBrewer", quietly = TRUE)) {
    stop(
      "The 'htmltools' and 'RColorBrewer' packages are required for this functionality. ",
      "Please install them using install.packages(c('htmltools', 'RColorBrewer'))."
    )
  }

  if (!requireNamespace("plotly", quietly = TRUE)) {
    stop("plotly package is required for this function. ",
         "Please install it with: install.packages('plotly')")
  }

  if (!requireNamespace("scales", quietly = TRUE)) {
    stop("scales package is required for this function. ",
         "Please install it with: install.packages('scales')")
  }

  if (!requireNamespace("DT", quietly = TRUE)) {
    stop("DT package is required for this function. ",
         "Please install it with: install.packages('DT')")
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
    if (!is.null(category)) {
      dfm_td <- dfm_td[dfm_td[[doc_var]] == category, ]
    }
    docvar_levels <- unique(dfm_td[[doc_var]])
    print(paste("doc_var has", length(docvar_levels), "levels:",
                paste(docvar_levels, collapse = ", ")))
  } else {
    docvar_levels <- NULL
  }

  if (use_category_specific && !is.null(doc_var) && !is.null(category_params)) {
    cat_levels <- unique(dfm_td[[doc_var]])
    cat_levels <- cat_levels[!is.na(cat_levels)]

    results_list <- list()

    for (level in cat_levels) {
      level_str <- as.character(level)

      if (level_str %in% names(category_params)) {
        level_co_occur_n <- category_params[[level_str]]$co_occur_n %||% co_occur_n
        level_top_node_n <- category_params[[level_str]]$top_node_n %||% top_node_n
      } else {
        level_co_occur_n <- co_occur_n
        level_top_node_n <- top_node_n
      }

      level_indices <- which(dfm_object@docvars[[doc_var]] == level)
      if (length(level_indices) > 0) {
        level_dfm <- dfm_object[level_indices, ]

        if (quanteda::ndoc(level_dfm) > 0) {
          result <- plot_cooccurrence_network(
            dfm_object = level_dfm,
            doc_var = NULL,
            co_occur_n = level_co_occur_n,
            top_node_n = level_top_node_n,
            nrows = 1,
            width = width,
            height = height,
            use_category_specific = FALSE
          )
          results_list[[level_str]] <- result
        }
      }
    }

    return(results_list)
  }

  build_table <- function(net, group_label) {
    layout_dff <- net$layout_df %>%
      dplyr::select(-c("x", "y")) %>%
      dplyr::mutate_if(is.numeric, round, digits = 3)

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
          style = "font-weight: bold; text-align: center; font-size: 18px; font-family: 'Montserrat', sans-serif; color: #0c1f4a;"
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
      dplyr::mutate_if(is.numeric, round, digits = 3)

    summary_table <- DT::datatable(summary_df, rownames = FALSE,
                                   extensions = 'Buttons',
                                   options = list(scrollX = TRUE,
                                                  width = "80%",
                                                  dom = 'Bfrtip',
                                                  buttons = c('copy', 'csv', 'excel',
                                                             'pdf', 'print'))) %>%
      DT::formatStyle(columns = colnames(summary_df), `font-size` = "16px")

    htmltools::tagList(
      htmltools::tags$div(
        style = "margin-bottom: 20px;",
        htmltools::tags$p(
          group_label,
          style = "font-weight: bold; text-align: center; font-size: 18px; font-family: 'Montserrat', sans-serif; color: #0c1f4a;"
        )
      ),
      summary_table
    )
  }

  build_network_plot <- function(data, group_level = NULL) {
    term_co_occur <- data %>%
      widyr::pairwise_count(term, document, sort = TRUE) %>%
      dplyr::filter(n >= co_occur_n)

    graph <- igraph::graph_from_data_frame(term_co_occur, directed = FALSE)
    if (igraph::vcount(graph) == 0) {
      message("No co-occurrence relationships meet the threshold.")
      return(NULL)
    }
    igraph::V(graph)$degree      <- igraph::degree(graph)
    igraph::V(graph)$betweenness <- igraph::betweenness(graph)
    igraph::V(graph)$closeness   <- igraph::closeness(graph)
    igraph::V(graph)$eigenvector <- igraph::eigen_centrality(graph)$vector
    igraph::V(graph)$community   <- igraph::cluster_leiden(graph)$membership

    layout_mat <- igraph::layout_with_fr(graph)
    layout_df <- as.data.frame(layout_mat) %>% stats::setNames(c("x", "y"))
    layout_df <- layout_df %>%
      dplyr::mutate(label       = igraph::V(graph)$name,
                    degree      = igraph::V(graph)$degree,
                    betweenness = igraph::V(graph)$betweenness,
                    closeness   = igraph::V(graph)$closeness,
                    eigenvector = igraph::V(graph)$eigenvector,
                    community   = igraph::V(graph)$community)

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

    node_data <- layout_df %>%
      dplyr::mutate(degree_log = log1p(degree),
                    size       = scales::rescale(degree_log, to = c(12, 30)),
                    text_size  = scales::rescale(degree_log, to = c(14, 20)),
                    alpha      = scales::rescale(degree_log, to = c(0.2, 1)),
                    hover_text = paste("Word:", label,
                                       "<br>Degree:", degree,
                                       "<br>Betweenness:", round(betweenness, 2),
                                       "<br>Closeness:", round(closeness, 2),
                                       "<br>Eigenvector:", round(eigenvector, 2),
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
    node_data$color <- palette[as.character(node_data$community)]

    p <- plotly::plot_ly(width = width, height = height)
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
    for(comm in community_levels) {
      comm_data <- dplyr::filter(node_data, community == comm)
      p <- p %>% plotly::add_markers(data = comm_data, x = ~x, y = ~y,
                                     marker = list(size = ~size, color = palette[comm],
                                                   showscale = FALSE,
                                                   line = list(width = 3, color = '#FFFFFF')),
                                     hoverinfo = 'text', text = ~hover_text,
                                     showlegend = TRUE, name = paste("Community", comm),
                                     legendgroup = "Community")
    }
    top_nodes <- dplyr::arrange(node_data, desc(degree)) %>% head(top_node_n)
    annotations <- if (nrow(top_nodes) > 0) {
      lapply(1:nrow(top_nodes), function(i) {
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
      list(list(x = 0, y = 0, text = "", showarrow = FALSE, visible = FALSE))
    }

    layout_args <- list(
      dragmode = "pan",
      title = list(text = "Word Co-occurrence Network",
                   font = list(size = 18, color = "#0c1f4a", family = "Montserrat")),
      showlegend = TRUE,
      xaxis = list(title = "", showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
      yaxis = list(title = "", showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
      margin = list(l = 40, r = 100, t = 60, b = 40),
      annotations = annotations,
      legend = list(title = list(text = "Co-occurrence"),
                    orientation = "v", x = 1.1, y = 1,
                    xanchor = "left", yanchor = "top")
    )

    p <- do.call(plotly::layout, c(list(p), layout_args))
    list(plot = p, layout_df = layout_df, graph = graph)
  }

  if (!is.null(doc_var) && length(docvar_levels) > 1) {
    plots_list <- dfm_td %>%
      dplyr::ungroup() %>%
      dplyr::group_by(!!rlang::sym(doc_var)) %>%
      dplyr::group_map(~ {
        group_level <- .y[[doc_var]]
        print(paste("Processing group level:", group_level))

        if (is.null(group_level)) {
          stop("doc_var is missing or not found in the current group")
        }

        net <- build_network_plot(.x, group_level)
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
                font = list(size = 18, color = "#0c1f4a", family = "Montserrat")
              )
            )
          )
        } else {
          NULL
        }
      })

    plots_list <- plots_list[!sapply(plots_list, is.null)]

    if (length(plots_list) == 0) {
      stop("No valid plots generated. All category levels may have insufficient data for analysis.")
    }

    combined_plot <- plotly::subplot(plots_list, nrows = nrows, shareX = TRUE, shareY = TRUE,
                                     titleX = TRUE, titleY = TRUE)

    table_list <- lapply(docvar_levels, function(level) {
      print(paste("Generating table for level:", level))
      group_data <- dplyr::filter(dfm_td, !!rlang::sym(doc_var) == level)
      net <- build_network_plot(group_data)
      if (!is.null(net)) build_table(net, level) else NULL
    })

    summary_list <- lapply(docvar_levels, function(level) {
      print(paste("Generating summary for level:", level))
      group_data <- dplyr::filter(dfm_td, !!rlang::sym(doc_var) == level)
      net <- build_network_plot(group_data)
      if (!is.null(net)) build_summary(net, level) else NULL
    })

    table_list <- table_list[!sapply(table_list, is.null)]
    summary_list <- summary_list[!sapply(summary_list, is.null)]

    return(list(
      plot = combined_plot,
      table = if(length(table_list) > 0) table_list %>% htmltools::tagList() %>% htmltools::browsable() else NULL,
      summary = if(length(summary_list) > 0) summary_list %>% htmltools::tagList() %>% htmltools::browsable() else NULL
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


#' @title Analyze and Visualize Word Correlation Networks
#'
#' @description
#' This function creates a word correlation network based on a document-feature matrix (dfm).
#'
#' @param dfm_object A quanteda document-feature matrix (dfm).
#' @param doc_var A document-level metadata variable (default: NULL).
#' @param common_term_n Minimum number of common terms for filtering terms (default: 30).
#' @param corr_n Minimum correlation value for filtering terms (default: 0.4).
#' @param top_node_n Number of top nodes to display (default: 40).
#' @param nrows Number of rows to display in the table (default: 1).
#' @param height The height of the resulting Plotly plot, in pixels (default: 1000).
#' @param width The width of the resulting Plotly plot, in pixels (default: 900).
#' @param use_category_specific Logical; if TRUE, uses category-specific
#'   parameters (default: FALSE).
#' @param category_params A named list of parameters for each category level (default: NULL).
#'
#' @return A list containing the Plotly plot, a data frame of the network
#'   layout, and the igraph graph object.
#'
#' @importFrom igraph graph_from_data_frame V vcount degree betweenness
#'   closeness eigen_centrality layout_with_fr
#' @importFrom dplyr count filter mutate select group_by summarise
#' @importFrom tibble as_tibble
#' @importFrom tidytext tidy
#' @importFrom widyr pairwise_cor
#' @importFrom stats quantile
#' @importFrom shiny showNotification
#' @importFrom utils head
#' @importFrom grDevices colorRampPalette
#'
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
#'   # Overall
#'   word_correlation_network_results <- TextAnalysisR::plot_correlation_network(
#'                                         dfm_object,
#'                                         doc_var = "reference_type",
#'                                         common_term_n = 30,
#'                                         corr_n = 0.4,
#'                                         top_node_n = 0,
#'                                         nrows = 1,
#'                                         height = 800,
#'                                         width = 900)
#'
#'   print(word_correlation_network_results$plot)
#'   print(word_correlation_network_results$table)
#'   print(word_correlation_network_results$summary)
#'
#'   # Journal article
#'  category_params <- list(
#'    "journal_article" = list(common_term_n = 30, corr_n = 0.4, top_node_n = 20),
#'    "thesis" = list(common_term_n = 20, corr_n = 0.4, top_node_n = 20)
#' )
#'
#'  word_correlation_category <- TextAnalysisR::plot_correlation_network(
#'    dfm_object,
#'    doc_var = "reference_type",
#'    use_category_specific = TRUE,
#'    category_params = category_params)
#'
#'  print(word_correlation_category$journal_article$plot)
#'  print(word_correlation_category$journal_article$table)
#'  print(word_correlation_category$journal_article$summary)
#'
#'  # Thesis
#'  print(word_correlation_category$thesis$plot)
#'  print(word_correlation_category$thesis$table)
#'  print(word_correlation_category$thesis$summary)
#'
#' }
plot_correlation_network <- function(dfm_object,
                                     doc_var = NULL,
                                     common_term_n = 130,
                                     corr_n = 0.4,
                                     top_node_n = 40,
                                     nrows = 1,
                                     height = 1000,
                                     width = 900,
                                     use_category_specific = FALSE,
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
    print(paste("doc_var has", length(docvar_levels), "levels:",
                paste(docvar_levels, collapse = ", ")))
  } else {
    docvar_levels <- NULL
  }

  if (use_category_specific && !is.null(doc_var) && !is.null(category_params)) {
    cat_levels <- unique(dfm_td[[doc_var]])
    cat_levels <- cat_levels[!is.na(cat_levels)]

    results_list <- list()

    for (level in cat_levels) {
      level_str <- as.character(level)

      if (level_str %in% names(category_params)) {
        level_common_term_n <- category_params[[level_str]]$common_term_n %||% common_term_n
        level_corr_n <- category_params[[level_str]]$corr_n %||% corr_n
        level_top_node_n <- category_params[[level_str]]$top_node_n %||% top_node_n
      } else {
        level_common_term_n <- common_term_n
        level_corr_n <- corr_n
        level_top_node_n <- top_node_n
      }

      level_indices <- which(dfm_object@docvars[[doc_var]] == level)
      if (length(level_indices) > 0) {
        level_dfm <- dfm_object[level_indices, ]

        if (quanteda::ndoc(level_dfm) > 0) {
          result <- plot_correlation_network(
            dfm_object = level_dfm,
            doc_var = NULL,
            common_term_n = level_common_term_n,
            corr_n = level_corr_n,
            top_node_n = level_top_node_n,
            nrows = 1,
            width = width,
            height = height,
            use_category_specific = FALSE
          )
          results_list[[level_str]] <- result
        }
      }
    }

    return(results_list)
  }

  build_table <- function(net, group_label) {
    layout_dff <- net$layout_df %>%
      dplyr::select(-c("x", "y")) %>%
      dplyr::mutate_if(is.numeric, round, digits = 3)

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
          style = "font-weight: bold; text-align: center; font-size: 18px; font-family: 'Montserrat', sans-serif; color: #0c1f4a;"
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
      dplyr::mutate_if(is.numeric, round, digits = 3)

    summary_table <- DT::datatable(summary_df, rownames = FALSE,
                                   extensions = 'Buttons',
                                   options = list(scrollX = TRUE,
                                                  width = "80%",
                                                  dom = 'Bfrtip',
                                                  buttons = c('copy', 'csv', 'excel',
                                                             'pdf', 'print'))) %>%
      DT::formatStyle(columns = colnames(summary_df), `font-size` = "16px")

    htmltools::tagList(
      htmltools::tags$div(
        style = "margin-bottom: 20px;",
        htmltools::tags$p(
          group_label,
          style = "font-weight: bold; text-align: center; font-size: 18px; font-family: 'Montserrat', sans-serif; color: #0c1f4a;"
        )
      ),
      summary_table
    )
  }

  build_network_plot <- function(data, group_level = NULL) {
    term_cor <- data %>%
      group_by(term) %>%
      filter(n() >= common_term_n) %>%
      widyr::pairwise_cor(term, document, sort = TRUE) %>%
      dplyr::ungroup() %>%
      dplyr::filter(correlation > corr_n)

    graph <- igraph::graph_from_data_frame(term_cor, directed = FALSE)
    if(igraph::vcount(graph) == 0) {
      message("No correlation relationships meet the threshold.")
      return(NULL)
    }
    igraph::V(graph)$degree      <- igraph::degree(graph)
    igraph::V(graph)$betweenness <- igraph::betweenness(graph)
    igraph::V(graph)$closeness   <- igraph::closeness(graph)
    igraph::V(graph)$eigenvector <- igraph::eigen_centrality(graph)$vector
    igraph::V(graph)$community   <- igraph::cluster_leiden(graph)$membership

    layout_mat <- igraph::layout_with_fr(graph)
    layout_df <- as.data.frame(layout_mat) %>% stats::setNames(c("x", "y"))
    layout_df <- layout_df %>%
      dplyr::mutate(label       = igraph::V(graph)$name,
                    degree      = igraph::V(graph)$degree,
                    betweenness = igraph::V(graph)$betweenness,
                    closeness   = igraph::V(graph)$closeness,
                    eigenvector = igraph::V(graph)$eigenvector,
                    community   = igraph::V(graph)$community)

    edge_data <- igraph::as_data_frame(graph, what = "edges") %>%
      dplyr::mutate(x    = layout_df$x[match(from, layout_df$label)],
                    y    = layout_df$y[match(from, layout_df$label)],
                    xend = layout_df$x[match(to, layout_df$label)],
                    yend = layout_df$y[match(to, layout_df$label)],
                    correlation = correlation,
                    correlation_rounded = round(correlation, 3)) %>%
      dplyr::select(dplyr::all_of(c("from", "to", "x", "y", "xend", "yend",
                                    "correlation", "correlation_rounded"))) %>%
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
      group_by(line_group) %>%
      dplyr::summarise(
        min_corr = min(correlation, na.rm = TRUE),
        max_corr = max(correlation, na.rm = TRUE)
      ) %>%
      dplyr::mutate(label = paste0("Correlation: ", round(min_corr, 2), " - ",
                                   round(max_corr, 2))) %>%
      dplyr::pull(label)

    node_data <- layout_df %>%
      dplyr::mutate(
        degree_log = log1p(degree),
        size = scales::rescale(degree_log, to = c(12, 30)),
        text_size = scales::rescale(degree_log, to = c(14, 20)),
        alpha = scales::rescale(degree_log, to = c(0.2, 1)),
        hover_text = paste(
          "Word:", label,
          "<br>Degree:", degree,
          "<br>Betweenness:", round(betweenness, 2),
          "<br>Closeness:", round(closeness, 2),
          "<br>Eigenvector:", round(eigenvector, 2),
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
    node_data$color <- palette[as.character(node_data$community)]

    p <- plotly::plot_ly(width = width, height = height)
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
                            customdata = ~correlation,
                            hovertemplate = paste0("Correlation: %{customdata:.3f}<br>",
                                                   "Source: %{text}<br>Target: %{meta}<extra></extra>"),
                            text = ~from,
                            meta = ~to,
                            showlegend = FALSE)
      }
    }
    for(comm in community_levels) {
      comm_data <- dplyr::filter(node_data, community == comm)
      p <- p %>% plotly::add_markers(data = comm_data, x = ~x, y = ~y,
                                     marker = list(size = ~size, color = palette[comm],
                                                   showscale = FALSE,
                                                   line = list(width = 3, color = '#FFFFFF')),
                                     hoverinfo = 'text', text = ~hover_text,
                                     showlegend = TRUE, name = paste("Community", comm),
                                     legendgroup = "Community")
    }
    top_nodes <- dplyr::arrange(node_data, desc(degree)) %>% head(top_node_n)
    annotations <- if (nrow(top_nodes) > 0) {
      lapply(1:nrow(top_nodes), function(i) {
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
      list(list(x = 0, y = 0, text = "", showarrow = FALSE, visible = FALSE))
    }

    layout_args <- list(
      dragmode = "pan",
      title = list(text = "Word Correlation Network",
                   font = list(size = 18, color = "#0c1f4a", family = "Montserrat")),
      showlegend = TRUE,
      xaxis = list(title = "", showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
      yaxis = list(title = "", showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
      margin = list(l = 40, r = 100, t = 60, b = 40),
      annotations = annotations,
      legend = list(title = list(text = "Correlation"),
                    orientation = "v", x = 1.1, y = 1,
                    xanchor = "left", yanchor = "top"),
      hoverlabel = list(namelength = -1)
    )

    p <- do.call(plotly::layout, c(list(p), layout_args))
    list(plot = p, layout_df = layout_df, graph = graph)
  }

  if (!is.null(doc_var) && length(docvar_levels) > 1) {
    plots_list <- dfm_td %>%
      dplyr::ungroup() %>%
      dplyr::group_by(!!rlang::sym(doc_var)) %>%
      dplyr::group_map(~ {
        group_level <- .y[[doc_var]]
        print(paste("Processing group level:", group_level))

        if (is.null(group_level)) {
          stop("doc_var is missing or not found in the current group")
        }

        net <- build_network_plot(.x, group_level)
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
                font = list(size = 18, color = "#0c1f4a", family = "Montserrat")
              )
            )
          )
        } else {
          NULL
        }
      })

    plots_list <- plots_list[!sapply(plots_list, is.null)]

    if (length(plots_list) == 0) {
      stop("No valid plots generated. All category levels may have insufficient data for analysis.")
    }

    combined_plot <- plotly::subplot(plots_list, nrows = nrows, shareX = TRUE, shareY = TRUE,
                                     titleX = TRUE, titleY = TRUE)

    table_list <- lapply(docvar_levels, function(level) {
      print(paste("Generating table for level:", level))
      group_data <- dplyr::filter(dfm_td, !!rlang::sym(doc_var) == level)
      net <- build_network_plot(group_data)
      if (!is.null(net)) build_table(net, level) else NULL
    })

    summary_list <- lapply(docvar_levels, function(level) {
      print(paste("Generating summary for level:", level))
      group_data <- dplyr::filter(dfm_td, !!rlang::sym(doc_var) == level)
      net <- build_network_plot(group_data)
      if (!is.null(net)) build_summary(net, level) else NULL
    })

    table_list <- table_list[!sapply(table_list, is.null)]
    summary_list <- summary_list[!sapply(summary_list, is.null)]

    return(list(
      plot = combined_plot,
      table = if(length(table_list) > 0) table_list %>% htmltools::tagList() %>% htmltools::browsable() else NULL,
      summary = if(length(summary_list) > 0) summary_list %>% htmltools::tagList() %>% htmltools::browsable() else NULL
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
#'
#' @return A list containing plot, table, nodes, edges, and stats
#' @export
#'
semantic_cooccurrence_network <- function(dfm_object,
                                        doc_var = NULL,
                                        co_occur_n = 10,
                                        top_node_n = 30,
                                        node_label_size = 14,
                                        pattern = NULL,
                                        showlegend = TRUE,
                                        seed = NULL,
                                        feature_type = "words",
                                        ngram_range = 2,
                                        texts = NULL,
                                        embeddings = NULL) {

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
    message("Creating embedding-based co-occurrence network with virtual edges")
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
  igraph::V(graph)$community <- igraph::cluster_leiden(graph)$membership

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
    visNetwork::visPhysics(
      solver = "barnesHut",
      barnesHut = list(
        gravitationalConstant = -1500,
        centralGravity = 0.4,
        springLength = 100,
        springConstant = 0.05,
        avoidOverlap = 0.3
      ),
      stabilization = list(enabled = TRUE, iterations = 1000)
    ) %>%
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
    ) else .} %>%
    visNetwork::visLayout(randomSeed = ifelse(is.null(seed), 2025, seed))

  stats_list <- list(
    nodes = igraph::vcount(graph),
    edges = igraph::ecount(graph),
    density = round(igraph::edge_density(graph), 3),
    diameter = igraph::diameter(graph)
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
#'
#' @return A list containing plot, table, nodes, edges, and stats
#' @export
#'
semantic_correlation_network <- function(dfm_object,
                                       doc_var = NULL,
                                       common_term_n = 20,
                                       corr_n = 0.4,
                                       top_node_n = 30,
                                       node_label_size = 14,
                                       pattern = NULL,
                                       showlegend = TRUE,
                                       seed = NULL,
                                       feature_type = "words",
                                       ngram_range = 2,
                                       texts = NULL,
                                       embeddings = NULL) {

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
    message("Creating embedding-based correlation network")
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
  igraph::V(graph)$community <- igraph::cluster_leiden(graph)$membership

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
    visNetwork::visPhysics(
      solver = "barnesHut",
      barnesHut = list(
        gravitationalConstant = -1500,
        centralGravity = 0.4,
        springLength = 100,
        springConstant = 0.05,
        avoidOverlap = 0.3
      ),
      stabilization = list(enabled = TRUE, iterations = 1000)
    ) %>%
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
    ) else .} %>%
    visNetwork::visLayout(randomSeed = ifelse(is.null(seed), 2025, seed))

  stats_list <- list(
    nodes = igraph::vcount(graph),
    edges = igraph::ecount(graph),
    density = round(igraph::edge_density(graph), 3)
  )

  list(
    plot = plot,
    table = layout_df,
    nodes = nodes,
    edges = edges,
    stats = stats_list
  )
}
