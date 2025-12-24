#' @importFrom utils modifyList
#' @importFrom stats cor
#' @importFrom quanteda.textstats textstat_frequency
NULL

# Suppress R CMD check notes for NSE variables
utils::globalVariables(c("term", "word_frequency", "collocation", "collocation_ordered",
                         "pos", "entity", "n", "count", "feature", "frequency"))

#' @title Plot Word Probabilities by Topic
#'
#' @description
#' Creates a faceted bar plot showing the top terms and their probabilities (beta values)
#' for each topic in a topic model.
#'
#' @param top_topic_terms A data frame containing topic terms with columns: topic, term, and beta.
#'   Typically created using get_topic_terms() or similar functions.
#' @param topic_label Optional topic labels. Can be either a named vector mapping topic numbers
#'   to labels, or a character string specifying a column name in top_topic_terms (default: NULL).
#' @param ncol Number of columns for facet wrap layout (default: 3).
#' @param height The height of the resulting Plotly plot, in pixels (default: 1200).
#' @param width The width of the resulting Plotly plot, in pixels (default: 800).
#' @param ylab Y-axis label (default: "Word probability").
#' @param title Plot title (default: NULL for auto-generated title).
#' @param colors Color palette for topics (default: NULL for auto-generated colors).
#' @param measure_label Label for the probability measure (default: "Beta").
#' @param ... Additional arguments passed to plotly::ggplotly().
#'
#' @return A plotly object showing word probabilities faceted by topic.
#'
#' @family visualization
#' @export
#'
#' @examples
#' if (interactive()) {
#'   top_terms <- data.frame(
#'     topic = rep(1:2, each = 5),
#'     term = c("learning", "student", "education", "school", "teacher",
#'              "technology", "computer", "digital", "software", "system"),
#'     beta = c(0.05, 0.04, 0.03, 0.02, 0.01, 0.06, 0.05, 0.04, 0.03, 0.02)
#'   )
#'   plot <- plot_word_probability(top_terms)
#'   print(plot)
#' }
plot_word_probability <- function(top_topic_terms,
                                   topic_label = NULL,
                                   ncol = 3,
                                   height = 1200,
                                   width = 800,
                                   ylab = "Word probability",
                                   title = NULL,
                                   colors = NULL,
                                   measure_label = "Beta",
                                   ...) {

  if (!"topic" %in% colnames(top_topic_terms)) {
    stop("The data frame must contain a 'topic' column.")
  }

  top_topic_terms <- top_topic_terms %>%
    dplyr::mutate(topic = as.character(topic))

  if (!is.null(topic_label)) {
    if (is.vector(topic_label) && !is.null(names(topic_label))) {
      manual_labels_df <- data.frame(
        topic = names(topic_label),
        label = unname(topic_label),
        stringsAsFactors = FALSE
      )
      top_topic_terms <- top_topic_terms %>%
        dplyr::left_join(manual_labels_df, by = "topic") %>%
        dplyr::mutate(labeled_topic = ifelse(!is.na(label), label, paste("Topic", topic))) %>%
        dplyr::select(-label)
    } else if (is.character(topic_label) && length(topic_label) == 1) {
      if (!topic_label %in% colnames(top_topic_terms)) {
        stop(paste("Column", topic_label, "not found in top_topic_terms."))
      }
      top_topic_terms <- top_topic_terms %>%
        dplyr::mutate(labeled_topic = as.character(.data[[topic_label]]))
    } else {
      top_topic_terms <- top_topic_terms %>%
        dplyr::mutate(labeled_topic = paste("Topic", topic))
    }
  } else {
    top_topic_terms <- top_topic_terms %>%
      dplyr::mutate(labeled_topic = paste("Topic", topic))
  }

  top_topic_terms <- top_topic_terms %>%
    dplyr::mutate(
      ord = factor(topic, levels = sort(as.numeric(unique(topic)))),
      term = tidytext::reorder_within(term, beta, labeled_topic)
    ) %>%
    dplyr::arrange(ord) %>%
    dplyr::ungroup()

  levelt <- top_topic_terms %>%
    dplyr::arrange(as.numeric(topic)) %>%
    dplyr::distinct(labeled_topic) %>%
    dplyr::pull(labeled_topic)

  top_topic_terms$labeled_topic <- factor(top_topic_terms$labeled_topic, levels = levelt)

  ggplot_obj <- ggplot2::ggplot(
    top_topic_terms,
    ggplot2::aes(term, beta, fill = labeled_topic,
                 text = paste("Topic:", labeled_topic, "<br>", measure_label, ":", sprintf("%.3f", beta)))
  ) +
    ggplot2::geom_col(show.legend = FALSE, alpha = 0.8) +
    ggplot2::facet_wrap(~ labeled_topic, scales = "free", ncol = ncol, strip.position = "top") +
    tidytext::scale_x_reordered() +
    ggplot2::scale_y_continuous(labels = numform::ff_num(zero = 0, digits = 3)) +
    ggplot2::coord_flip() +
    ggplot2::xlab("") +
    ggplot2::ylab(ylab) +
    ggplot2::theme_minimal(base_size = 14) +
    ggplot2::theme(
      legend.position = "none",
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.line = element_line(color = "#3B3B3B", linewidth = 0.3),
      axis.ticks = element_line(color = "#3B3B3B", linewidth = 0.3),
      strip.text.x = element_text(
        size = 16,
        color = "#0c1f4a",
        lineheight = ifelse(width > 1000, 1.1, 1.2),
        margin = margin(l = 10, r = 10)
      ),
      panel.spacing.x = unit(ifelse(width > 1000, 2.2, 1.6), "lines"),
      panel.spacing.y = unit(ifelse(width > 1000, 2.2, 1.6), "lines"),
      axis.text.x = element_text(size = 16, color = "#3B3B3B", hjust = 1, margin = margin(r = 20)),
      axis.text.y = element_text(size = 16, color = "#3B3B3B", margin = margin(t = 20)),
      axis.title = element_text(size = 16, color = "#0c1f4a"),
      axis.title.x = element_text(margin = margin(t = 25)),
      axis.title.y = element_text(margin = margin(r = 25))
    )

  if (!is.null(colors)) {
    ggplot_obj <- ggplot_obj + ggplot2::scale_fill_manual(values = colors)
  }

  p <- plotly::ggplotly(ggplot_obj, height = height, width = width, tooltip = "text", ...) %>%
    plotly::layout(
      xaxis = list(
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif"),
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif")
      ),
      yaxis = list(
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif"),
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif")
      ),
      hoverlabel = list(
        bgcolor = "#0c1f4a",
        font = list(size = 16, color = "white", family = "Roboto, sans-serif"),
        bordercolor = "#0c1f4a",
        align = "left"
      )
    )

  # Skip title to avoid overlap with facet strip labels
  p <- p %>% plotly::layout(margin = list(t = 40, b = 60, l = 80, r = 100))

  p
}



#' @title Plot Per-Document Per-Topic Probabilities
#'
#' @description
#' This function generates a bar plot showing the prevalence of each topic across all documents.
#'
#' @param stm_model A fitted STM model object.
#'   where \code{stm_model} is a fitted Structural Topic Model created using \code{stm::stm()}.
#' @param gamma_data Optional pre-computed gamma data frame (default: NULL). If provided, used instead of stm_model.
#' @param top_n The number of topics to display, ordered by their mean prevalence.
#' @param height The height of the resulting Plotly plot, in pixels (default: 800).
#' @param width The width of the resulting Plotly plot, in pixels (default: 1000).
#' @param topic_labels Optional topic labels (default: NULL).
#' @param colors Optional color palette for topics (default: NULL).
#' @param verbose Logical, if TRUE, prints progress messages.
#' @param ... Further arguments passed to \code{tidytext::tidy}.
#'
#' @return A \code{ggplot} object showing a bar plot of topic prevalence.
#'   Topics are ordered by their
#' mean gamma value (average prevalence across documents).
#'
#' @family visualization
#' @export
#'
#' @examples
#' if (interactive()) {
#'
#' mydata <- TextAnalysisR::SpecialEduTech
#'
#'  united_tbl <- TextAnalysisR::unite_cols(
#'    mydata,
#'    listed_vars = c("title", "keyword", "abstract")
#'  )
#'
#'  tokens <- TextAnalysisR::prep_texts(united_tbl, text_field = "united_texts")
#'
#'  dfm_object <- quanteda::dfm(tokens)
#'
#'  out <- quanteda::convert(dfm_object, to = "stm")
#'
#' stm_15 <- stm::stm(
#'   data = out$meta,
#'   documents = out$documents,
#'   vocab = out$vocab,
#'   max.em.its = 75,
#'   init.type = "Spectral",
#'   K = 15,
#'   prevalence = ~ reference_type + s(year),
#'   verbose = TRUE)
#'
#' topic_probability_plot <- TextAnalysisR::plot_topic_probability(
#'  stm_model = stm_15,
#'  top_n = 10,
#'  height = 800,
#'  width = 1000,
#'  verbose = TRUE)
#'
#' print(topic_probability_plot)
#' }
plot_topic_probability <- function(stm_model = NULL,
                                   gamma_data = NULL,
                                   top_n = 10,
                                   height = 800,
                                   width = 1000,
                                   topic_labels = NULL,
                                   colors = NULL,
                                   verbose = TRUE,
                                   ...) {

    if (!is.null(gamma_data)) {
      gamma_terms <- gamma_data
      if (!is.null(top_n) && top_n < nrow(gamma_terms)) {
        gamma_terms <- gamma_terms %>%
          top_n(top_n, gamma)
      }
    } else if (!is.null(stm_model)) {
      gamma_td <- tidytext::tidy(stm_model, matrix="gamma", ...)
      gamma_terms <- gamma_td %>%
        group_by(topic) %>%
        summarise(gamma = mean(gamma)) %>%
        arrange(desc(gamma)) %>%
        mutate(topic = stats::reorder(topic, gamma)) %>%
        top_n(top_n, gamma)
    } else {
      stop("Either stm_model or gamma_data must be provided")
    }

    if (!is.null(topic_labels)) {
      if ("topic_label" %in% names(gamma_terms)) {
        gamma_terms <- gamma_terms %>%
          mutate(topic_display = topic_label)
      } else {
        gamma_terms <- gamma_terms %>%
          mutate(topic_display = topic)
      }
    } else {
      gamma_terms <- gamma_terms %>%
        mutate(topic_display = paste("Topic", topic))
    }

    if ("tt" %in% names(gamma_terms)) {
      gamma_terms <- gamma_terms %>%
        arrange(tt) %>%
        mutate(topic_display = factor(topic_display, levels = unique(topic_display)))
    } else {
      gamma_terms <- gamma_terms %>%
        mutate(topic_display = factor(topic_display, levels = unique(topic_display)))
    }

    hover_text <- if ("terms" %in% names(gamma_terms)) {
      paste0("Topic: ", gamma_terms$topic_display, "<br>Terms: ", gamma_terms$terms, "<br>Gamma: ", sprintf("%.3f", gamma_terms$gamma))
    } else {
      paste0("Topic: ", gamma_terms$topic_display, "<br>Gamma: ", sprintf("%.3f", gamma_terms$gamma))
    }

    ggplot_obj <- ggplot(gamma_terms, aes(x = topic_display, y = gamma, fill = topic_display,
                                          text = hover_text)) +
      geom_col(alpha = 0.8) +
      coord_flip() +
      scale_y_continuous(labels = numform::ff_num(zero = 0, digits = 2)) +
      xlab("") +
      ylab("Topic Proportion") +
      theme_minimal(base_size = 14) +
      theme(
        legend.position = "none",
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(color = "#3B3B3B", linewidth = 0.3),
        axis.ticks = element_line(color = "#3B3B3B", linewidth = 0.3),
        strip.text.x = element_text(size = 16, color = "#0c1f4a"),
        axis.text.x = element_text(size = 16, color = "#3B3B3B"),
        axis.text.y = element_text(size = 16, color = "#3B3B3B"),
        axis.title = element_text(size = 16, color = "#0c1f4a"),
        axis.title.x = element_text(margin = margin(t = 10)),
        axis.title.y = element_text(margin = margin(r = 10))
      )

    if (!is.null(colors)) {
      ggplot_obj <- ggplot_obj + scale_fill_manual(values = colors)
    }

    plotly::ggplotly(ggplot_obj, tooltip = "text", height = height, width = width) %>%
      plotly::layout(
        xaxis = list(
          tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif"),
          titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif")
        ),
        yaxis = list(
          tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif"),
          titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif")
        ),
        margin = list(t = 50, b = 40, l = 80, r = 40),
        hoverlabel = list(
          font = list(size = 16, family = "Roboto, sans-serif")
        )
      )
}

#' @title Plot Topic Effects for Categorical Variables
#'
#' @description
#' Creates a faceted plot showing how categorical variables affect topic proportions.
#'
#' @param effects_data Data frame with columns: topic, value, proportion, lower, upper
#' @param ncol Number of columns for faceting (default: 2)
#' @param height Plot height in pixels (default: 800)
#' @param width Plot width in pixels (default: 1000)
#' @param title Plot title (default: "Category Effects")
#'
#' @return A plotly object
#'
#' @family visualization
#' @export
plot_topic_effects_categorical <- function(effects_data,
                                           ncol = 2,
                                           height = 800,
                                           width = 1000,
                                           title = "Category Effects") {

  if (is.null(effects_data) || nrow(effects_data) == 0) {
    return(plotly::plot_ly(type = "scatter", mode = "markers") %>%
             plotly::add_annotations(
               text = "No categorical effects available.<br>Please run the effect estimation first.",
               x = 0.5, y = 0.5,
               showarrow = FALSE,
               font = list(size = 16, color = "#6c757d")
             ))
  }

  effects_data <- effects_data %>%
    dplyr::mutate(topic_label = paste("Topic", topic))

  ggplot_obj <- ggplot(effects_data, aes(x = value, y = proportion)) +
    facet_wrap(~topic_label, ncol = ncol, scales = "free") +
    scale_y_continuous(labels = numform::ff_num(zero = 0, digits = 3)) +
    xlab("") +
    ylab("Topic proportion") +
    geom_errorbar(
      aes(ymin = lower, ymax = upper),
      width = 0.1,
      linewidth = 0.5,
      color = "#337ab7"
    ) +
    geom_point(color = "#337ab7", size = 1.5) +
    theme_minimal(base_size = 14) +
    theme(
      legend.position = "none",
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.line = element_line(color = "#3B3B3B", linewidth = 0.3),
      axis.ticks = element_line(color = "#3B3B3B", linewidth = 0.3),
      strip.text.x = element_text(size = 16, color = "#0c1f4a", margin = margin(b = 30, t = 15)),
      axis.text.x = element_text(size = 16, color = "#3B3B3B", hjust = 1, margin = margin(t = 20)),
      axis.text.y = element_text(size = 16, color = "#3B3B3B", margin = margin(r = 20)),
      axis.title = element_text(size = 16, color = "#0c1f4a"),
      axis.title.x = element_text(margin = margin(t = 25)),
      axis.title.y = element_text(margin = margin(r = 25)),
      plot.margin = margin(t = 40, b = 40)
    )

  plotly::ggplotly(ggplot_obj, height = height, width = width) %>%
    plotly::layout(
      title = list(
        text = title,
        font = list(size = 18, color = "#0c1f4a", family = "Roboto, sans-serif"),
        x = 0.5,
        xref = "paper",
        xanchor = "center",
        y = 0.99,
        yref = "paper",
        yanchor = "top"
      ),
      xaxis = list(
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif"),
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif")
      ),
      yaxis = list(
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif"),
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif")
      ),
      margin = list(t = 100, b = 40, l = 80, r = 40),
      hoverlabel = list(
        font = list(size = 16, family = "Roboto, sans-serif")
      )
    )
}

#' @title Plot Topic Effects for Continuous Variables
#'
#' @description
#' Creates a faceted plot showing how continuous variables affect topic proportions.
#'
#' @param effects_data Data frame with columns: topic, value, proportion, lower, upper
#' @param ncol Number of columns for faceting (default: 2)
#' @param height Plot height in pixels (default: 800)
#' @param width Plot width in pixels (default: 1000)
#' @param title Plot title (default: "Continuous Variable Effects")
#'
#' @return A plotly object
#'
#' @family visualization
#' @export
plot_topic_effects_continuous <- function(effects_data,
                                          ncol = 2,
                                          height = 800,
                                          width = 1000,
                                          title = "Continuous Variable Effects") {

  effects_data <- effects_data %>%
    dplyr::mutate(topic_label = paste("Topic", topic))

  ggplot_obj <- ggplot(effects_data, aes(x = value, y = proportion)) +
    facet_wrap(~topic_label, ncol = ncol, scales = "free") +
    scale_y_continuous(labels = numform::ff_num(zero = 0, digits = 3)) +
    geom_ribbon(aes(ymin = lower, ymax = upper), fill = "#337ab7", alpha = 0.2) +
    geom_line(linewidth = 0.5, color = "#337ab7") +
    xlab("") +
    ylab("Topic proportion") +
    theme_minimal(base_size = 14) +
    theme(
      legend.position = "none",
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.line = element_line(color = "#3B3B3B", linewidth = 0.3),
      axis.ticks = element_line(color = "#3B3B3B", linewidth = 0.3),
      strip.text.x = element_text(size = 16, color = "#0c1f4a", margin = margin(b = 30, t = 15)),
      axis.text.x = element_text(size = 16, color = "#3B3B3B", hjust = 1, margin = margin(t = 20)),
      axis.text.y = element_text(size = 16, color = "#3B3B3B", margin = margin(r = 20)),
      axis.title = element_text(size = 16, color = "#0c1f4a"),
      axis.title.x = element_text(margin = margin(t = 25)),
      axis.title.y = element_text(margin = margin(r = 25)),
      plot.margin = margin(t = 40, b = 40)
    )

  plotly::ggplotly(ggplot_obj, height = height, width = width) %>%
    plotly::layout(
      title = list(
        text = title,
        font = list(size = 18, color = "#0c1f4a", family = "Roboto, sans-serif"),
        x = 0.5,
        xref = "paper",
        xanchor = "center",
        y = 0.99,
        yref = "paper",
        yanchor = "top"
      ),
      xaxis = list(
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif"),
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif")
      ),
      yaxis = list(
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif"),
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif")
      ),
      margin = list(t = 100, b = 40, l = 80, r = 40),
      hoverlabel = list(
        font = list(size = 16, family = "Roboto, sans-serif")
      )
    )
}


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
#'   texts <- c("machine learning", "deep learning", "artificial intelligence")
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


plot_semantic_topics <- function(topic_model,
                                    plot_type = "topics",
                                    height = 600,
                                    width = 800,
                                    title = NULL) {

  if (!requireNamespace("plotly", quietly = TRUE)) {
    stop("plotly package is required for visualization")
  }

  tryCatch({
    plot_obj <- switch(plot_type,
      "topics" = {
        topic_counts <- table(topic_model$topic_assignments)
        topic_data <- data.frame(
          topic = names(topic_counts),
          count = as.numeric(topic_counts),
          stringsAsFactors = FALSE
        )


        topic_data$keywords <- sapply(topic_data$topic, function(t) {
          keywords <- topic_model$topic_keywords[[t]]
          if (length(keywords) > 0) {
            paste(keywords[1:min(5, length(keywords))], collapse = ", ")
          } else {
            "No keywords"
          }
        })

        plotly::plot_ly(
          data = topic_data,
          x = ~topic,
          y = ~count,
          text = ~keywords,
          type = "bar",
          marker = list(color = "steelblue", opacity = 0.8),
          hovertemplate = "Topic: %{x}<br>Documents: %{y}<br>Keywords: %{text}<extra></extra>",
          width = width,
          height = height
        ) %>%
        plotly::layout(
          title = if (!is.null(title)) {
            list(
              text = title,
              font = list(size = 18, color = "#0c1f4a", family = "Roboto, sans-serif"),
              x = 0.5,
              xref = "paper",
              xanchor = "center",
              y = 0.98,
              yref = "paper",
              yanchor = "top"
            )
          } else {
            list(
              text = "Topic Distribution",
              font = list(size = 18, color = "#0c1f4a", family = "Roboto, sans-serif"),
              x = 0.5,
              xref = "paper",
              xanchor = "center",
              y = 0.98,
              yref = "paper",
              yanchor = "top"
            )
          },
          xaxis = list(title = "Topic"),
          yaxis = list(title = "Number of Documents"),
          showlegend = FALSE
        )
      },
      "hierarchy" = {
        if ("hclust_result" %in% names(topic_model)) {
          hclust_result <- topic_model$hclust_result
        } else {
          dist_matrix <- stats::dist(topic_model$embeddings, method = "euclidean")
          hclust_result <- stats::hclust(dist_matrix, method = "ward.D2")
        }

        dend <- stats::as.dendrogram(hclust_result)

        plotly::plot_ly() %>%
        plotly::add_segments(
          x = dend$members,
          y = dend$height,
          xend = dend$members,
          yend = 0,
          line = list(color = "black", width = 1)
        ) %>%
        plotly::layout(
          title = if (!is.null(title)) {
            list(
              text = title,
              font = list(size = 18, color = "#0c1f4a", family = "Roboto, sans-serif"),
              x = 0.5,
              xref = "paper",
              xanchor = "center",
              y = 0.98,
              yref = "paper",
              yanchor = "top"
            )
          } else {
            list(
              text = "Topic Hierarchy",
              font = list(size = 18, color = "#0c1f4a", family = "Roboto, sans-serif"),
              x = 0.5,
              xref = "paper",
              xanchor = "center",
              y = 0.98,
              yref = "paper",
              yanchor = "top"
            )
          },
          xaxis = list(title = "Documents", showticklabels = FALSE),
          yaxis = list(title = "Distance"),
          showlegend = FALSE,
          width = width,
          height = height
        )
      },
      "similarity" = {
        if ("similarity_matrix" %in% names(topic_model)) {
          similarity_matrix <- topic_model$similarity_matrix
        } else {
          sklearn_metrics <- reticulate::import("sklearn.metrics.pairwise")
          similarity_matrix <- sklearn_metrics$cosine_similarity(topic_model$embeddings)
          similarity_matrix <- as.matrix(similarity_matrix)
        }

        plotly::plot_ly(
          z = similarity_matrix,
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
              font = list(size = 18, color = "#0c1f4a", family = "Roboto, sans-serif"),
              x = 0.5,
              xref = "paper",
              xanchor = "center",
              y = 0.98,
              yref = "paper",
              yanchor = "top"
            )
          } else {
            list(
              text = "Document Similarity Heatmap",
              font = list(size = 18, color = "#0c1f4a", family = "Roboto, sans-serif"),
              x = 0.5,
              xref = "paper",
              xanchor = "center",
              y = 0.98,
              yref = "paper",
              yanchor = "top"
            )
          },
          xaxis = list(title = "Documents"),
          yaxis = list(title = "Documents")
        )
      },
      "evolution" = {
        stop("Topic evolution visualization requires temporal metadata")
      },
      stop("Unsupported plot type: ", plot_type)
    )

    return(plot_obj)

  }, error = function(e) {
    stop("Error creating semantic topic visualization: ", e$message)
  })
}


#' Plot Term Frequency Trends by Continuous Variable
#'
#' @description
#' Creates a faceted line plot showing how term frequencies vary across
#' a continuous variable (e.g., year, time period).
#'
#' @param term_data Data frame containing term frequencies with columns:
#'   continuous_var, term, and word_frequency
#' @param continuous_var Name of the continuous variable column
#' @param terms Character vector of terms to display (optional, filters if provided)
#' @param title Plot title (default: NULL, auto-generated)
#' @param height Plot height in pixels (default: 600)
#' @param width Plot width in pixels (default: NULL, auto)
#'
#' @return A plotly object with faceted line plots
#'
#' @family visualization
#' @export
#'
#' @examples
#' \dontrun{
#' term_df <- data.frame(
#'   year = rep(2010:2020, each = 3),
#'   term = rep(c("learning", "education", "technology"), 11),
#'   word_frequency = sample(10:100, 33, replace = TRUE)
#' )
#' plot_term_trends_continuous(term_df, "year", c("learning", "education"))
#' }
plot_term_trends_continuous <- function(term_data,
                                         continuous_var,
                                         terms = NULL,
                                         title = NULL,
                                         height = 600,
                                         width = NULL) {

  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required. Please install it.")
  }
  if (!requireNamespace("plotly", quietly = TRUE)) {
    stop("Package 'plotly' is required. Please install it.")
  }
  if (!requireNamespace("scales", quietly = TRUE)) {
    stop("Package 'scales' is required. Please install it.")
  }

  if (!continuous_var %in% names(term_data)) {
    stop("Continuous variable '", continuous_var, "' not found in data")
  }

  if (!"term" %in% names(term_data) && !"word" %in% names(term_data)) {
    stop("term or word column not found in data")
  }

  if ("word" %in% names(term_data) && !"term" %in% names(term_data)) {
    term_data$term <- term_data$word
  }

  if (!"word_frequency" %in% names(term_data) && !"count" %in% names(term_data)) {
    stop("word_frequency or count column not found in data")
  }

  if ("count" %in% names(term_data) && !"word_frequency" %in% names(term_data)) {
    term_data$word_frequency <- term_data$count
  }

  if (!is.null(terms)) {
    term_data <- term_data %>%
      dplyr::filter(term %in% terms) %>%
      dplyr::mutate(term = factor(term, levels = terms))
  }

  if (is.null(title)) {
    title <- paste("Term Frequency by", continuous_var)
  }

  p <- ggplot2::ggplot(
    term_data,
    ggplot2::aes(
      x = .data[[continuous_var]],
      y = word_frequency,
      group = term
    )
  ) +
    ggplot2::geom_point(color = "#337ab7", alpha = 0.6, size = 2.5) +
    ggplot2::geom_line(color = "#337ab7", alpha = 0.6, linewidth = 0.5) +
    ggplot2::facet_wrap(~term, scales = "free") +
    ggplot2::scale_y_continuous(labels = scales::number_format(accuracy = 1)) +
    ggplot2::labs(y = "Word Frequency", x = continuous_var) +
    ggplot2::theme_minimal(base_size = 14) +
    ggplot2::theme(
      legend.position = "none",
      axis.line = ggplot2::element_line(color = "#3B3B3B", linewidth = 0.3),
      axis.ticks = ggplot2::element_line(color = "#3B3B3B", linewidth = 0.3),
      strip.text.x = ggplot2::element_text(size = 16, color = "#0c1f4a", family = "Roboto"),
      axis.text.x = ggplot2::element_text(size = 16, color = "#3B3B3B", family = "Roboto"),
      axis.text.y = ggplot2::element_text(size = 16, color = "#3B3B3B", family = "Roboto"),
      axis.title = ggplot2::element_text(size = 16, color = "#0c1f4a", family = "Roboto"),
      axis.title.x = ggplot2::element_text(margin = ggplot2::margin(t = 15)),
      axis.title.y = ggplot2::element_text(margin = ggplot2::margin(r = 15)),
      plot.margin = ggplot2::margin(t = 5, r = 10, b = 25, l = 15)
    )

  plot_args <- list(p)
  if (!is.null(height)) plot_args$height <- height
  if (!is.null(width)) plot_args$width <- width

  p_plot <- do.call(plotly::ggplotly, plot_args)

  for (i in seq_along(p_plot$x$layout$annotations)) {
    p_plot$x$layout$annotations[[i]]$font <- list(
      size = 16,
      color = "#0c1f4a",
      family = "Roboto, sans-serif"
    )
  }

  axis_names <- names(p_plot$x$layout)
  for (axis_name in axis_names) {
    if (grepl("^xaxis", axis_name)) {
      p_plot$x$layout[[axis_name]]$tickfont <- list(
        size = 14,
        color = "#3B3B3B",
        family = "Roboto, sans-serif"
      )
      p_plot$x$layout[[axis_name]]$titlefont <- list(
        size = 16,
        color = "#0c1f4a",
        family = "Roboto, sans-serif"
      )
    }
    if (grepl("^yaxis", axis_name)) {
      p_plot$x$layout[[axis_name]]$tickfont <- list(
        size = 14,
        color = "#3B3B3B",
        family = "Roboto, sans-serif"
      )
      p_plot$x$layout[[axis_name]]$titlefont <- list(
        size = 16,
        color = "#0c1f4a",
        family = "Roboto, sans-serif"
      )
    }
  }

  p_plot %>%
    plotly::layout(
      margin = list(l = 80, r = 150, t = 40, b = 100),
      font = list(
        family = "Roboto, sans-serif",
        size = 14,
        color = "#3B3B3B"
      ),
      hoverlabel = list(
        font = list(size = 14, family = "Roboto, sans-serif")
      )
    ) %>%
    plotly::config(displayModeBar = TRUE)
}



#' Plot Part-of-Speech Tag Frequencies
#'
#' @description
#' Creates a bar plot showing the frequency distribution of part-of-speech tags.
#'
#' @param pos_data Data frame containing POS data with columns:
#'   \itemize{
#'     \item \code{pos}: Part-of-speech tag
#'     \item \code{n}: (optional) Pre-computed frequency count
#'   }
#'   If \code{n} is not present, frequencies will be computed from the data.
#' @param top_n Number of top POS tags to display (default: 20)
#' @param title Plot title (default: "Part-of-Speech Tag Frequency")
#' @param color Bar color (default: "#337ab7")
#' @param height Plot height in pixels (default: 500)
#' @param width Plot width in pixels (default: NULL for auto)
#'
#' @return A plotly object
#'
#' @family visualization
#' @export
#'
#' @examples
#' if (interactive()) {
#'   pos_df <- data.frame(
#'     pos = c("NOUN", "VERB", "ADJ", "ADV", "PRON"),
#'     n = c(500, 400, 250, 150, 100)
#'   )
#'   plot_pos_frequencies(pos_df)
#' }
plot_pos_frequencies <- function(pos_data,
                                  top_n = 20,
                                  title = "Part-of-Speech Tag Frequency",
                                  color = "#337ab7",
                                  height = 500,
                                  width = NULL) {

  if (!requireNamespace("plotly", quietly = TRUE)) {
    stop("Package 'plotly' is required. Please install it.")
  }

  if (is.null(pos_data) || nrow(pos_data) == 0) {
    return(plotly::plot_ly() %>%
      plotly::layout(
        xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
        yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
        annotations = list(
          list(
            text = "No POS data available",
            x = 0.5, y = 0.5,
            xref = "paper", yref = "paper",
            showarrow = FALSE,
            font = list(size = 16, color = "#6B7280", family = "Roboto")
          )
        )
      ))
  }

  if (!"n" %in% names(pos_data)) {
    pos_freq <- pos_data %>%
      dplyr::count(pos, sort = TRUE) %>%
      dplyr::slice_head(n = top_n)
  } else {
    pos_freq <- pos_data %>%
      dplyr::arrange(dplyr::desc(n)) %>%
      dplyr::slice_head(n = top_n)
  }

  plotly::plot_ly(
    data = pos_freq,
    x = ~stats::reorder(pos, n),
    y = ~n,
    type = "bar",
    marker = list(color = color),
    hoverinfo = "text",
    hovertext = ~paste0(pos, "\nFrequency: ", n),
    height = height,
    width = width
  ) %>%
    plotly::layout(
      title = list(
        text = title,
        font = list(size = 18, color = "#0c1f4a", family = "Roboto, sans-serif")
      ),
      xaxis = list(
        title = "POS Tag",
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif"),
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
      ),
      yaxis = list(
        title = "Frequency",
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif"),
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
      ),
      margin = list(b = 100, l = 60, r = 20, t = 60),
      hoverlabel = list(
        align = "left",
        font = list(size = 14, color = "white", family = "Roboto, sans-serif"),
        bgcolor = "#0c1f4a"
      )
    )
}


#' Plot Named Entity Frequencies
#'
#' @description
#' Creates a bar plot showing the frequency distribution of named entity types.
#'
#' @param entity_data Data frame containing entity data with columns:
#'   \itemize{
#'     \item \code{entity}: Named entity type (e.g., "PERSON", "ORG", "GPE")
#'     \item \code{n}: (optional) Pre-computed frequency count
#'   }
#'   If \code{n} is not present, frequencies will be computed from the data.
#' @param top_n Number of top entity types to display (default: 20)
#' @param title Plot title (default: "Named Entity Type Frequency")
#' @param color Bar color (default: "#10B981")
#' @param height Plot height in pixels (default: 500)
#' @param width Plot width in pixels (default: NULL for auto)
#'
#' @return A plotly object
#'
#' @family visualization
#' @export
#'
#' @examples
#' if (interactive()) {
#'   entity_df <- data.frame(
#'     entity = c("PERSON", "ORG", "GPE", "DATE", "MONEY"),
#'     n = c(300, 250, 200, 150, 100)
#'   )
#'   plot_entity_frequencies(entity_df)
#' }
plot_entity_frequencies <- function(entity_data,
                                     top_n = 20,
                                     title = "Named Entity Type Frequency",
                                     color = "#10B981",
                                     height = 500,
                                     width = NULL) {

  if (!requireNamespace("plotly", quietly = TRUE)) {
    stop("Package 'plotly' is required. Please install it.")
  }

  if (is.null(entity_data) || nrow(entity_data) == 0) {
    return(plotly::plot_ly(type = "scatter", mode = "markers") %>%
      plotly::layout(
        xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
        yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
        annotations = list(
          list(
            text = "No named entities found",
            x = 0.5, y = 0.5,
            xref = "paper", yref = "paper",
            showarrow = FALSE,
            font = list(size = 16, color = "#6B7280", family = "Roboto")
          )
        )
      ))
  }

  if (!"n" %in% names(entity_data)) {
    entity_freq <- entity_data %>%
      dplyr::count(entity, sort = TRUE) %>%
      dplyr::slice_head(n = top_n)
  } else {
    entity_freq <- entity_data %>%
      dplyr::arrange(dplyr::desc(n)) %>%
      dplyr::slice_head(n = top_n)
  }

  plotly::plot_ly(
    data = entity_freq,
    x = ~stats::reorder(entity, n),
    y = ~n,
    type = "bar",
    marker = list(color = color),
    hoverinfo = "text",
    hovertext = ~paste0(entity, "\nFrequency: ", n),
    height = height,
    width = width
  ) %>%
    plotly::layout(
      title = list(
        text = title,
        font = list(size = 18, color = "#0c1f4a", family = "Roboto, sans-serif")
      ),
      xaxis = list(
        title = "",
        tickangle = -45,
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
      ),
      yaxis = list(
        title = "Frequency",
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif"),
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
      ),
      margin = list(b = 150, l = 60, r = 20, t = 60),
      hoverlabel = list(
        align = "left",
        font = list(size = 16, color = "white", family = "Roboto, sans-serif"),
        bgcolor = "#0c1f4a"
      )
    )
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


#' Plot Cluster Top Terms
#'
#' @description
#' Creates a horizontal bar plot showing the top terms in a cluster or document group.
#'
#' @param terms Named numeric vector of term frequencies, or data frame with
#'   'term' and 'frequency' columns
#' @param cluster_id Cluster identifier for the title (default: NULL)
#' @param title Custom title (default: NULL, auto-generated from cluster_id)
#' @param n_terms Number of top terms to display (default: 10)
#' @param color Bar color (default: "#337ab7")
#' @param height Plot height in pixels (default: 500)
#' @param width Plot width in pixels (default: NULL for auto)
#'
#' @return A plotly object
#'
#' @family visualization
#' @export
plot_cluster_terms <- function(terms,
                                cluster_id = NULL,
                                title = NULL,
                                n_terms = 10,
                                color = "#337ab7",
                                height = 500,
                                width = NULL) {

  if (!requireNamespace("plotly", quietly = TRUE)) {
    stop("Package 'plotly' is required. Please install it.")
  }

  if (is.null(terms) || length(terms) == 0) {
    return(create_empty_plot_message("No terms available for this cluster"))
  }

  if (is.data.frame(terms)) {
    if (!all(c("term", "frequency") %in% names(terms))) {
      stop("Data frame must have 'term' and 'frequency' columns")
    }
    terms <- terms %>%
      dplyr::arrange(dplyr::desc(frequency)) %>%
      dplyr::slice_head(n = n_terms)
    term_names <- terms$term
    term_values <- terms$frequency
  } else {
    top_terms <- utils::head(sort(terms, decreasing = TRUE), n_terms)
    term_names <- names(top_terms)
    term_values <- as.numeric(top_terms)
  }

  if (is.null(title)) {
    title <- if (!is.null(cluster_id)) {
      paste("Top Terms in Cluster", cluster_id)
    } else {
      "Top Terms"
    }
  }

  plotly::plot_ly(
    x = term_values,
    y = term_names,
    type = "bar",
    orientation = "h",
    marker = list(color = color),
    hovertemplate = "%{y}<br>Frequency: %{x:.4f}<extra></extra>",
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
        title = list(text = "Frequency"),
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif"),
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
      ),
      yaxis = list(
        title = "",
        categoryorder = "total ascending",
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
      ),
      font = list(family = "Roboto, sans-serif", size = 16, color = "#3B3B3B"),
      hoverlabel = list(
        align = "left",
        font = list(size = 16, color = "white", family = "Roboto, sans-serif"),
        bgcolor = "#0c1f4a"
      ),
      margin = list(l = 120, r = 40, t = 80, b = 60)
    ) %>%
    plotly::config(displayModeBar = TRUE)
}
