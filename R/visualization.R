#' @importFrom utils modifyList
#' @importFrom stats cor
#' @importFrom quanteda.textstats textstat_frequency
NULL

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
        titlefont = list(size = 16, color = "#0c1f4a", family = "Montserrat, sans-serif")
      ),
      yaxis = list(
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif"),
        titlefont = list(size = 16, color = "#0c1f4a", family = "Montserrat, sans-serif")
      ),
      hoverlabel = list(
        bgcolor = "#0c1f4a",
        font = list(size = 15, color = "white", family = "Roboto, sans-serif"),
        bordercolor = "#0c1f4a",
        align = "left"
      )
    )

  if (!is.null(title)) {
    p <- p %>% plotly::layout(
      title = list(
        text = title,
        font = list(size = 18, color = "#0c1f4a", family = "Montserrat, sans-serif"),
        x = 0.5,
        xref = "paper",
        xanchor = "center",
        y = 0.98,
        yref = "paper",
        yanchor = "top"
      ),
      margin = list(t = 100, b = 60, l = 80, r = 100)
    )
  } else {
    p <- p %>% plotly::layout(margin = list(t = 40, b = 60, l = 80, r = 100))
  }

  p
}


#' @title Plot Word Frequency
#'
#' @description
#' Creates a bar plot showing the most frequent words in a document-feature matrix (dfm).
#'
#' @param dfm_object A document-feature matrix created by quanteda::dfm().
#' @param n The number of top words to display (default: 20).
#' @param height The height of the resulting Plotly plot, in pixels (default: 800).
#' @param width The width of the resulting Plotly plot, in pixels (default: 1000).
#' @param ... Additional arguments passed to plotly::ggplotly().
#'
#' @return A plotly object showing word frequency.
#'
#' @export
#'
#' @examples
#' if (interactive()) {
#'   texts <- c("mathematics technology", "education technology", "learning support")
#'   dfm <- quanteda::dfm(quanteda::tokens(texts))
#'   plot <- plot_word_frequency(dfm, n = 5)
#'   print(plot)
#' }
plot_word_frequency <- function(dfm_object,
                                n = 20,
                                height = NULL,
                                width = NULL,
                                ...) {

  if (!inherits(dfm_object, "dfm")) {
    stop("Input must be a quanteda dfm object")
  }

  freq_df <- quanteda.textstats::textstat_frequency(dfm_object, n = n) %>%
    dplyr::mutate(
      feature = stats::reorder(feature, frequency)
    )

  ggplot_obj <- ggplot2::ggplot(freq_df,
                                ggplot2::aes(x = feature, y = frequency,
                                            text = paste("Word:", feature,
                                                       "<br>Frequency:", frequency))) +
    ggplot2::geom_point(color = "#0c1f4a", size = 2.5, alpha = 0.9) +
    ggplot2::scale_x_discrete(expand = ggplot2::expansion(add = 0.5)) +
    ggplot2::coord_flip() +
    ggplot2::labs(x = "", y = "Frequency") +
    ggplot2::theme_minimal(base_size = 14) +
    ggplot2::theme(
      panel.grid.major.x = ggplot2::element_blank(),
      panel.grid.major.y = ggplot2::element_line(color = "#E0E0E0", linewidth = 0.3),
      panel.grid.minor = ggplot2::element_blank(),
      axis.line.x = ggplot2::element_line(color = "#3B3B3B", linewidth = 0.3),
      axis.line.y = ggplot2::element_blank(),
      axis.ticks.x = ggplot2::element_line(color = "#3B3B3B", linewidth = 0.3),
      axis.ticks.y = ggplot2::element_blank(),
      axis.text.x = ggplot2::element_text(size = 16, color = "#3B3B3B", margin = ggplot2::margin(t = 3)),
      axis.text.y = ggplot2::element_text(size = 16, color = "#3B3B3B", margin = ggplot2::margin(r = 3)),
      axis.title = ggplot2::element_text(size = 16, color = "#0c1f4a", face = "bold"),
      axis.title.x = ggplot2::element_text(margin = ggplot2::margin(t = 5)),
      axis.title.y = ggplot2::element_text(margin = ggplot2::margin(r = 5)),
      plot.margin = ggplot2::margin(t = 5, r = 10, b = 5, l = 5)
    )

  plotly::ggplotly(ggplot_obj, height = height, width = width, tooltip = "text", ...) %>%
    plotly::layout(
      autosize = TRUE,
      margin = list(t = 20, b = 80, l = 80, r = 20),
      xaxis = list(
        tickfont = list(size = 16, color = "#3B3B3B"),
        titlefont = list(size = 16, color = "#0c1f4a")
      ),
      yaxis = list(
        tickfont = list(size = 16, color = "#3B3B3B"),
        titlefont = list(size = 16, color = "#0c1f4a")
      ),
      hoverlabel = list(
        bgcolor = "#0c1f4a",
        font = list(size = 15, color = "white"),
        bordercolor = "#0c1f4a",
        align = "left"
      )
    )
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
        strip.text.x = element_text(size = 16, color = "#3B3B3B"),
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
          titlefont = list(size = 16, color = "#3B3B3B", family = "Montserrat, sans-serif")
        ),
        yaxis = list(
          tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif"),
          titlefont = list(size = 16, color = "#3B3B3B", family = "Montserrat, sans-serif")
        ),
        margin = list(t = 50, b = 40, l = 80, r = 40),
        hoverlabel = list(
          font = list(size = 15)
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
      strip.text.x = element_text(size = 16, color = "#3B3B3B", margin = margin(b = 30, t = 15)),
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
        font = list(size = 18, color = "#0c1f4a", family = "Montserrat"),
        x = 0.5,
        xref = "paper",
        xanchor = "center",
        y = 0.99,
        yref = "paper",
        yanchor = "top"
      ),
      xaxis = list(
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif"),
        titlefont = list(size = 16, color = "#3B3B3B", family = "Montserrat, sans-serif")
      ),
      yaxis = list(
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif"),
        titlefont = list(size = 16, color = "#3B3B3B", family = "Montserrat, sans-serif")
      ),
      margin = list(t = 100, b = 40, l = 80, r = 40),
      hoverlabel = list(
        font = list(size = 15)
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
      strip.text.x = element_text(size = 16, color = "#3B3B3B", margin = margin(b = 30, t = 15)),
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
        font = list(size = 18, color = "#0c1f4a", family = "Montserrat"),
        x = 0.5,
        xref = "paper",
        xanchor = "center",
        y = 0.99,
        yref = "paper",
        yanchor = "top"
      ),
      xaxis = list(
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif"),
        titlefont = list(size = 16, color = "#3B3B3B", family = "Montserrat, sans-serif")
      ),
      yaxis = list(
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif"),
        titlefont = list(size = 16, color = "#3B3B3B", family = "Montserrat, sans-serif")
      ),
      margin = list(t = 100, b = 40, l = 80, r = 40),
      hoverlabel = list(
        font = list(size = 15)
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
              font = list(size = 20, color = "#0c1f4a", family = "Roboto"),
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
              font = list(size = 20, color = "#0c1f4a", family = "Roboto"),
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
            titlefont = list(size = 18, color = "#0c1f4a", family = "Roboto"),
            tickfont = list(size = 18, color = "#3B3B3B", family = "Roboto")
          ),
          yaxis = list(
            title = "Documents",
            titlefont = list(size = 18, color = "#0c1f4a", family = "Roboto"),
            tickfont = list(size = 18, color = "#3B3B3B", family = "Roboto")
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
              font = list(size = 20, color = "#0c1f4a", family = "Roboto"),
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
              font = list(size = 20, color = "#0c1f4a", family = "Roboto"),
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
            titlefont = list(size = 18, color = "#0c1f4a", family = "Roboto"),
            tickfont = list(size = 18, color = "#3B3B3B", family = "Roboto")
          ),
          yaxis = list(
            title = paste("Component 2",
                          if (!is.null(analysis_result) && !is.null(analysis_result$variance_explained) &&
                              length(analysis_result$variance_explained) > 1)
                            paste0("(", round(analysis_result$variance_explained[2] * 100, 1), "%)")
                          else ""),
            titlefont = list(size = 18, color = "#0c1f4a", family = "Roboto"),
            tickfont = list(size = 18, color = "#3B3B3B", family = "Roboto")
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
                font = list(size = 20, color = "#0c1f4a", family = "Roboto"),
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
                font = list(size = 20, color = "#0c1f4a", family = "Roboto"),
                x = 0.5,
                xref = "paper",
                xanchor = "center",
                y = 0.98,
                yref = "paper",
                yanchor = "top"
              )
            },
            xaxis = list(
              title = "Document Index",
              titlefont = list(size = 18, family = "Roboto"),
              tickfont = list(size = 18, family = "Roboto")
            ),
            yaxis = list(
              title = "Cluster",
              titlefont = list(size = 18, family = "Roboto"),
              tickfont = list(size = 18, family = "Roboto")
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
                font = list(size = 20, color = "#0c1f4a", family = "Roboto"),
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
                font = list(size = 20, color = "#0c1f4a", family = "Roboto"),
                x = 0.5,
                xref = "paper",
                xanchor = "center",
                y = 0.98,
                yref = "paper",
                yanchor = "top"
              )
            },
            xaxis = list(
              title = "Component 1",
              titlefont = list(size = 18, family = "Roboto"),
              tickfont = list(size = 18, family = "Roboto")
            ),
            yaxis = list(
              title = "Component 2",
              titlefont = list(size = 18, family = "Roboto"),
              tickfont = list(size = 18, family = "Roboto")
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
              font = list(size = 18, color = "#0c1f4a", family = "Montserrat"),
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
              font = list(size = 18, color = "#0c1f4a", family = "Montserrat"),
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
              font = list(size = 18, color = "#0c1f4a", family = "Montserrat"),
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
              font = list(size = 18, color = "#0c1f4a", family = "Montserrat"),
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
              font = list(size = 18, color = "#0c1f4a", family = "Montserrat"),
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
              font = list(size = 18, color = "#0c1f4a", family = "Montserrat"),
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
