
#' @title Process Files
#'
#' @description
#' This function processes different types of files and text input based on the dataset choice.
#'
#' @param dataset_choice A character string indicating the dataset choice.
#' @param file_info A data frame containing file information with a column named 'filepath' (default: NULL).
#' @param text_input A character string containing text input (default: NULL).
#'
#' @return A data frame containing the processed data.
#'
#' @importFrom utils read.csv
#'
#' @export
#'
#' @examples
#' if (interactive()) {
#'   mydata <- TextAnalysisR::process_files(dataset_choice = "Upload an Example Dataset")
#'   head(mydata)
#'
#'   file_info <- data.frame(filepath = "inst/extdata/SpecialEduTech.xlsx")
#'   mydata <- TextAnalysisR::process_files(dataset_choice = "Upload Your File",
#'                                           file_info = file_info)
#'   head(mydata)
#'
#'   text_input <- paste0("The purpose of this study was to conduct a content analysis of ",
#'                        "research on technology use.")
#'   mydata <- TextAnalysisR::process_files(dataset_choice = "Copy and Paste Text",
#'                                           text_input = text_input)
#'   head(mydata)
#' }
process_files <- function(dataset_choice, file_info = NULL, text_input = NULL) {

  if (!requireNamespace("readxl", quietly = TRUE) ||
      !requireNamespace("pdftools", quietly = TRUE) ||
      !requireNamespace("officer", quietly = TRUE)) {
    stop(
      "The 'readxl', 'pdftools' and 'officer' packages are required for this functionality. ",
      "Please install them using install.packages(c('readxl', 'pdftools', 'officer'))."
    )
  }

  if (dataset_choice == "Upload an Example Dataset") {
    data <- TextAnalysisR::SpecialEduTech
    data <- tibble::as_tibble(data)
  } else if (dataset_choice == "Copy and Paste Text") {
    if (is.null(text_input)) stop("No text provided")
    data <- tibble::tibble(text = text_input)
  } else if (dataset_choice == "Upload Your File") {
    if (is.null(file_info)) stop("No file provided")

    data_list <- lapply(seq_len(nrow(file_info)), function(i) {
      filepath <- file_info$filepath[i]
      ext <- tolower(tools::file_ext(filepath))

      df <- tryCatch({
        if (ext %in% c("xlsx", "xls", "xlsm")) {
          readxl::read_excel(filepath, col_names = TRUE)
        } else if (ext == "csv") {
          read.csv(filepath, header = TRUE, stringsAsFactors = FALSE)
        } else if (ext == "pdf") {
          tryCatch({
            pages <- pdftools::pdf_text(filepath)
            lines <- unlist(lapply(pages, function(page) {
              lines <- strsplit(page, "\n")[[1]]
              trimws(lines)
            }))
            lines <- lines[lines != ""]
            data.frame(text = lines, stringsAsFactors = FALSE)
          }, error = function(e) {
            message("Error processing PDF file: ", filepath, ": ", e$message)
            data.frame(text = "", stringsAsFactors = FALSE)
          })
        } else if (ext == "docx") {
          doc <- officer::read_docx(filepath)
          doc_summary <- officer::docx_summary(doc)
          lines <- unlist(lapply(doc_summary$text, function(x) {
            trimws(unlist(strsplit(x, "\n")))
          }))
          lines <- lines[lines != ""]
          data.frame(text = lines, stringsAsFactors = FALSE)
        } else if (ext == "txt") {
          tryCatch({
            lines <- readLines(filepath, warn = FALSE, encoding = "UTF-8")
            lines <- trimws(lines)
            lines <- lines[lines != ""]
            data.frame(text = lines, stringsAsFactors = FALSE)
          }, error = function(e) {
            message("Error processing TXT file: ", filepath, ": ", e$message)
            data.frame(text = "", stringsAsFactors = FALSE)
          })
        } else {
          stop("Unsupported file extension: ", ext)
        }
      }, error = function(e) {
        message("Error processing file: ", filepath, ": ", e$message)
        NULL
      })

      if (is.null(df)) return(NULL)
      tibble::as_tibble(df)
    })

    data_list <- Filter(Negate(is.null), data_list)
    data <- dplyr::bind_rows(data_list)
  }

  return(data)
}



#' @title Unite Text Columns
#'
#' @description
#' This function unites specified text columns in a data frame into a single column named "united_texts" while retaining the original columns.
#'
#' @param df A data frame that contains text data.
#' @param listed_vars A character vector of column names to be united into "united_texts".
#'
#' @return A data frame with a new column "united_texts" created by uniting the specified variables.
#'
#' @export
#'
#' @examples
#' if (interactive()) {
#'   df <- TextAnalysisR::SpecialEduTech
#'
#'   united_tbl <- TextAnalysisR::unite_text_cols(df, listed_vars = c("title", "keyword", "abstract"))
#'   print(united_tbl)
#' }
unite_text_cols <- function(df, listed_vars) {
  united_texts_tbl <- df %>%
    dplyr::select(all_of(unname(listed_vars))) %>%
    tidyr::unite(col = "united_texts", sep = " ", remove = TRUE)

  docvar_tbl <- df

  united_tbl <- dplyr::bind_cols(united_texts_tbl, docvar_tbl)

  return(united_tbl)
}


#' @title Preprocess Text Data
#'
#' @description
#' Preprocesses text data by:
#' - Constructing a corpus
#' - Tokenizing text into words
#' - Converting to lowercase
#' - Specifying a minimum token length.
#'
#' Typically used before constructing a dfm and fitting an STM model.
#'
#' @param united_tbl A data frame that contains text data.
#' @param text_field The name of the column that contains the text data.
#' @param min_char The minimum number of characters for a token to be included (default: 2).
#' @param remove_punct Logical; remove punctuation from the text (default: TRUE).
#' @param remove_symbols Logical; remove symbols from the text (default: TRUE).
#' @param remove_numbers Logical; remove numbers from the text (default: TRUE).
#' @param remove_url Logical; remove URLs from the text (default: TRUE).
#' @param remove_separators Logical; remove separators from the text (default: TRUE).
#' @param split_hyphens Logical; split hyphenated words into separate tokens (default: TRUE).
#' @param split_tags Logical; split tags into separate tokens (default: TRUE).
#' @param include_docvars Logical; include document variables in the tokens object (default: TRUE).
#' @param keep_acronyms Logical; keep acronyms in the text (default: FALSE).
#' @param padding Logical; add padding to the tokens object (default: FALSE).
#' @param verbose Logical; print verbose output (default: FALSE).
#' @param ... Additional arguments passed to \code{quanteda::tokens}.
#'
#' @return A \code{tokens} object that contains the preprocessed text data.
#'
#' @import quanteda
#'
#' @export
#'
#' @examples
#' if (interactive()) {
#' df <- TextAnalysisR::SpecialEduTech
#'
#' united_tbl <- TextAnalysisR::unite_text_cols(df, listed_vars = c("title", "keyword", "abstract"))
#'
#' tokens <- TextAnalysisR::preprocess_texts(united_tbl,
#'                                          text_field = "united_texts",
#'                                          min_char = 2,
#'                                          remove_punct = TRUE,
#'                                          remove_symbols = TRUE,
#'                                          remove_numbers = TRUE,
#'                                          remove_url = TRUE,
#'                                          remove_separators = TRUE,
#'                                          split_hyphens = TRUE,
#'                                          split_tags = TRUE,
#'                                          include_docvars = TRUE,
#'                                          keep_acronyms = FALSE,
#'                                          padding = FALSE,
#'                                          verbose = FALSE)
#' print(tokens)
#' }
preprocess_texts <- function(united_tbl,
                             text_field = "united_texts",
                             min_char = 2,
                             remove_punct = TRUE,
                             remove_symbols = TRUE,
                             remove_numbers = TRUE,
                             remove_url = TRUE,
                             remove_separators = TRUE,
                             split_hyphens = TRUE,
                             split_tags = TRUE,
                             include_docvars = TRUE,
                             keep_acronyms = FALSE,
                             padding = FALSE,
                             verbose = FALSE,
                             ...) {

  corp <- quanteda::corpus(united_tbl, text_field = text_field)

  toks <- tokens(corp,
                 what = "word",
                 remove_punct = remove_punct,
                 remove_symbols = remove_symbols,
                 remove_numbers = remove_numbers,
                 remove_url = remove_url,
                 remove_separators = remove_separators,
                 split_hyphens = split_hyphens,
                 split_tags = split_tags,
                 include_docvars = include_docvars,
                 keep_acronyms = keep_acronyms,
                 padding = padding,
                 verbose = verbose)

  toks_lower <- quanteda::tokens_tolower(toks, keep_acronyms = keep_acronyms)

  tokens <- quanteda::tokens_select(toks_lower,
                                    min_nchar = min_char,
                                    verbose = FALSE)

  return(tokens)
}


#' @title Detect Multi-Word Expressions
#'
#' @description
#' This function detects multi-word expressions (collocations) of specified sizes that appear at least a specified number of times in the provided tokens.
#'
#' @param tokens A \code{tokens} object from the \code{quanteda} package.
#' @param size A numeric vector specifying the sizes of the collocations to detect (default: 2:5).
#' @param min_count The minimum number of occurrences for a collocation to be considered (default: 2).
#'
#' @return A character vector of detected collocations.
#'
#' @export
#'
#' @examples
#' if (interactive()) {
#'   df <- TextAnalysisR::SpecialEduTech
#'
#'   united_tbl <- TextAnalysisR::unite_text_cols(df, listed_vars = c("title", "keyword", "abstract"))
#'
#'   tokens <- TextAnalysisR::preprocess_texts(united_tbl, text_field = "united_texts")
#'
#'   collocations <- TextAnalysisR::detect_multi_word_expressions(tokens, size = 2:5, min_count = 2)
#'   print(collocations)
#' }
detect_multi_word_expressions <- function(tokens, size = 2:5, min_count = 2) {
  tstat <- quanteda.textstats::textstat_collocations(tokens, size = size, min_count = min_count)
  tstat_collocation <- tstat$collocation
  return(tstat_collocation)
}


#' @title Plot Word Frequency
#'
#' @description
#' Given a document-feature matrix (dfm), this function computes the most frequent terms
#' and creates a ggplot-based visualization of term frequencies.
#'
#' @param dfm_object A \code{quanteda} dfm object.
#' @param n The number of top terms to display (default: 20).
#' @param ... Further arguments passed to \code{quanteda.textstats::textstat_frequency}.
#'
#' @return A \code{ggplot} object visualizing the top terms by their frequency. The plot
#' shows each term on one axis and frequency on the other, with points representing their
#' observed frequencies.
#'
#' @export
#'
#' @examples
#' if (interactive()) {
#'   df <- TextAnalysisR::SpecialEduTech
#'
#'   united_tbl <- TextAnalysisR::unite_text_cols(df, listed_vars = c("title", "keyword", "abstract"))
#'
#'   tokens <- TextAnalysisR::preprocess_texts(united_tbl, text_field = "united_texts")
#'
#'   dfm_object <- quanteda::dfm(tokens)
#'
#'   word_frequency_plot <- TextAnalysisR::plot_word_frequency(dfm_object, n = 20)
#'   print(word_frequency_plot)
#' }
plot_word_frequency <-
  function(dfm_object, n = 20, ...) {
    word_freq <- quanteda.textstats::textstat_frequency(dfm_object, n = n, ...)
    word_frequency_plot <- ggplot(word_freq, aes(x = reorder(feature, frequency), y = frequency)) +
      geom_point(colour = "#5f7994", size = 1) +
      coord_flip() +
      labs(x = NULL, y = "Word frequency") +
      theme_minimal(base_size = 11) +
      theme(
        legend.position = "none",
        panel.grid.minor = element_blank(),
        axis.line = element_line(color = "#3B3B3B", linewidth = 0.3),
        axis.ticks = element_line(color = "#3B3B3B", linewidth = 0.3),
        strip.text.x = element_text(size = 11, color = "#3B3B3B"),
        axis.text.x = element_text(size = 11, color = "#3B3B3B"),
        axis.text.y = element_text(size = 11, color = "#3B3B3B"),
        axis.title = element_text(size = 11, color = "#3B3B3B"),
        axis.title.x = element_text(margin = margin(t = 9)),
        axis.title.y = element_text(margin = margin(r = 9))
      )
    return(word_frequency_plot)
  }


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
#'
#' @return A list containing the Plotly plot, a data frame of the network layout, and the igraph graph object.
#'
#' @importFrom igraph graph_from_data_frame V vcount degree betweenness closeness eigen_centrality layout_with_fr
#' @importFrom plotly plot_ly add_segments add_markers layout add_trace
#' @importFrom dplyr count filter mutate select group_by summarise
#' @importFrom tibble as_tibble
#' @importFrom tidytext tidy
#' @importFrom widyr pairwise_count
#' @importFrom scales rescale
#' @importFrom stats quantile
#' @importFrom DT datatable
#' @importFrom shiny showNotification
#' @importFrom rlang sym
#' @importFrom utils head
#' @importFrom grDevices colorRampPalette
#'
#' @export
#'
#' @examples
#' if (interactive()) {
#'   df <- TextAnalysisR::SpecialEduTech
#'
#'   united_tbl <- TextAnalysisR::unite_text_cols(df, listed_vars = c("title", "keyword", "abstract"))
#'
#'   tokens <- TextAnalysisR::preprocess_texts(united_tbl, text_field = "united_texts")
#'
#'   dfm_object <- quanteda::dfm(tokens)
#'
#'   word_co_occurrence_network_results <- TextAnalysisR::word_co_occurrence_network(
#'                                         dfm_object,
#'                                         doc_var = "reference_type",
#'                                         co_occur_n = 50,
#'                                         top_node_n = 30,
#'                                         nrows = 1,
#'                                         height = 800,
#'                                         width = 900)
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
                                       width = 900) {

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
    print(paste("doc_var has", length(docvar_levels), "levels:", paste(docvar_levels, collapse = ", ")))
  } else {
    docvar_levels <- NULL
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
          style = "font-weight: bold; text-align: center; font-size: 14pt;"
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
                                                  buttons = c('copy', 'csv', 'excel', 'pdf', 'print'))) %>%
      DT::formatStyle(columns = colnames(summary_df), `font-size` = "16px")

    htmltools::tagList(
      htmltools::tags$div(
        style = "margin-bottom: 20px;",
        htmltools::tags$p(
          group_label,
          style = "font-weight: bold; text-align: center; font-size: 14pt;"
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
    annotations <- lapply(1:nrow(top_nodes), function(i) {
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
#' @param common_term_n Minimum number of common terms for filtering terms (default: 30).
#' @param corr_n Minimum correlation value for filtering terms (default: 0.4).
#' @param top_node_n Number of top nodes to display (default: 40).
#' @param nrows Number of rows to display in the table (default: 1).
#' @param height The height of the resulting Plotly plot, in pixels (default: 1000).
#' @param width The width of the resulting Plotly plot, in pixels (default: 900).
#'
#' @return A list containing the Plotly plot, a data frame of the network layout, and the igraph graph object.
#'
#' @importFrom igraph graph_from_data_frame V vcount degree betweenness closeness eigen_centrality layout_with_fr
#' @importFrom plotly plot_ly add_segments add_markers layout add_trace
#' @importFrom dplyr count filter mutate select group_by summarise
#' @importFrom tibble as_tibble
#' @importFrom tidytext tidy
#' @importFrom widyr pairwise_cor
#' @importFrom scales rescale
#' @importFrom stats quantile
#' @importFrom DT datatable
#' @importFrom shiny showNotification
#' @importFrom utils head
#' @importFrom grDevices colorRampPalette
#'
#' @export
#'
#' @examples
#' if (interactive()) {
#'   df <- TextAnalysisR::SpecialEduTech
#'
#'   united_tbl <- TextAnalysisR::unite_text_cols(df, listed_vars = c("title", "keyword", "abstract"))
#'
#'   tokens <- TextAnalysisR::preprocess_texts(united_tbl, text_field = "united_texts")
#'
#'   dfm_object <- quanteda::dfm(tokens)
#'
#'   word_correlation_network_results <- TextAnalysisR::word_correlation_network(
#'                                       dfm_object,
#'                                       doc_var = "reference_type",
#'                                       common_term_n = 30,
#'                                       corr_n = 0.4,
#'                                       top_node_n = 40,
#'                                       nrows = 1,
#'                                       height = 1000,
#'                                       width = 900)
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
                                     width = 900) {

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
    print(paste("doc_var has", length(docvar_levels), "levels:", paste(docvar_levels, collapse = ", ")))
  } else {
    docvar_levels <- NULL
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
          style = "font-weight: bold; text-align: center; font-size: 14pt;"
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
                                                  buttons = c('copy', 'csv', 'excel', 'pdf', 'print'))) %>%
      DT::formatStyle(columns = colnames(summary_df), `font-size` = "16px")

    htmltools::tagList(
      htmltools::tags$div(
        style = "margin-bottom: 20px;",
        htmltools::tags$p(
          group_label,
          style = "font-weight: bold; text-align: center; font-size: 14pt;"
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
      group_by(line_group) %>%
      dplyr::summarise(
        min_corr = min(correlation, na.rm = TRUE),
        max_corr = max(correlation, na.rm = TRUE)
      ) %>%
      dplyr::mutate(label = paste0("Correlation: ", round(min_corr, 2), " - ", round(max_corr, 2))) %>%
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
    annotations <- lapply(1:nrow(top_nodes), function(i) {
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


#' @title Analyze and Visualize Word Frequencies Across a Continuous Variable
#'
#' @description
#' This function analyzes and visualizes word frequencies across a continuous variable.
#'
#' @param dfm_object A quanteda document-feature matrix (dfm).
#' @param continuous_variable A continuous variable in the metadata.
#' @param selected_terms A vector of terms to analyze trends for.
#' @param height The height of the resulting Plotly plot, in pixels (default: 500).
#' @param width The width of the resulting Plotly plot, in pixels (default: 900).
#'
#' @return A list containing Plotly objects and tables with the results.
#'
#' @details This function requires a fitted STM model object and a quanteda dfm object.
#' The continuous variable should be a column in the metadata of the dfm object.
#' The selected terms should be a vector of terms to analyze trends for.
#' The required packages are 'htmltools', 'splines', and 'broom' (plus additional ones loaded internally).
#'
#' @importFrom stats glm reformulate binomial
#' @importFrom plotly ggplotly plot_ly add_trace layout
#' @importFrom DT datatable
#'
#' @export
#'
#' @examples
#' if (interactive()) {
#'   df <- TextAnalysisR::SpecialEduTech
#'
#'   united_tbl <- TextAnalysisR::unite_text_cols(df, listed_vars = c("title", "keyword", "abstract"))
#'
#'   tokens <- TextAnalysisR::preprocess_texts(united_tbl, text_field = "united_texts")
#'
#'   dfm_object <- quanteda::dfm(tokens)
#'
#'   word_frequency_distribution_results <- TextAnalysisR::word_frequency_distribution(
#'                              dfm_object,
#'                              continuous_variable = "year",
#'                              selected_terms = c("calculator", "computer"),
#'                              height = 500,
#'                              width = 900)
#'   print(word_frequency_distribution_results$plot)
#'   print(word_frequency_distribution_results$table)
#' }
word_frequency_distribution <- function(dfm_object,
                                 continuous_variable,
                                 selected_terms,
                                 height = 500,
                                 width = 900) {

  if (!requireNamespace("htmltools", quietly = TRUE) ||
      !requireNamespace("MASS", quietly = TRUE) ||
      !requireNamespace("pscl", quietly = TRUE) ||
      !requireNamespace("broom", quietly = TRUE)) {
    stop(
      "The 'htmltools', 'pscl', 'MASS', and 'broom' packages are required for this functionality. ",
      "Please install them using install.packages(c('htmltools', 'MASS', 'pscl', 'broom'))."
    )
  }

  dfm_outcome_obj <- dfm_object
  dfm_td <- tidytext::tidy(dfm_object)

  dfm_outcome_obj@docvars$document <- dfm_outcome_obj@docvars$docname_

  dfm_td <- dfm_td %>%
    left_join(dfm_outcome_obj@docvars,
              by = c("document" = "document"))

  con_var_term_counts <- dfm_td %>%
    tibble::as_tibble() %>%
    group_by(!!rlang::sym(continuous_variable)) %>%
    mutate(word_frequency = n()) %>%
    ungroup()

  con_var_term_gg <- con_var_term_counts %>%
    mutate(term = factor(term, levels = selected_terms)) %>%
    mutate(across(where(is.numeric), ~ round(., 3))) %>%
    filter(term %in% selected_terms) %>%
    ggplot(aes(
      x = !!rlang::sym(continuous_variable),
      y = word_frequency,
      group = term
    )) +
    geom_point(color = "#337ab7", alpha = 0.6, size = 1) +
    geom_line(color = "#337ab7", alpha = 0.6, linewidth = 0.5) +
    facet_wrap(~ term, scales = "free") +
    ggplot2::scale_y_continuous(labels = scales::number_format(accuracy = 1)) +
    labs(y = "Word Frequency") +
    theme_minimal(base_size = 11) +
    theme(
      legend.position = "none",
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.line = element_line(color = "#3B3B3B", linewidth = 0.3),
      axis.ticks = element_line(color = "#3B3B3B", linewidth = 0.3),
      strip.text.x = element_text(size = 11, color = "#3B3B3B", face = "bold"),
      axis.text.x = element_text(size = 11, color = "#3B3B3B"),
      axis.text.y = element_text(size = 11, color = "#3B3B3B"),
      axis.title = element_text(size = 11, color = "#3B3B3B"),
      axis.title.x = element_text(margin = margin(t = 9)),
      axis.title.y = element_text(margin = margin(r = 11))
    )

  con_var_term_plotly <- plotly::ggplotly(
    con_var_term_gg,
    height = height,
    width = width
  ) %>%
    plotly::layout(
      margin = list(l = 40, r = 150, t = 60, b = 40)
    )

  significance_results <- con_var_term_counts %>%
    mutate(word = term) %>%
    filter(word %in% selected_terms) %>%
    group_by(word) %>%
    group_modify(~ {
      continuous_var <- if (is.null(continuous_variable) ||
                            length(continuous_variable) == 0) {
        stop("No continuous variable selected.")
      } else {
        continuous_variable[1]
      }

      df <- .x %>%
        dplyr::mutate(
          word_frequency = as.numeric(word_frequency),
          !!continuous_var := as.numeric(!!rlang::sym(continuous_var))
        ) %>%
        dplyr::filter(is.finite(word_frequency) &
                        is.finite(!!rlang::sym(continuous_var)))

      if (length(unique(df$word_frequency)) <= 1) {
        return(tibble::tibble(term = NA, estimate = NA, std.error = NA,
                              statistic = NA, p.value = NA,
                              `odds ratio` = NA, var.diag = NA,
                              `std.error (odds ratio)` = NA,
                              model_type = "Insufficient data"))
      }

      if (length(unique(df[[continuous_var]])) <= 1) {
        return(tibble::tibble(term = NA, estimate = NA, std.error = NA,
                              statistic = NA, p.value = NA,
                              `odds ratio` = NA, var.diag = NA,
                              `std.error (odds ratio)` = NA,
                              model_type = "Insufficient data"))
      }

      if (nrow(df) < 2) {
        return(tibble::tibble(term = NA, estimate = NA, std.error = NA,
                              statistic = NA, p.value = NA,
                              `odds ratio` = NA, var.diag = NA,
                              `std.error (odds ratio)` = NA,
                              model_type = "Insufficient data"))
      }

      formula_simple <- as.formula(paste0("word_frequency ~ ", continuous_var))

      mean_count <- mean(df$word_frequency, na.rm = TRUE)
      var_count <- var(df$word_frequency, na.rm = TRUE)
      dispersion_ratio <- ifelse(mean_count != 0, var_count / mean_count, NA)
      prop_zero <- mean(df$word_frequency == 0, na.rm = TRUE)

      model <- NULL

      if (prop_zero > 0.5) {
        model <- tryCatch(
          pscl::zeroinfl(formula_simple, data = df, dist = "negbin", link = "logit"),
          error = function(e) {
            return(NULL)
          }
        )
        if (!is.null(model)) {
          model_type <- "Zero-Inflated Negative Binomial"
        }
      }

      if (is.null(model)) {
        model <- tryCatch(
          MASS::glm.nb(formula_simple, data = df, control = glm.control(maxit = 200)),
          error = function(e) {
            return(NULL)
          }
        )
        if (is.null(model)) {
          model <- glm(formula_simple, family = poisson(link = "log"), data = df)
          model_type <- "Poisson"
        } else {
          model_type <- "Negative Binomial"
        }
      }

      tidy_result <- broom::tidy(model) %>%
        dplyr::mutate(
          `odds ratio` = exp(estimate),
          var.diag = diag(vcov(model)),
          `std.error (odds ratio)` = sqrt(`odds ratio`^2 * var.diag),
          model_type = model_type
        )

      return(tidy_result)
    }) %>%
    ungroup() %>%
    dplyr::select(word, model_type, term, estimate, std.error,
                  `odds ratio`, `std.error (odds ratio)`, statistic, p.value) %>%
    rename(
      logit = estimate,
      `z-statistic` = statistic
    )


  significance_results_tables <- significance_results %>%
    mutate(word = factor(word, levels = selected_terms)) %>%
    arrange(word) %>%
    group_by(word) %>%
    group_map(~ {
      htmltools::tagList(
        htmltools::tags$div(
          style = "margin-bottom: 20px;",
          htmltools::tags$p(
            .y$word,
            style = "font-weight: bold; text-align: center; font-size: 11pt;"
          )
        ),
        .x %>%
          mutate_if(is.numeric, ~ round(., 3)) %>%
          DT::datatable(
            rownames = FALSE,
            extensions = 'Buttons',
            options = list(
              scrollX = TRUE,
              width = "80%",
              dom = 'Bfrtip',
              buttons = c('copy', 'csv', 'excel', 'pdf', 'print')
            )
          ) %>%
          DT::formatStyle(
            columns = names(.x),
            `font-size` = "16px"
          )
      )
    })

  list(
    plot = con_var_term_plotly,
    table = htmltools::tagList(significance_results_tables) %>% htmltools::browsable()
  )
}


#' @title Evaluate Optimal Number of Topics
#'
#' @description
#' This function performs a search for the optimal number of topics (K) using \code{stm::searchK}
#' and visualizes diagnostics, including held-out likelihood, residuals, semantic coherence,
#' and lower bound metrics.
#'
#' @param dfm_object A \code{quanteda} document-feature matrix (dfm).
#' @param topic_range A numeric vector specifying the range of topics (K) to search over.
#' @param max.em.its Maximum number of EM iterations (default: 75).
#' @param categorical_var An optional character string for a categorical variable in the metadata.
#' @param continuous_var An optional character string for a continuous variable in the metadata.
#' @param height The height of the resulting Plotly plot in pixels (default: 600).
#' @param width The width of the resulting Plotly plot in pixels (default: 800).
#' @param verbose Logical; if TRUE, prints progress information.
#' @param ... Further arguments passed to \code{stm::searchK}.
#'
#' @return A \code{plotly} object showing the diagnostics for the number of topics (K).
#'
#' @importFrom quanteda convert
#' @importFrom stm searchK
#' @importFrom plotly plot_ly subplot layout
#' @importFrom dplyr mutate select
#' @importFrom stats as.formula
#' @importFrom utils str
#'
#' @export
#'
#' @examples
#' if (interactive()) {
#'   df <- TextAnalysisR::SpecialEduTech
#'
#'   united_tbl <- TextAnalysisR::unite_text_cols(df, listed_vars = c("title", "keyword", "abstract"))
#'
#'   tokens <- TextAnalysisR::preprocess_texts(united_tbl, text_field = "united_texts")
#'
#'   dfm_object <- quanteda::dfm(tokens)
#'
#'   optimal_topic_range <- TextAnalysisR::evaluate_optimal_topic_number(
#'                            dfm_object = dfm_object,
#'                            topic_range = 5:30,
#'                            max.em.its = 75,
#'                            categorical_var = "reference_type",
#'                            continuous_var = "year",
#'                            height = 600,
#'                            width = 800,
#'                            verbose = TRUE)
#'   print(optimal_topic_range)
#' }
evaluate_optimal_topic_number <- function(dfm_object,
                                          topic_range,
                                          max.em.its = 75,
                                          categorical_var = NULL,
                                          continuous_var = NULL,
                                          height = 600,
                                          width = 800,
                                          verbose = TRUE, ...) {

  out <- quanteda::convert(dfm_object, to = "stm")

  if (is.null(out$meta) || is.null(out$documents) || is.null(out$vocab)) {
    stop("Conversion to STM format failed. Please ensure your dfm_object is correctly formatted.")
  }

  categorical_var <- if (!is.null(categorical_var)) as.character(categorical_var) else NULL
  continuous_var <- if (!is.null(continuous_var)) as.character(continuous_var) else NULL

  if (!is.null(categorical_var)) {
    categorical_var <- unlist(strsplit(categorical_var, ",\\s*"))
  }
  if (!is.null(continuous_var)) {
    continuous_var <- unlist(strsplit(continuous_var, ",\\s*"))
  }

  missing_vars <- setdiff(c(categorical_var, continuous_var), names(out$meta))
  if (length(missing_vars) > 0) {
    stop("The following variables are missing in the metadata: ", paste(missing_vars, collapse = ", "))
  }

  terms <- c()
  if (!is.null(categorical_var) && length(categorical_var) > 0) {
    terms <- c(terms, categorical_var)
  }
  if (!is.null(continuous_var) && length(continuous_var) > 0) {
    terms <- c(terms, continuous_var)
  }

  prevalence_formula <- if (length(terms) > 0) {
    as.formula(paste("~", paste(terms, collapse = " + ")))
  } else {
    NULL
  }

  search_result <- tryCatch({
    stm::searchK(
      data = out$meta,
      documents = out$documents,
      vocab = out$vocab,
      max.em.its = max.em.its,
      init.type = "Spectral",
      K = topic_range,
      prevalence = prevalence_formula,
      verbose = verbose,
      ...
    )
  }, error = function(e) {
    stop("Error in stm::searchK: ", e$message)
  })

  # print(search_result$results)

  search_result$results$heldout <- as.numeric(search_result$results$heldout)
  search_result$results$residual <- as.numeric(search_result$results$residual)
  search_result$results$semcoh <- as.numeric(search_result$results$semcoh)
  search_result$results$lbound <- as.numeric(search_result$results$lbound)

  p1 <- plotly::plot_ly(
    data = search_result$results,
    x = ~K,
    y = ~heldout,
    type = 'scatter',
    mode = 'lines+markers',
    text = ~paste("K:", K, "<br>Held-out Likelihood:", round(heldout, 3)),
    hoverinfo = 'text',
    width = width,
    height = height
  )

  p2 <- plotly::plot_ly(
    data = search_result$results,
    x = ~K,
    y = ~residual,
    type = 'scatter',
    mode = 'lines+markers',
    text = ~paste("K:", K, "<br>Residuals:", round(residual, 3)),
    hoverinfo = 'text',
    width = width,
    height = height
  )

  p3 <- plotly::plot_ly(
    data = search_result$results,
    x = ~K,
    y = ~semcoh,
    type = 'scatter',
    mode = 'lines+markers',
    text = ~paste("K:", K, "<br>Semantic Coherence:", round(semcoh, 3)),
    hoverinfo = 'text',
    width = width,
    height = height
  )

  p4 <- plotly::plot_ly(
    data = search_result$results,
    x = ~K,
    y = ~lbound,
    type = 'scatter',
    mode = 'lines+markers',
    text = ~paste("K:", K, "<br>Lower Bound:", round(lbound, 3)),
    hoverinfo = 'text',
    width = width,
    height = height
  )

  plotly::subplot(p1, p2, p3, p4, nrows = 2, margin = 0.1) %>%
    plotly::layout(
      title = list(
        text = "Model Diagnostics by Number of Topics (K)",
        font = list(size = 16)
      ),
      showlegend = FALSE,
      margin = list(t = 100, b = 150, l = 50, r = 50),
      annotations = list(
        list(
          x = 0.25, y = 1.05, text = "Held-out Likelihood", showarrow = FALSE,
          xref = 'paper', yref = 'paper', xanchor = 'center', yanchor = 'bottom',
          font = list(size = 14)
        ),
        list(
          x = 0.75, y = 1.05, text = "Residuals", showarrow = FALSE,
          xref = 'paper', yref = 'paper', xanchor = 'center', yanchor = 'bottom',
          font = list(size = 14)
        ),
        list(
          x = 0.25, y = 0.5, text = "Semantic Coherence", showarrow = FALSE,
          xref = 'paper', yref = 'paper', xanchor = 'center', yanchor = 'bottom',
          font = list(size = 14)
        ),
        list(
          x = 0.75, y = 0.5, text = "Lower Bound", showarrow = FALSE,
          xref = 'paper', yref = 'paper', xanchor = 'center', yanchor = 'bottom',
          font = list(size = 14)
        ),
        list(
          x = 0.5, y = -0.2, text = "Number of Topics (K)", showarrow = FALSE,
          xref = 'paper', yref = 'paper', xanchor = 'center', yanchor = 'top',
          font = list(size = 14)
        )
      ),
      yaxis = list(
        title = list(
          text = "Metric Value",
          font = list(size = 14)
        )
      )
    )
}

#' Select Top Terms for Each Topic
#'
#' This function selects the top terms for each topic based on their word probability distribution (beta).
#'
#' @param stm_model An STM model object.
#' @param top_term_n The number of top terms to display for each topic (default: 10).
#' @param verbose Logical, if TRUE, prints progress messages.
#' @param ... Further arguments passed to tidytext::tidy.
#'
#' @return A data frame containing the top terms for each topic.
#'
#' @export
#'
#' @examples
#' if (interactive()) {
#'   df <- TextAnalysisR::SpecialEduTech
#'
#'   united_tbl <- TextAnalysisR::unite_text_cols(df, listed_vars = c("title", "keyword", "abstract"))
#'
#'   tokens <- TextAnalysisR::preprocess_texts(united_tbl, text_field = "united_texts")
#'
#'   dfm_object <- quanteda::dfm(tokens)
#'
#'   stm_15 <- TextAnalysisR::create_stm_model(
#'   dfm_object,
#'   topic_n = 15,
#'   max.em.its = 75,
#'   categorical_var = "reference_type",
#'   continuous_var = "year",
#'   verbose = TRUE
#'   )
#'
#'   out <- quanteda::convert(dfm_object, to = "stm")
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
#' top_topic_terms <- TextAnalysisR::select_top_topic_terms(
#'   stm_model = stm_15,
#'   top_term_n = 10,
#'   verbose = TRUE
#'   )
#' print(top_topic_terms)
#' }
select_top_topic_terms <- function(stm_model,
                             top_term_n = 10,
                             verbose = TRUE,
                             ...) {

  beta_td <- tidytext::tidy(stm_model, matrix = "beta", ...)

  top_topic_terms <- beta_td %>%
    dplyr::group_by(topic) %>%
    dplyr::slice_max(order_by = beta, n = top_term_n) %>%
    dplyr::ungroup()

  return(top_topic_terms)
}


#' Generate Topic Labels Using OpenAI's API
#'
#' This function generates descriptive labels for each topic based on their top terms using OpenAI's ChatCompletion API.
#'
#' @param top_topic_terms A data frame containing the top terms for each topic.
#' @param model A character string specifying which OpenAI model to use (default: "gpt-3.5-turbo").
#' @param system A character string containing the system prompt for the OpenAI API.
#' If NULL, the function uses the default system prompt.
#' @param user A character string containing the user prompt for the OpenAI API.
#' If NULL, the function uses the default user prompt.
#' @param temperature A numeric value controlling the randomness of the output (default: 0.5).
#' @param openai_api_key A character string containing the OpenAI API key.
#' If NULL, the function attempts to load the key from the OPENAI_API_KEY environment variable or the .env file in the working directory.
#' @param verbose Logical, if TRUE, prints progress messages.
#'
#' @return A data frame containing the top terms for each topic along with their generated labels.
#'
#' @export
#'
#' @examples
#' if (interactive()) {
#'   df <- TextAnalysisR::SpecialEduTech
#'
#'   united_tbl <- TextAnalysisR::unite_text_cols(df, listed_vars = c("title", "keyword", "abstract"))
#'
#'   tokens <- TextAnalysisR::preprocess_texts(united_tbl, text_field = "united_texts")
#'
#'   dfm_object <- quanteda::dfm(tokens)
#'
#'   out <- quanteda::convert(dfm_object, to = "stm")
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
#' top_topic_terms <- TextAnalysisR::select_top_topic_terms(
#'   stm_model = stm_15,
#'   top_term_n = 10,
#'   verbose = TRUE
#'   )
#'
#' top_labeled_topic_terms <- TextAnalysisR::generate_topic_labels(
#'   top_topic_terms,
#'   model = "gpt-3.5-turbo",
#'   temperature = 0.5,
#'   openai_api_key = "your_openai_api_key",
#'   verbose = TRUE)
#' print(top_labeled_topic_terms)
#'
#' # You can also load the Open AI API key from the .env file in the working directory as follows:
#' # OPENAI_API_KEY=your_openai_api_key
#'
#' top_labeled_topic_terms <- TextAnalysisR::generate_topic_labels(
#'   top_topic_terms,
#'   model = "gpt-3.5-turbo",
#'   temperature = 0.5,
#'   verbose = TRUE)
#' print(top_labeled_topic_terms)
#' }
generate_topic_labels <- function(top_topic_terms,
                                  model = "gpt-3.5-turbo",
                                  system = NULL,
                                  user = NULL,
                                  temperature = 0.5,
                                  openai_api_key = NULL,
                                  verbose = TRUE) {

  if (!requireNamespace("dotenv", quietly = TRUE) ||
      !requireNamespace("httr", quietly = TRUE) ||
      !requireNamespace("jsonlite", quietly = TRUE)) {
    stop(
      "The 'dotenv', 'httr', and 'jsonlite' packages are required for this functionality. ",
      "Please install them using install.packages(c('dotenv', 'httr', 'jsonlite'))."
    )
  }


  if (file.exists(".env")) {
    dotenv::load_dot_env()
  }

  if (is.null(openai_api_key)) {
    openai_api_key <- Sys.getenv("OPENAI_API_KEY")
  }

  if (nzchar(openai_api_key) == FALSE) {
    stop("No OpenAI API key provided or found in OPENAI_API_KEY environment variable.")
  }

  system <- "
You are a highly skilled data scientist specializing in generating concise and descriptive topic labels based on provided top terms for each topic.
Each topic consists of a list of terms ordered from most to least significant (by beta scores).

Your objective is to create precise labels that capture the essence of each topic by following these guidelines:

1. Use Person-First Language
   - Prioritize respectful and inclusive language.
   - Avoid terms that may be considered offensive or stigmatizing.
   - For example, use 'students with learning disabilities' instead of 'disabled students'.
   - Use 'students with visual impairments' instead of 'impaired students'
   - Use 'students with blindness' instead of 'blind students'.

1. Analyze Top Terms' Significance
   - Primary Focus: Emphasize high beta-score terms as they strongly define the topic.
   - Secondary Consideration: Include lower-scoring terms if they add essential context.

2. Synthesize the Topic Label
   - Clarity: Make sure the label is clear and easily understandable.
   - Conciseness: Aim for a short phrase of about 5-7 words.
   - Relevance: Reflect the collective meaning of the most influential terms.
   - Creativity: Use descriptive phrasing without sacrificing clarity.

3. Maintain Consistency
   - Capitalize the first word of all topic labels.
   - Keep formatting and terminology uniform across all labels.
   - Avoid ambiguity or generic wording that does not fit the provided top terms.

4. Adhere to Style Guidelines
   - Capitalization: Use title case for labels.
   - Avoid Jargon: Maintain accessibility; only use technical terms if absolutely necessary.
   - Uniqueness: Ensure each label is distinct and does not overlap significantly with others.

5. Handle Edge Cases
   - Conflicting Top Terms: If the terms suggest different directions, prioritize those with higher beta scores.
   - Low-Scoring Terms: Include them only if they add meaningful context.

6. Iterative Improvement
   - If the generated label is insufficiently representative, re-check term significance and revise accordingly.
   - Always adhere to these guidelines.

Example
----------
Top Terms (highest to lowest beta score):
virtual manipulatives (.035)
manipulatives (.022)
mathematical (.014)
app (.013)
solving (.013)
learning disability (.012)
algebra (.012)
area (.011)
tool (.010)
concrete manipulatives (.010)

Generated Topic Label:
Visual-based technology for mathematical problem solving

Focus on incorporating the most significant keywords while following the guidelines above to produce a concise, descriptive topic label.
"
  top_topic_terms <- top_topic_terms %>%
    dplyr::group_by(topic) %>%
    dplyr::arrange(desc(beta)) %>%
    dplyr::ungroup()

  unique_topics <- top_topic_terms %>%
    dplyr::distinct(topic) %>%
    dplyr::arrange(as.numeric(topic)) %>%
    dplyr::mutate(topic = row_number(), topic_label = NA)

  if (verbose) {
    if (!requireNamespace("progress", quietly = TRUE)) {
      utils::install.packages("progress")
    }
    pb <- progress::progress_bar$new(
      format = " Processing [:bar] :percent ETA: :eta",
      total = nrow(unique_topics),
      clear = FALSE, width = 60 )
  }

  for (i in seq_len(nrow(unique_topics))) {
    if (verbose) {
      pb$tick()
    }

    current_topic <- unique_topics$topic[i]

    selected_terms <- top_topic_terms %>%
      dplyr::filter(topic == current_topic) %>%
      dplyr::pull(term)

    user <- paste0(
      "You have a topic with keywords listed from most to least significant:",
      paste(selected_terms, collapse = ", "),
      "Please create a concise and descriptive label (5-7 words) that:",
      "1. Reflects the collective meaning of these keywords.",
      "2. Gives higher priority to the most significant terms.",
      "3. Adheres to the style guidelines provided in the system message."
    )

    body_list <- list(
      model = model,
      messages = list(
        list(role = "system", content = system),
        list(role = "user", content = user)
      ),
      temperature = temperature,
      max_tokens = 50
    )

    response <- httr::POST(
      url = "https://api.openai.com/v1/chat/completions",
      httr::add_headers(
        `Content-Type` = "application/json",
        `Authorization` = paste("Bearer", openai_api_key)
      ),
      body = jsonlite::toJSON(body_list, auto_unbox = TRUE),
      encode = "json"
    )

    if (httr::status_code(response) != 200) {
      warning(sprintf("OpenAI API request failed for topic '%s': %s",
                      current_topic, httr::content(response, "text", encoding = "UTF-8")))
      next
    }

    res_json <- jsonlite::fromJSON(httr::content(response, "text", encoding = "UTF-8"))

    if (verbose) {
      cat("Response JSON structure for topic", i, ":\n")
      print(str(res_json))
    }

    if (!is.null(res_json$choices) && nrow(res_json$choices) > 0) {
      if (!is.null(res_json$choices$message$content)) {
        topic_label <- res_json$choices$message$content[1]
        topic_label <- trimws(topic_label)
        topic_label <- gsub('^"(.*)"$', '\\1', topic_label)
        unique_topics$topic_label[i] <- topic_label
      } else {
        warning(sprintf("Unexpected response structure for topic '%s': %s", current_topic, jsonlite::toJSON(res_json, auto_unbox = TRUE)))
        next
      }
    } else {
      warning(sprintf("Unexpected response structure for topic '%s': %s", current_topic, jsonlite::toJSON(res_json, auto_unbox = TRUE)))
      next
    }

    Sys.sleep(1)
  }

  top_labeled_topic_terms <- top_topic_terms %>%
    dplyr::left_join(unique_topics, by = "topic") %>%
    dplyr::select(topic_label, topic, term, beta) %>%
    dplyr::arrange(topic, desc(beta))

  return(top_labeled_topic_terms)
}


#' @title Plot Highest Word Probabilities for Each Topic
#'
#' @description
#' This function provides a visualization of the top terms for each topic,
#' ordered by their word probability distribution for each topic (beta).
#'
#' @param top_topic_terms A data frame containing the top terms for each topic.
#' @param topic_label A character vector of topic labels for each topic. If NULL, the function uses the topic number.
#' @param ncol The number of columns in the facet plot (default: 3).
#' @param height The height of the resulting Plotly plot, in pixels (default: 1200).
#' @param width The width of the resulting Plotly plot, in pixels (default: 800).
#' @param ... Additional arguments passed to \code{plotly::layout}.
#'
#' @return A \code{Plotly} object showing a facet-wrapped chart of top terms for each topic,
#' ordered by their per-topic probability (beta). Each facet represents a topic.
#'
#' @details The function uses the \code{ggplot2} package to create a facet-wrapped chart of top terms for each topic,
#'
#'
#' @importFrom stats reorder
#' @importFrom numform ff_num
#' @importFrom plotly ggplotly layout
#' @importFrom tidytext reorder_within scale_x_reordered
#'
#' @export
#'
#' @examples
#' if (interactive()) {
#'  df <- TextAnalysisR::SpecialEduTech
#'
#'  united_tbl <- TextAnalysisR::unite_text_cols(df, listed_vars = c("title", "keyword", "abstract"))
#'
#'  tokens <- TextAnalysisR::preprocess_texts(united_tbl, text_field = "united_texts")
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
#' top_topic_terms <- TextAnalysisR::select_top_topic_terms(
#'   stm_model = stm_15,
#'   top_term_n = 10,
#'   verbose = TRUE
#'   )
#'
#' top_labeled_topic_terms <- TextAnalysisR::generate_topic_labels(
#'   top_topic_terms = top_topic_terms,
#'   model = "gpt-3.5-turbo",
#'   temperature = 0.5,
#'   openai_api_key = "your_openai_api_key",
#'   verbose = TRUE)
#' top_labeled_topic_terms
#'
#'
#' TextAnalysisR::plot_word_probabilities(
#'   top_labeled_topic_terms,
#'   topic_label = "topic_label",
#'   ncol = 3,
#'   height = 1200,
#'   width = 800
#'   )
#'
#' TextAnalysisR::plot_word_probabilities(
#'   top_topic_terms,
#'   ncol = 3,
#'   height = 1200,
#'   width = 800
#'   )
#'
#'
#'  manual_labels <- c("1" = "Mathematical technology for students with LD",
#'                     "2" = "STEM technology",
#'                     "3" = "CAI for math problem solving")
#'
#' word_probability_plot <- TextAnalysisR::word_probability_plot(
#'                          top_topic_terms,
#'                          topic_label = manual_labels,
#'                          ncol = 3,
#'                          height = 1200,
#'                          width = 800)
#' print(word_probability_plot)
#'
#' }
word_probability_plot <- function(top_topic_terms,
                                    topic_label = NULL,
                                    ncol = 3,
                                    height = 1200,
                                    width = 800,
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
                 text = paste("Topic:", labeled_topic, "<br>Beta:", sprintf("%.3f", beta)))
  ) +
    ggplot2::geom_col(show.legend = FALSE, alpha = 0.9) +
    ggplot2::facet_wrap(~ labeled_topic, scales = "free", ncol = ncol, strip.position = "top") +
    tidytext::scale_x_reordered() +
    ggplot2::scale_y_continuous(labels = scales::number_format(accuracy = 0.001)) +
    ggplot2::coord_flip() +
    ggplot2::xlab("") +
    ggplot2::ylab("Word probability") +
    ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(
      legend.position = "none",
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.line = element_line(color = "#3B3B3B", linewidth = 0.3),
      axis.ticks = element_line(color = "#3B3B3B", linewidth = 0.3),
      strip.text.x = element_text(
        size = 11,
        face = "bold",
        lineheight = ifelse(width > 1000, 1.1, 1.5),
        margin = margin(l = 20, r = 20)
      ),
      panel.spacing.x = unit(ifelse(width > 1000, 2.2, 1.6), "lines"),
      panel.spacing.y = unit(ifelse(width > 1000, 2.2, 1.6), "lines"),
      axis.text.x = element_text(size = 11, color = "#3B3B3B", hjust = 1, margin = margin(r = 20)),
      axis.text.y = element_text(size = 11, color = "#3B3B3B", margin = margin(t = 20)),
      axis.title = element_text(size = 11, color = "#3B3B3B"),
      axis.title.x = element_text(margin = margin(t = 25)),
      axis.title.y = element_text(margin = margin(r = 25)),
      plot.margin = margin(t = 40, b = 40, l = 100, r = 40)
    )

  plotly::ggplotly(ggplot_obj, height = height, width = width, tooltip = "text", ...) %>%
    plotly::layout(margin = list(t = 40, b = 40))
}



#' @title Plot Per-Document Per-Topic Probabilities
#'
#' @description
#' This function generates a bar plot showing the prevalence of each topic across all documents.
#'
#' @param stm_model A fitted STM model object.
#'   where \code{stm_model} is a fitted Structural Topic Model created using \code{stm::stm()}.
#' @param top_n The number of topics to display, ordered by their mean prevalence.
#' @param height The height of the resulting Plotly plot, in pixels (default: 800).
#' @param width The width of the resulting Plotly plot, in pixels (default: 1000).
#' @param verbose Logical, if TRUE, prints progress messages.
#' @param ... Further arguments passed to \code{tidytext::tidy}.
#'
#' @return A \code{ggplot} object showing a bar plot of topic prevalence. Topics are ordered by their
#' mean gamma value (average prevalence across documents).
#'
#' @export
#'
#' @examples
#' if (interactive()) {
#'
#' df <- TextAnalysisR::SpecialEduTech
#'
#'  united_tbl <- TextAnalysisR::unite_text_cols(df, listed_vars = c("title", "keyword", "abstract"))
#'
#'  tokens <- TextAnalysisR::preprocess_texts(united_tbl, text_field = "united_texts")
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
#' topic_probability_plot <- TextAnalysisR::topic_probability_plot(
#'  stm_model= stm_15,
#'  top_n = 10,
#'  height = 800,
#'  width = 1000,
#'  verbose = TRUE)
#'
#' print(topic_probability_plot)
#' }
topic_probability_plot <- function(stm_model,
                                   top_n = 10,
                                   height = 800,
                                   width = 1000,
                                   verbose = TRUE,
                                   ...) {

    gamma_td <- tidytext::tidy(stm_model, matrix="gamma", ...)

    gamma_terms <- gamma_td %>%
      group_by(topic) %>%
      summarise(gamma = mean(gamma)) %>%
      arrange(desc(gamma)) %>%
      mutate(topic = reorder(topic, gamma)) %>%
      top_n(top_n, gamma)

    ggplot_obj <-ggplot(gamma_terms, aes(topic, gamma, fill = topic)) +
      geom_col(alpha = 0.8) +
      coord_flip() +
      scale_y_continuous(labels = ff_num(zero = 0, digits = 2)) +
      xlab("") +
      ylab("Topic proportion") +
      theme_minimal(base_size = 10) +
      theme(
        legend.position = "none",
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(color = "#3B3B3B", linewidth = 0.3),
        axis.ticks = element_line(color = "#3B3B3B", linewidth = 0.3),
        strip.text.x = element_text(size = 10, color = "#3B3B3B"),
        axis.text.x = element_text(size = 10, color = "#3B3B3B"),
        axis.text.y = element_text(size = 10, color = "#3B3B3B"),
        axis.title = element_text(size = 10, color = "#3B3B3B"),
        axis.title.x = element_text(margin = margin(t = 9)),
        axis.title.y = element_text(margin = margin(r = 9))
      )

    plotly::ggplotly(ggplot_obj, height = height, width = width) %>%
      plotly::layout(margin = list(t = 40, b = 40))
}


#' @title Create a Table for Per-Document Per-Topic Probabilities
#'
#' @description
#' This function generates a table of mean topic prevalence across all documents.
#'
#' @param stm_model A fitted STM model object.
#' @param top_n The number of topics to display, ordered by their mean prevalence.
#' @param verbose Logical, if TRUE, prints progress messages.
#' @param ... Further arguments passed to \code{tidytext::tidy}.
#'
#' @return A \code{tibble} containing columns \code{topic} and \code{gamma}, where \code{topic}
#' is a factor representing each topic (relabeled with a "Topic X" format), and \code{gamma} is the
#' mean topic prevalence across all documents. Numeric values are rounded to three decimal places.
#'
#' @export
#'
#' @examples
#' if (interactive()) {
#'
#' df <- TextAnalysisR::SpecialEduTech
#'
#'  united_tbl <- TextAnalysisR::unite_text_cols(df, listed_vars = c("title", "keyword", "abstract"))
#'
#'  tokens <- TextAnalysisR::preprocess_texts(united_tbl, text_field = "united_texts")
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
#' topic_probability_table <- TextAnalysisR::topic_probability_table(
#'    stm_model= stm_15,
#'    top_n = 10,
#'    verbose = TRUE)
#'
#' print(topic_probability_table)
#' }
topic_probability_table <- function(stm_model,
                                    top_n = 10,
                                    verbose = TRUE,
                                    ...) {

  gamma_td <- tidytext::tidy(stm_model, matrix="gamma", ...)

  gamma_terms <- gamma_td %>%
    group_by(topic) %>%
    summarise(gamma = mean(gamma)) %>%
    arrange(desc(gamma)) %>%
    mutate(topic = reorder(topic, gamma)) %>%
    top_n(top_n, gamma) %>%
    mutate(tt = as.numeric(topic)) %>%
    mutate(ord = topic) %>%
    mutate(topic = paste('Topic', topic)) %>%
    arrange(ord)

  levelt = paste("Topic", gamma_terms$ord) %>% unique()
  gamma_terms$topic = factor(gamma_terms$topic, levels = levelt)

  gamma_terms %>%
    select(topic, gamma) %>%
    mutate_if(is.numeric, ~ round(., 3)) %>%
    DT::datatable(
      rownames = FALSE,
      extensions = 'Buttons',
      options = list(
        scrollX = TRUE,
        scrollY = "400px",
        width = "80%",
        dom = 'Bfrtip',
        buttons = c('copy', 'csv', 'excel', 'pdf', 'print')
      )
    ) %>%
    DT::formatStyle(
      columns = c("topic", "gamma"),
      fontSize = '16px'
    )
}


