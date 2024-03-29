# Preprocess Text Data ----

#' @title Preprocess text data
#'
#' @name preprocess_texts
#'
#' @description
#' Preprocess text data by conducting the following functions:
#' construct a corpus; segment texts in a corpus into tokens; preprocess tokens;
#' convert the features of tokens to lowercase;
#' remove stopwords; specify the minimum length in characters for tokens (at least 2).
#'
#' @param data A data frame that contains text as data.
#' @param text_field A name of column that contains text data in a data frame.
#' @param ... Further arguments passed to \code{corpus}.
#'
#' @export
#' @return A tokens object output from \code{quanteda::tokens}.
#' The result is a list of tokenized and preprocessed text data.
#'
#' @examples
#' suppressWarnings({
#' SpecialEduTech %>% preprocess_texts(text_field = "abstract")
#' })
#'
#' @import quanteda
#' @importFrom magrittr %>%
#' @importFrom rlang := enquos

preprocess_texts <-
    function(data, text_field = "united_texts", ...) {

        # Construct a corpus
        corp <- quanteda::corpus(data, text_field = text_field, ...)

        # Segment texts in a corpus into tokens (words or sentences) by word boundaries
        toks <- quanteda::tokens(corp)

        # Preprocess tokens
        toks_clean <- quanteda::tokens(
            toks,
            what = "word",
            remove_punct = TRUE,
            remove_symbols = TRUE,
            remove_numbers = TRUE,
            remove_url = TRUE,
            remove_separators = TRUE,
            split_hyphens = TRUE,
            split_tags = TRUE,
            include_docvars = TRUE,
            padding = FALSE,
            verbose = FALSE)

        # Convert the features of tokens to lowercase.
        toks_lower <- quanteda::tokens_tolower(toks_clean,
                                               keep_acronyms = FALSE)

        # Remove English stopwords.
        toks_lower_no_stop <- toks_lower %>%
            quanteda::tokens_remove(quanteda::stopwords("en"),
                                    valuetype = "glob",
                                    window = 0,
                                    verbose = FALSE,
                                    padding = TRUE)

        # Specify the minimum length in characters for tokens (at least 2).
        toks_lower_no_stop_adj <- toks_lower_no_stop %>%
            quanteda::tokens_select(min_nchar=2L,
                                    verbose = FALSE)

        return(toks_lower_no_stop_adj)
    }


#' @title Plot word frequency results.
#'
#' @name plot_word_frequency
#'
#' @description
#' Plot the frequently observed top n terms.
#'
#' @param data A document-feature matrix (dfm) object through the quanteda package.
#' @param n The number of top n features (terms or words).
#' @param ... Further arguments passed to \code{quanteda.textstats::textstat_frequency}.
#'
#' @export
#' @return A ggplot object output from \code{quanteda.textstats::textstat_frequency} and \code{ggplot2::ggplot}.
#' The result is a ggplot object representing the word frequency plot.
#'
#' @examples
#' suppressWarnings({
#' if(requireNamespace("quanteda")){
#' dfm <- SpecialEduTech %>%
#'        preprocess_texts(text_field = "abstract") %>%
#'        quanteda::dfm()
#' dfm %>% plot_word_frequency(n = 20)
#' }
#' })
#'
#' @importFrom magrittr %>%
#' @importFrom rlang := enquos
#' @importFrom ggplot2 ggplot geom_point coord_flip labs theme_bw
#'
plot_word_frequency <-
    function(data, n = 20, ...) {
        word_frequency_plot <- data %>%
            quanteda.textstats::textstat_frequency(n = n, ...) %>%
            ggplot(aes(x = stats::reorder(feature, frequency), y = frequency)) +
            geom_point(colour = "#5f7994", size = 1) +
            coord_flip() +
            labs(x = NULL, y = "Word frequency") +
            theme_bw(base_size = 12)
        return(word_frequency_plot)
    }


#' @title Examine highest per-term per-topic probabilities
#'
#' @name examine_top_terms
#'
#' @description
#' Examine highest per-term per-topic probabilities.
#'
#' @param data A tidy data frame that includes per-term per-topic probabilities (beta).
#' @param top_n A number of highest per-term per-topic probabilities in each document (number of top_n can be changed).
#' @param ... Further arguments passed to \code{dplyr::group_by}.
#'
#' @export
#' @return A tibble (data frame) object with a list of word probabilities from \code{tidytext::tidy}.
#' The result is a data frame containing word probabilities for each topic.
#'
#' @examples
#' suppressWarnings({
#' if(requireNamespace("quanteda", "tidytext")){
#' dfm <- SpecialEduTech %>%
#'        preprocess_texts(text_field = "abstract") %>%
#'        quanteda::dfm()
#' data <- tidytext::tidy(stm_15, document_names = rownames(dfm), log = FALSE)
#' data %>% examine_top_terms(top_n = 5) %>%
#' dplyr::mutate_if(is.numeric, ~ round(., 3)) %>%
#' DT::datatable(rownames = FALSE)
#' }
#' })
#'
#' @import dplyr
#' @importFrom magrittr %>%
#' @importFrom rlang := enquos
#'
examine_top_terms <-

  function(data, top_n, ...) {
    topic_term <- data %>%
      group_by(topic, ...) %>%
      top_n(top_n, beta) %>%
      ungroup()

    return(topic_term)
  }


# Display text mining results from the structural topic model ----

#' @title Plot topic per-term per-topic probabilities
#'
#' @name plot_topic_term
#'
#' @description
#' Plot per-term per-topic probabilities with highest word probabilities.
#'
#' @param data A tidy data frame that includes per-term per-topic probabilities (beta).
#' @param ncol A number of columns in the facet plot.
#' @param topic_names (Labeled) topic names
#' @param ... Further arguments passed to \code{dplyr::group_by}.
#'
#' @export
#' @return A ggplot object output from \code{stm::stm}, \code{tidytext::tidy}, and \code{ggplot2::ggplot}.
#' The result is a ggplot object representing the topic-term plot.
#'
#' @examples
#' suppressWarnings({
#' if(requireNamespace("quanteda", "tidytext")){
#' dfm <- SpecialEduTech %>%
#'        preprocess_texts(text_field = "abstract") %>%
#'        quanteda::dfm()
#' data <- tidytext::tidy(stm_15, document_names = rownames(dfm), log = FALSE)
#' data %>% examine_top_terms(top_n = 2) %>%
#' plot_topic_term(ncol = 3)
#' }
#' })
#'
#' @import dplyr
#' @import ggplot2
#' @importFrom magrittr %>%
#' @importFrom rlang := enquos
#' @importFrom tidytext scale_x_reordered reorder_within
#'
plot_topic_term <-
  function(data, ncol = ncol, topic_names = NULL, ...) {

    topic_term <- data %>%
      mutate(
        ord = factor(topic, levels = c(min(topic): max(topic))),
        tt = as.numeric(topic),
        topic = paste("Topic", topic),
        term = reorder_within(term, beta, topic)) %>%
      arrange(ord)

    levelt = paste("Topic", topic_term$ord) %>% unique()

    topic_term$topic = factor(topic_term$topic,
                              levels = levelt)
    if(!is.null(topic_names)){
      topic_term$topic = topic_names[topic_term$tt]
      topic_term <- topic_term %>%
        mutate(topic = as.character(topic)) %>%
        mutate(topic = ifelse(!is.na(topic), topic, paste('Topic',tt)))
      topic_term$topic =
        factor(topic_term$topic, levels = topic_term$topic %>% unique())
    }

    topic_term$tt = NULL

    topic_term_plot <- ggplot(topic_term, aes(term, beta, fill = topic)) +
      geom_col(show.legend = FALSE, alpha = 0.8) +
      facet_wrap(~ topic, scales = "free", ncol = ncol) +
      scale_x_reordered() +
      scale_y_continuous(labels = numform::ff_num(zero = 0, digits = 3)) +
      coord_flip() +
      xlab("") +
      ylab("Word probability") +
      theme_minimal(base_size = 12) +
      theme(
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(color = "#3B3B3B", size = 0.3),
        axis.ticks = element_line(color = "#3B3B3B", size = 0.3),
        strip.text.x = element_text(size = 12, color = "#3B3B3B"),
        axis.text.x = element_text(size = 12, color = "#3B3B3B"),
        axis.text.y = element_text(size = 12, color = "#3B3B3B"),
        axis.title = element_text(size = 12, color = "#3B3B3B"),
        axis.title.x = element_text(margin = margin(t = 7)),
        axis.title.y = element_text(margin = margin(r = 7)))

    return(topic_term_plot)
  }


#' @title Plot per-document per-topic probabilities
#'
#' @name topic_probability_plot
#'
#' @description
#' Plot per-document per-topic probabilities.
#'
#' @param data A tidy data frame that includes per-document per-topic probabilities (gamma).
#' @param top_n A number of highest per-document per-topic probabilities (number of top_n can be changed).
#' @param ... Further arguments passed.
#'
#' @export
#' @return A ggplot object output from \code{stm::stm}, \code{tidytext::tidy}, and \code{ggplot2::ggplot}.
#'
#' @examples
#' suppressWarnings({
#' if(requireNamespace("quanteda", "tidytext")){
#' dfm <- SpecialEduTech %>%
#'        preprocess_texts(text_field = "abstract") %>%
#'        quanteda::dfm()
#' data <- tidytext::tidy(stm_15, matrix = "gamma", document_names = rownames(dfm), log = FALSE)
#' data %>% topic_probability_plot(top_n = 15) %>% plotly::ggplotly()
#' }
#' })
#'
#' @import dplyr
#' @import ggplot2
#' @importFrom magrittr %>%
#' @importFrom stats reorder
#' @importFrom plotly ggplotly
#'
topic_probability_plot <-
  function(data, top_n, ...) {

    gamma_terms <- data %>%
      group_by(topic) %>%
      summarise(gamma = mean(gamma)) %>%
      arrange(desc(gamma)) %>%
      mutate(topic = reorder(topic, gamma))

    topic_by_prevalence_plot <- gamma_terms %>%
      ggplot(aes(topic, gamma, fill = topic)) +
      geom_col(alpha = 0.8) +
      coord_flip() +
      scale_y_continuous(labels = numform::ff_num(zero = 0, digits = 2)) +
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
        return(topic_by_prevalence_plot)
    }


#' @title Visualize a table for per-document per-topic probabilities
#'
#' @name topic_probability_table
#'
#' @description
#' Create a table of per-document per-topic probabilities.
#'
#' @param data A tidy data frame that includes per-document per-topic probabilities (gamma).
#' @param top_n A number of highest per-document per-topic probabilities (number of top_n can be changed).
#' @param ... Further arguments passed.
#'
#' @export
#' @return A tibble (data frame) object with a list of topic probabilities from \code{tidytext::tidy}.
#' The result is a ggplot object representing the topic-term plot.
#'
#' @examples
#' suppressWarnings({
#' if(requireNamespace("quanteda", "tidytext")){
#' dfm <- SpecialEduTech %>%
#'        preprocess_texts(text_field = "abstract") %>%
#'        quanteda::dfm()
#' data <- tidytext::tidy(stm_15, matrix = "gamma", document_names = rownames(dfm), log = FALSE)
#' data %>% topic_probability_table(top_n = 15) %>% DT::datatable(rownames = FALSE)
#' }
#' })
#'
#' @import dplyr
#' @import ggplot2
#' @importFrom magrittr %>%
#' @importFrom stats reorder
#' @importFrom DT datatable
#'
topic_probability_table <-
    function(data, top_n, ...) {

      gamma_terms <- data %>%
        group_by(topic) %>%
        summarise(gamma = mean(gamma)) %>%
        arrange(gamma) %>%
        mutate(topic = reorder(topic, gamma))

      topic_by_prevalence_table <- gamma_terms %>%
        top_n(top_n, gamma) %>%
        mutate(tt = as.numeric(topic)) %>%
        mutate(ord = topic) %>%
        mutate(topic = paste('Topic', topic)) %>%  arrange(ord)

      levelt = paste("Topic", topic_by_prevalence_table$ord) %>% unique()

      topic_by_prevalence_table$topic = factor(topic_by_prevalence_table$topic,
                                               levels = levelt)
      topic_by_prevalence_table_output <- topic_by_prevalence_table %>%
        select(topic, gamma) %>%
        mutate_if(is.numeric, ~ round(., 3))

        return(topic_by_prevalence_table_output)
    }
