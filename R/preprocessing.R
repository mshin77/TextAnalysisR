#' @importFrom utils modifyList
#' @importFrom stats cor
NULL

# Text Preprocessing Functions
# Functions for data import, text preprocessing, and feature extraction

#' Get Available Document-Feature Matrix with Fallback
#'
#' @description Returns the first non-NULL DFM from a priority fallback chain.
#' Useful when multiple DFM processing stages exist and you need the most processed available version.
#'
#' @param dfm_lemma Optional lemmatized DFM (highest priority)
#' @param dfm_outcome Optional preprocessed DFM (medium priority)
#' @param dfm_final Optional final processed DFM (medium-low priority)
#' @param dfm_init Optional initial DFM (lowest priority)
#'
#' @return The first non-NULL DFM from the priority chain, or NULL if all are NULL
#'
#' @details
#' Priority order (highest to lowest):
#' 1. dfm_lemma - Lemmatized tokens (most processed)
#' 2. dfm_outcome - Preprocessed tokens
#' 3. dfm_final - Final processed version
#' 4. dfm_init - Initial unprocessed tokens
#'
#' @family preprocessing
#' @export
#'
#' @examples
#' \dontrun{
#' dfm1 <- quanteda::dfm(quanteda::tokens("assistive technology supports learning"))
#' result <- get_available_dfm(dfm_init = dfm1)
#' }
get_available_dfm <- function(dfm_lemma = NULL, dfm_outcome = NULL, dfm_final = NULL, dfm_init = NULL) {
  if (!is.null(dfm_lemma)) {
    return(dfm_lemma)
  }

  if (!is.null(dfm_outcome)) {
    return(dfm_outcome)
  }

  if (!is.null(dfm_final)) {
    return(dfm_final)
  }

  if (!is.null(dfm_init)) {
    return(dfm_init)
  }

  return(NULL)
}


#' Get Available Tokens with Fallback
#'
#' @description Returns the first non-NULL tokens object from a priority fallback chain.
#' Useful when multiple token processing stages exist and you need the most processed available version.
#'
#' @param final_tokens Optional fully processed tokens (highest priority)
#' @param processed_tokens Optional partially processed tokens
#' @param preprocessed_tokens Optional early-stage preprocessed tokens
#' @param united_tbl Optional data frame with united_texts column (lowest priority, will be tokenized)
#'
#' @return The first non-NULL tokens from the priority chain, or NULL if all are NULL
#'
#' @details
#' Priority order (highest to lowest):
#' 1. final_tokens - Fully processed tokens
#' 2. processed_tokens - Partially processed tokens
#' 3. preprocessed_tokens - Early stage preprocessed tokens
#' 4. united_tbl - Raw text (will be tokenized if used)
#'
#' @family preprocessing
#' @export
#'
#' @examples
#' \dontrun{
#' tokens <- get_available_tokens(
#'   final_tokens = my_final_tokens,
#'   processed_tokens = my_processed_tokens
#' )
#' }
get_available_tokens <- function(final_tokens = NULL,
                                  processed_tokens = NULL,
                                  preprocessed_tokens = NULL,
                                  united_tbl = NULL) {
  if (!is.null(final_tokens)) {
    return(final_tokens)
  }

  if (!is.null(processed_tokens)) {
    return(processed_tokens)
  }

  if (!is.null(preprocessed_tokens)) {
    return(preprocessed_tokens)
  }

  if (!is.null(united_tbl) && "united_texts" %in% names(united_tbl)) {
    toks <- quanteda::tokens(united_tbl$united_texts, what = "word")
    other_cols <- united_tbl[, !names(united_tbl) %in% "united_texts", drop = FALSE]
    if (ncol(other_cols) > 0) {
      quanteda::docvars(toks) <- other_cols
    }
    return(toks)
  }

  return(NULL)
}

#' @title Process Files
#'
#' @description
#' This function processes different types of files and text input based on the dataset choice.
#'
#' @param dataset_choice A character string indicating the dataset choice.
#' @param file_info A data frame containing file information with a column
#'   named 'filepath' (default: NULL).
#' @param text_input A character string containing text input (default: NULL).
#'
#' @return A data frame containing the processed data.
#'
#' @importFrom utils read.csv
#'
#' @family preprocessing
#' @export
#'
#' @examples
#' if (interactive()) {
#'   mydata <- TextAnalysisR::SpecialEduTech
#'   mydata <- TextAnalysisR::import_files(dataset_choice = "Upload an Example Dataset")
#'   head(mydata)
#'
#'   file_info <- data.frame(filepath = "inst/extdata/SpecialEduTech.xlsx")
#'   mydata <- TextAnalysisR::import_files(dataset_choice = "Upload Your File",
#'                                           file_info = file_info)
#'   head(mydata)
#'
#'
#'   text_input <- paste("Virtual manipulatives for algebra instruction",
#'                       "manipulatives mathematics learning disability",
#'                       "This study examined virtual manipulatives effects on",
#'                       "students with learning disabilities")
#'   mydata <- TextAnalysisR::import_files(dataset_choice = "Copy and Paste Text",
#'                                           text_input = text_input)
#'   head(mydata)
#' }
import_files <- function(dataset_choice, file_info = NULL, text_input = NULL) {

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
          # Note: This extracts text only. For image extraction from DOCX,
          # use process_docx_multimodal() function (requires Python + Vision API)
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


.extract_multimodal_pdf <- function(file_path, vision_provider,
                                    vision_model, api_key,
                                    describe_images) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("reticulate package required")
  }

  if (is.null(vision_model)) {
    vision_model <- if (vision_provider == "ollama") "llava" else "gpt-4o"
  }

  if (vision_provider == "ollama") {
    ollama_check <- check_vision_models("ollama")
    if (!isTRUE(ollama_check$available)) {
      stop("Ollama not available: ", ollama_check$message)
    }
  }

  pdf_result <- extract_pdf_multimodal(
    file_path = file_path,
    vision_provider = vision_provider,
    vision_model = vision_model,
    api_key = api_key,
    describe_images = describe_images
  )

  if (!pdf_result$success) {
    stop(pdf_result$message)
  }

  combined_lines <- strsplit(pdf_result$combined_text, "\n")[[1]]
  combined_lines <- combined_lines[nchar(trimws(combined_lines)) > 0]

  list(
    success = TRUE,
    data = data.frame(text = combined_lines, stringsAsFactors = FALSE),
    type = "multimodal",
    method = "multimodal",
    message = paste("Extracted text and", pdf_result$num_images,
                    "image descriptions"),
    num_images = pdf_result$num_images,
    vision_provider = pdf_result$vision_provider
  )
}

.extract_python_pdf <- function(file_path) {
  env_check <- check_python_env()
  if (!env_check$available) {
    stop("Python environment not available")
  }

  pdf_data <- process_pdf_file_py(file_path, content_type = "auto")

  if (!pdf_data$success || is.null(pdf_data$data)) {
    stop(pdf_data$message)
  }

  list(
    success = TRUE,
    data = pdf_data$data,
    type = pdf_data$type,
    method = "python",
    message = paste("Python extraction:", pdf_data$message)
  )
}

.extract_r_pdf <- function(file_path) {
  if (!requireNamespace("pdftools", quietly = TRUE)) {
    stop("pdftools package required")
  }

  pages <- pdftools::pdf_text(file_path)
  lines <- unlist(lapply(pages, function(page) {
    page_lines <- strsplit(page, "\n")[[1]]
    trimws(page_lines)
  }))
  lines <- lines[lines != ""]

  if (length(lines) == 0) {
    stop("No extractable text found")
  }

  list(
    success = TRUE,
    data = data.frame(text = lines, stringsAsFactors = FALSE),
    type = "text",
    method = "r",
    message = paste("R extraction: Extracted", length(lines), "lines")
  )
}

#' @title Process PDF File (Unified Entry Point)
#'
#' @description
#' Unified PDF processing with automatic fallback:
#' 1. Multimodal (Python + Vision) 2. Python pdfplumber 3. R pdftools
#'
#' @param file_path Character string path to PDF file
#' @param use_multimodal Logical, enable multimodal extraction
#' @param vision_provider Character, "ollama" or "openai"
#' @param vision_model Character, model name
#' @param api_key Character, OpenAI API key (if using OpenAI)
#' @param describe_images Logical, generate image descriptions
#'
#' @return List: success, data, type, method, message
#'
#' @family preprocessing
#' @export
process_pdf_unified <- function(file_path,
                                use_multimodal = FALSE,
                                vision_provider = "ollama",
                                vision_model = NULL,
                                api_key = NULL,
                                describe_images = TRUE) {

  if (!file.exists(file_path)) {
    return(list(success = FALSE, data = NULL, type = "error",
                method = "none", message = "File not found"))
  }

  if (use_multimodal) {
    prereq_check <- check_multimodal_prerequisites(
      vision_provider = vision_provider,
      vision_model = vision_model,
      api_key = api_key
    )

    if (!prereq_check$ready) {
      return(list(
        success = FALSE,
        data = NULL,
        type = "prerequisite_error",
        method = "multimodal",
        message = prereq_check$instructions,
        missing = prereq_check$missing
      ))
    }

    result <- tryCatch(
      .extract_multimodal_pdf(file_path, vision_provider, vision_model,
                              api_key, describe_images),
      error = function(e) list(
        success = FALSE,
        type = "extraction_error",
        message = paste("Multimodal extraction failed:", e$message)
      )
    )

    if (result$success) return(result)

    return(list(
      success = FALSE,
      data = NULL,
      type = "extraction_error",
      method = "multimodal",
      message = paste0(
        "Multimodal extraction encountered an error.\n\n",
        "Error: ", result$message, "\n\n",
        "Note: Pull the vision model using terminal/command prompt (not R code): ollama pull llava"
      )
    ))
  }

  result <- tryCatch(
    .extract_python_pdf(file_path),
    error = function(e) list(success = FALSE,
                             message = paste("Python failed:", e$message))
  )
  if (result$success) return(result)

  tryCatch(
    .extract_r_pdf(file_path),
    error = function(e) {
      list(success = FALSE, data = NULL, type = "error", method = "none",
           message = paste("All methods failed:", e$message))
    }
  )
}



#' @title Unite Text Columns
#'
#' @description
#' This function unites specified text columns in a data frame into a single
#' column named "united_texts" while retaining the original columns.
#'
#' @param df A data frame that contains text data.
#' @param listed_vars A character vector of column names to be united into
#'   "united_texts".
#'
#' @return A data frame with a new column "united_texts" created by uniting
#'   the specified variables.
#'
#' @family preprocessing
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
#'   print(united_tbl)
#' }
unite_cols <- function(df, listed_vars) {
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
#' Preprocesses text data following the complete workflow implemented in the Shiny application:
#' - Constructing a corpus from united texts
#' - Tokenizing text into words with configurable options
#' - Converting to lowercase with acronym preservation option
#' - Applying character length filtering
#' - Optional multi-word expression detection and compound term creation
#' - Stopword removal and lemmatization capabilities
#'
#' This function serves as the foundation for all subsequent text analysis workflows.
#'
#' @param united_tbl A data frame that contains text data.
#' @param text_field The name of the column that contains the text data.
#' @param min_char The minimum number of characters for a token to be included (default: 2).
#' @param lowercase Logical; convert all tokens to lowercase (default: TRUE). Recommended for most text analysis tasks.
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
#' @param remove_stopwords Logical; remove stopwords from the text (default: FALSE).
#' @param stopwords_source Character; source for stopwords, e.g., "snowball", "stopwords-iso" (default: "snowball").
#' @param stopwords_language Character; language for stopwords (default: "en").
#' @param custom_stopwords Character vector; additional words to remove (default: NULL).
#' @param custom_valuetype Character; valuetype for custom_stopwords pattern matching, one of "glob", "regex", or "fixed" (default: "glob").
#' @param verbose Logical; print verbose output (default: FALSE).
#' @param ... Additional arguments passed to \code{quanteda::tokens}.
#'
#' @return A \code{tokens} object that contains the preprocessed text data.
#'
#' @import quanteda
#'
#' @family preprocessing
#' @export
#'
#' @examples
#' if (interactive()) {
#' mydata <- TextAnalysisR::SpecialEduTech
#'
#' united_tbl <- TextAnalysisR::unite_cols(
#'   mydata,
#'   listed_vars = c("title", "keyword", "abstract")
#' )
#'
#' tokens <- TextAnalysisR::prep_texts(united_tbl,
#'                                          text_field = "united_texts",
#'                                          min_char = 2,
#'                                          lowercase = TRUE,
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
prep_texts <- function(united_tbl,
                             text_field = "united_texts",
                             min_char = 2,
                             lowercase = TRUE,
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
                             remove_stopwords = FALSE,
                             stopwords_source = "snowball",
                             stopwords_language = "en",
                             custom_stopwords = NULL,
                             custom_valuetype = "glob",
                             verbose = FALSE,
                             ...) {

  start_time <- Sys.time()

  tryCatch({
    if (verbose) message("Creating corpus...")
    corp <- quanteda::corpus(united_tbl, text_field = text_field)

    if (verbose) message("Tokenizing texts...")
    toks <- quanteda::tokens(corp,
                   what = "word",
                   remove_punct = remove_punct,
                   remove_symbols = remove_symbols,
                   remove_numbers = remove_numbers,
                   remove_url = remove_url,
                   remove_separators = remove_separators,
                   split_hyphens = split_hyphens,
                   split_tags = split_tags,
                   include_docvars = include_docvars,
                   padding = padding,
                   verbose = verbose,
                   ...)

    if (lowercase) {
      if (verbose) message("Converting to lowercase...")
      toks <- quanteda::tokens_tolower(toks, keep_acronyms = keep_acronyms)
    }

    if (verbose) message("Applying minimum character filter...")
    tokens <- quanteda::tokens_select(toks,
                                      min_nchar = min_char,
                                      verbose = FALSE)

    if (remove_stopwords) {
      if (verbose) message("Removing stopwords (", stopwords_language, ", ", stopwords_source, ")...")
      sw <- stopwords::stopwords(stopwords_language, source = stopwords_source)
      tokens <- quanteda::tokens_remove(tokens, pattern = sw, verbose = FALSE)
    }

    if (!is.null(custom_stopwords) && length(custom_stopwords) > 0) {
      if (verbose) message("Removing custom stopwords (", length(custom_stopwords), " words)...")
      tokens <- quanteda::tokens_remove(tokens, pattern = custom_stopwords,
                                        valuetype = custom_valuetype, verbose = FALSE)
    }

    processing_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

    if (verbose) {
      message("Text preprocessing completed in ", round(processing_time, 2), " seconds")
      message("Processed ", quanteda::ndoc(tokens), " documents with ",
              quanteda::ntoken(tokens), " total tokens")
    }

    return(tokens)

  }, error = function(e) {
    stop("Error in text preprocessing: ", e$message)
  })
}


#' @title Detect Multi-Word Expressions
#'
#' @description
#' This function detects multi-word expressions (collocations) of specified
#' sizes that appear at least a specified number of times in the provided tokens.
#'
#' @param tokens A \code{tokens} object from the \code{quanteda} package.
#' @param size A numeric vector specifying the sizes of the collocations to detect (default: 2:5).
#' @param min_count The minimum number of occurrences for a collocation to be
#'   considered (default: 2).
#'
#' @return A character vector of detected collocations.
#'
#' @family preprocessing
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
#'   collocations <- TextAnalysisR::detect_multi_words(tokens, size = 2:5, min_count = 2)
#'   print(collocations)
#' }
detect_multi_words <- function(tokens, size = 2:5, min_count = 2) {
  tstat <- quanteda.textstats::textstat_collocations(tokens, size = size, min_count = min_count)
  tstat_collocation <- tstat$collocation
  return(tstat_collocation)
}


#' Extract Part-of-Speech Tags from Tokens
#'
#' @description
#' Uses spaCy to extract part-of-speech (POS) tags from tokenized text.
#' Returns a data frame with token-level POS annotations.
#'
#' @param tokens A quanteda tokens object or character vector of texts.
#' @param include_lemma Logical; include lemmatized forms (default: TRUE).
#' @param include_entity Logical; include named entity recognition (default: FALSE).
#' @param include_dependency Logical; include dependency parsing (default: FALSE).
#' @param model Character; spaCy model to use (default: "en_core_web_sm").
#'
#' @return A data frame with columns:
#' \itemize{
#'   \item \code{doc_id}: Document identifier
#'   \item \code{sentence_id}: Sentence number within document
#'   \item \code{token_id}: Token position within sentence
#'   \item \code{token}: Original token
#'   \item \code{pos}: Universal POS tag (e.g., NOUN, VERB, ADJ)
#'   \item \code{tag}: Detailed POS tag (e.g., NN, VBD, JJ)
#'   \item \code{lemma}: Lemmatized form (if include_lemma = TRUE)
#'   \item \code{entity}: Named entity type (if include_entity = TRUE)
#'   \item \code{head_token_id}: Head token in dependency tree (if include_dependency = TRUE)
#'   \item \code{dep_rel}: Dependency relation type, e.g., nsubj, dobj (if include_dependency = TRUE)
#' }
#'
#' @details
#' This function requires the spacyr package and a working Python environment
#' with spaCy installed. If spaCy is not initialized, this function will
#' attempt to initialize it with the specified model.
#'
#' @family preprocessing
#' @export
#'
#' @examples
#' \dontrun{
#' tokens <- quanteda::tokens("The quick brown fox jumps over the lazy dog.")
#' pos_data <- extract_pos_tags(tokens)
#' print(pos_data)
#' }
extract_pos_tags <- function(tokens,
                             include_lemma = TRUE,
                             include_entity = FALSE,
                             include_dependency = FALSE,
                             model = "en_core_web_sm") {

  if (!requireNamespace("spacyr", quietly = TRUE)) {
    stop("Package 'spacyr' is required for POS tagging. Please install it with: install.packages('spacyr')")
  }

  # Convert tokens to text if needed
  if (inherits(tokens, "tokens")) {
    texts <- sapply(tokens, function(x) paste(x, collapse = " "))
  } else if (is.character(tokens)) {
    texts <- tokens
  } else {
    stop("tokens must be a quanteda tokens object or character vector")
  }

  # Try to initialize spaCy if not already done
  tryCatch({
    # Check if spaCy is initialized by trying a simple operation
    spacyr::spacy_parse("test", pos = FALSE, tag = FALSE, lemma = FALSE, entity = FALSE)
  }, error = function(e) {
    message("Initializing spaCy with model: ", model)
    spacyr::spacy_initialize(model = model)
  })

  # Parse with POS and tag enabled
  parsed <- spacyr::spacy_parse(
    texts,
    pos = TRUE,
    tag = TRUE,
    lemma = include_lemma,
    entity = include_entity,
    dependency = include_dependency
  )

  return(parsed)
}


#' Extract Named Entities from Tokens
#'
#' @description
#' Uses spaCy to extract named entities (NER) from tokenized text.
#' Returns a data frame with token-level entity annotations.
#'
#' @param tokens A quanteda tokens object or character vector of texts.
#' @param include_pos Logical; include POS tags (default: TRUE).
#' @param include_lemma Logical; include lemmatized forms (default: TRUE).
#' @param model Character; spaCy model to use (default: "en_core_web_sm").
#'
#' @return A data frame with columns:
#' \itemize{
#'   \item \code{doc_id}: Document identifier
#'   \item \code{token}: Original token
#'   \item \code{entity}: Named entity type (e.g., PERSON, ORG, GPE)
#'   \item \code{pos}: Universal POS tag (if include_pos = TRUE)
#'   \item \code{lemma}: Lemmatized form (if include_lemma = TRUE)
#' }
#'
#' @details
#' This function requires the spacyr package and a working Python environment
#' with spaCy installed. If spaCy is not initialized, this function will
#' attempt to initialize it with the specified model.
#'
#' @family preprocessing
#' @export
#'
#' @examples
#' \dontrun{
#' tokens <- quanteda::tokens("Apple Inc. was founded by Steve Jobs in California.")
#' entity_data <- extract_named_entities(tokens)
#' print(entity_data)
#' }
extract_named_entities <- function(tokens,
                                   include_pos = TRUE,
                                   include_lemma = TRUE,
                                   model = "en_core_web_sm") {

  if (!requireNamespace("spacyr", quietly = TRUE)) {
    stop("Package 'spacyr' is required for NER. Please install it with: install.packages('spacyr')")
  }

  # Convert tokens to text if needed
  if (inherits(tokens, "tokens")) {
    texts <- sapply(tokens, function(x) paste(x, collapse = " "))
  } else if (is.character(tokens)) {
    texts <- tokens
  } else {
    stop("tokens must be a quanteda tokens object or character vector")
  }

  # Try to initialize spaCy if not already done
  tryCatch({
    spacyr::spacy_parse("test", pos = FALSE, tag = FALSE, lemma = FALSE, entity = FALSE)
  }, error = function(e) {
    message("Initializing spaCy with model: ", model)
    spacyr::spacy_initialize(model = model)
  })

  # Parse with entity enabled

  parsed <- spacyr::spacy_parse(
    texts,
    pos = include_pos,
    tag = include_pos,
    lemma = include_lemma,
    entity = TRUE,
    dependency = FALSE
  )

  return(parsed)
}
