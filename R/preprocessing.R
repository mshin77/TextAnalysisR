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


#' Render PDF Pages to Base64 PNG
#'
#' @description
#' Renders each page of a PDF as a PNG image and returns base64-encoded strings.
#' Uses pdftools (R-native, no Python required).
#'
#' @param file_path Character string path to PDF file
#' @param dpi Numeric, rendering resolution (default: 150)
#'
#' @return List of base64-encoded PNG strings, one per page
#' @keywords internal
render_pdf_pages_to_base64 <- function(file_path, dpi = 150) {
  if (!requireNamespace("pdftools", quietly = TRUE)) {
    stop("pdftools package required for PDF rendering")
  }

  n_pages <- pdftools::pdf_info(file_path)$pages
  if (n_pages == 0) return(list())

  pages <- vector("list", n_pages)
  for (i in seq_len(n_pages)) {
    tryCatch({
      tmp <- tempfile(fileext = ".png")
      pdftools::pdf_convert(file_path, format = "png", pages = i,
                            dpi = dpi, filenames = tmp, verbose = FALSE)
      raw_bytes <- readBin(tmp, "raw", file.info(tmp)$size)
      unlink(tmp)
      pages[[i]] <- jsonlite::base64_enc(raw_bytes)
    }, error = function(e) {
      pages[[i]] <<- NULL
    })
  }
  Filter(Negate(is.null), pages)
}


.extract_multimodal_pdf <- function(file_path, vision_provider,
                                    vision_model, api_key,
                                    describe_images) {
  if (is.null(vision_model)) {
    vision_model <- switch(vision_provider,
      "ollama" = "llava",
      "openai" = "gpt-4.1",
      "gemini" = "gemini-2.5-flash",
      "llava"
    )
  }

  if (vision_provider == "ollama") {
    ollama_check <- check_vision_models("ollama")
    if (!isTRUE(ollama_check$available)) {
      stop("Ollama not available: ", ollama_check$message)
    }
  }

  text_pages <- tryCatch(pdftools::pdf_text(file_path), error = function(e) character(0))

  page_images <- render_pdf_pages_to_base64(file_path)

  image_descriptions <- list()
  num_described <- 0

  if (describe_images && length(page_images) > 0) {
    for (i in seq_along(page_images)) {
      page_text_len <- if (i <= length(text_pages)) nchar(trimws(text_pages[i])) else 0
      if (page_text_len > 500) next

      desc <- describe_image(
        image_base64 = page_images[[i]],
        provider = vision_provider,
        model = vision_model,
        api_key = api_key
      )
      if (!is.null(desc)) {
        image_descriptions[[length(image_descriptions) + 1]] <- desc
        num_described <- num_described + 1
      }
    }
  }

  all_text <- paste(trimws(text_pages), collapse = "\n\n")
  if (length(image_descriptions) > 0) {
    all_text <- paste0(all_text, "\n\n", paste(image_descriptions, collapse = "\n\n"))
  }

  combined_lines <- strsplit(all_text, "\n")[[1]]
  combined_lines <- combined_lines[nchar(trimws(combined_lines)) > 0]

  list(
    success = TRUE,
    data = data.frame(text = combined_lines, stringsAsFactors = FALSE),
    type = "multimodal",
    method = "multimodal",
    message = paste("Extracted text and", num_described, "page descriptions"),
    num_images = num_described,
    vision_provider = vision_provider
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
#' Unified PDF processing:
#' 1. Multimodal (R-native pdftools + Vision LLM) if enabled
#' 2. R pdftools text extraction as fallback
#'
#' @param file_path Character string path to PDF file
#' @param use_multimodal Logical, enable multimodal extraction
#' @param vision_provider Character, "ollama", "openai", or "gemini"
#' @param vision_model Character, model name
#' @param api_key Character, API key (if using openai/gemini)
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
        "Error: ", result$message,
        if (vision_provider == "ollama") "\n\nNote: Pull the vision model using terminal/command prompt (not R code): ollama pull llava" else ""
      )
    ))
  }

  tryCatch(
    .extract_r_pdf(file_path),
    error = function(e) {
      list(success = FALSE, data = NULL, type = "error", method = "none",
           message = paste("PDF extraction failed:", e$message))
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
    tidyr::unite(col = "united_texts", sep = " ", remove = FALSE)

  docvar_tbl <- df %>%
    dplyr::select(-all_of(unname(listed_vars)))

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
      if (!requireNamespace("stopwords", quietly = TRUE)) {
        stop("Package 'stopwords' is required for stopword removal. Install with: install.packages('stopwords')")
      }
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


#' Lemmatize Tokens with Batch Processing
#'
#' @description
#' Converts tokens to their lemmatized forms using spaCy, with batch processing
#' to handle large document collections without timeout issues.
#'
#' @param tokens A quanteda tokens object to lemmatize.
#' @param batch_size Integer; number of documents to process per batch (default: 50).
#' @param model Character; spaCy model to use (default: "en_core_web_sm").
#' @param verbose Logical; print progress messages (default: TRUE).
#'
#' @return A quanteda tokens object containing lemmatized tokens.
#'
#' @details
#' Uses spaCy for linguistic lemmatization producing proper dictionary forms
#' (e.g., "studies" -> "study", "better" -> "good").
#' Batch processing prevents timeout errors with large document collections.
#'
#' @family preprocessing
#' @export
#'
#' @examples
#' \dontrun{
#' tokens <- quanteda::tokens(c("The studies showed better results"))
#' lemmatized <- lemmatize_tokens(tokens, batch_size = 50)
#' }
lemmatize_tokens <- function(tokens,
                             batch_size = 50,
                             model = "en_core_web_sm",
                             verbose = TRUE) {

  if (!inherits(tokens, "tokens")) {
    stop("Input must be a quanteda tokens object")
  }

  texts <- sapply(tokens, paste, collapse = " ")
  doc_names <- names(texts)
  n_docs <- length(texts)

  if (n_docs == 0) {
    if (verbose) message("No documents to process")
    return(tokens)
  }

  start_time <- Sys.time()
  all_lemmas <- list()
  n_batches <- ceiling(n_docs / batch_size)

  for (i in seq_len(n_batches)) {
    start_idx <- (i - 1) * batch_size + 1
    end_idx <- min(i * batch_size, n_docs)

    if (verbose) {
      message(sprintf("Lemmatizing batch %d/%d (documents %d-%d)...",
                      i, n_batches, start_idx, end_idx))
    }

    batch_texts <- texts[start_idx:end_idx]
    batch_names <- doc_names[start_idx:end_idx]

    tryCatch({
      parsed <- spacy_parse_full(batch_texts, pos = FALSE, tag = FALSE,
                                 lemma = TRUE, entity = FALSE, model = model)

      for (j in seq_along(batch_names)) {
        doc_id <- batch_names[j]
        doc_lemmas <- parsed$lemma[parsed$doc_id == doc_id]
        all_lemmas[[doc_id]] <- doc_lemmas
      }
    }, error = function(e) {
      warning(sprintf("Error in batch %d: %s", i, e$message))
    })
  }

  if (length(all_lemmas) == 0) {
    stop("All batches failed. Check spaCy installation.")
  }

  lemmatized_tokens <- quanteda::as.tokens(all_lemmas)

  if (verbose) {
    processing_time <- difftime(Sys.time(), start_time, units = "secs")
    message(sprintf("Lemmatization completed in %.2f seconds (%d documents)",
                    as.numeric(processing_time), n_docs))
  }

  return(lemmatized_tokens)
}


#' Extract Text from PDF
#'
#' @description
#' Extracts text content from a PDF file using pdftools package.
#'
#' @param file_path Character string path to PDF file
#'
#' @return Data frame with columns: page (integer), text (character)
#'   Returns NULL if extraction fails or PDF is empty
#'
#' @details
#' Uses `pdftools::pdf_text()` to extract text from each page.
#' Preserves page structure and cleans whitespace.
#' Works best with text-based PDFs (not scanned images).
#'
#' @family pdf
#' @export
#'
#' @examples
#' \dontrun{
#' pdf_path <- "path/to/document.pdf"
#' text_data <- extract_text_from_pdf(pdf_path)
#' head(text_data)
#' }
extract_text_from_pdf <- function(file_path) {
  if (!requireNamespace("pdftools", quietly = TRUE)) {
    stop("Package 'pdftools' is required. Install it with: install.packages('pdftools')")
  }

  tryCatch({
    text_pages <- pdftools::pdf_text(file_path)

    if (length(text_pages) == 0) {
      message("PDF file is empty or contains no extractable text")
      return(NULL)
    }

    text_pages <- trimws(text_pages)
    text_pages <- gsub("\\s+", " ", text_pages)

    non_empty_pages <- which(nchar(text_pages) > 0)

    if (length(non_empty_pages) == 0) {
      message("PDF contains no readable text")
      return(NULL)
    }

    df <- data.frame(
      page = seq_along(text_pages),
      text = text_pages,
      stringsAsFactors = FALSE
    )

    df <- df[nchar(df$text) > 0, ]

    attr(df, "pdf_type") <- "text"
    attr(df, "total_pages") <- length(text_pages)
    attr(df, "file_name") <- basename(file_path)

    return(df)
  }, error = function(e) {
    warning(paste("Error extracting text from PDF:", e$message))
    return(NULL)
  })
}


#' Detect PDF Content Type
#'
#' @description
#' Analyzes PDF to determine if it contains readable text.
#'
#' @param file_path Character string path to PDF file
#'
#' @return Character string: "text" or "unknown"
#'
#' @details
#' Attempts text extraction using pdftools. Returns "text" if successful,
#' or "unknown" if extraction fails or PDF is empty.
#'
#' For table extraction from PDFs, use \code{\link{extract_tables_from_pdf_py}}.
#'
#' @family pdf
#' @export
#'
#' @examples
#' \dontrun{
#' pdf_path <- "path/to/document.pdf"
#' content_type <- detect_pdf_content_type(pdf_path)
#' print(content_type)
#' }
detect_pdf_content_type <- function(file_path) {
  if (requireNamespace("pdftools", quietly = TRUE)) {
    text <- tryCatch({
      pdftools::pdf_text(file_path)
    }, error = function(e) NULL)

    if (!is.null(text) && length(text) > 0) {
      combined_text <- paste(text, collapse = " ")
      if (nchar(trimws(combined_text)) > 50) {
        return("text")
      }
    }
  }

  return("unknown")
}


#' Process PDF File
#'
#' @description
#' Main function to process PDF files - extracts text content using pdftools.
#' For table extraction, use \code{\link{process_pdf_file_py}}.
#'
#' @param file_path Character string path to PDF file
#' @param content_type Character string: "auto" or "text" (default: "auto")
#'
#' @return List with:
#'   - data: Data frame with extracted content
#'   - type: Character string indicating content type ("text" or "error")
#'   - success: Logical indicating success
#'   - message: Character string with status message
#'
#' @details
#' This function extracts text content from PDFs using pdftools package.
#' Works best with text-based PDFs (not scanned images).
#'
#' For PDFs containing tables or complex layouts, use the Python-based
#' \code{\link{process_pdf_file_py}} which provides better table extraction.
#'
#' @family pdf
#' @export
#'
#' @examples
#' \dontrun{
#' pdf_path <- "path/to/document.pdf"
#' result <- process_pdf_file(pdf_path)
#'
#' if (result$success) {
#'   print(head(result$data))
#' } else {
#'   print(result$message)
#' }
#' }
process_pdf_file <- function(file_path, content_type = "auto") {
  if (!file.exists(file_path)) {
    return(list(
      data = NULL,
      type = "error",
      success = FALSE,
      message = "File not found"
    ))
  }

  data <- extract_text_from_pdf(file_path)

  if (!is.null(data)) {
    return(list(
      data = data,
      type = "text",
      success = TRUE,
      message = paste("Successfully extracted text from", nrow(data), "pages")
    ))
  }

  return(list(
    data = NULL,
    type = "error",
    success = FALSE,
    message = "Could not extract content from PDF. File may be password-protected, corrupted, or scanned without OCR."
  ))
}



#' Extract Text from PDF using Python
#'
#' @description
#' Extracts text content from a PDF file using pdfplumber (Python).
#' No Java required - uses Python environment.
#'
#' @param file_path Character string path to PDF file
#' @param envname Character string, name of Python virtual environment
#'   (default: "textanalysisr-env")
#'
#' @return Data frame with columns: page (integer), text (character)
#'   Returns NULL if extraction fails or PDF is empty
#'
#' @details
#' Uses pdfplumber Python library through reticulate.
#' Requires Python environment setup. See \code{setup_python_env()}.
#'
#' @family pdf
#' @export
#'
#' @examples
#' \dontrun{
#' setup_python_env()
#'
#' pdf_path <- "path/to/document.pdf"
#' text_data <- extract_text_from_pdf_py(pdf_path)
#' head(text_data)
#' }
extract_text_from_pdf_py <- function(file_path, envname = "textanalysisr-env") {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required.")
  }

  status <- check_python_env(envname)
  if (!status$available) {
    stop("Python environment '", envname, "' not found. Run setup_python_env() first.")
  }

  tryCatch({
    reticulate::use_virtualenv(envname, required = TRUE)

    pdf_module <- reticulate::import_from_path(
      "pdf_extraction",
      path = system.file("python", package = "TextAnalysisR")
    )

    result <- pdf_module$extract_text_from_pdf(file_path)

    if (!result$success) {
      message(result$message)
      return(NULL)
    }

    df <- do.call(rbind, lapply(result$data, as.data.frame))
    df$page <- as.integer(df$page)
    df$text <- as.character(df$text)

    attr(df, "pdf_type") <- "text"
    attr(df, "total_pages") <- result$total_pages
    attr(df, "file_name") <- basename(file_path)

    return(df)

  }, error = function(e) {
    warning(paste("Error extracting text from PDF:", e$message))
    return(NULL)
  })
}


#' Extract Tables from PDF using Python
#'
#' @description
#' Extracts tabular data from PDF using pdfplumber (Python).
#' No Java required - pure Python solution.
#'
#' @param file_path Character string path to PDF file
#' @param pages Integer vector of page numbers to process (NULL = all pages)
#' @param envname Character string, name of Python virtual environment
#'   (default: "textanalysisr-env")
#'
#' @return Data frame with extracted table data
#'   Returns NULL if no tables found or extraction fails
#'
#' @details
#' Uses pdfplumber Python library through reticulate.
#' Works with complex table layouts without Java dependency.
#'
#' @family pdf
#' @export
#'
#' @examples
#' \dontrun{
#' setup_python_env()
#'
#' pdf_path <- "path/to/table_document.pdf"
#' table_data <- extract_tables_from_pdf_py(pdf_path)
#' head(table_data)
#' }
extract_tables_from_pdf_py <- function(file_path, pages = NULL, envname = "textanalysisr-env") {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required.")
  }

  status <- check_python_env(envname)
  if (!status$available) {
    stop("Python environment '", envname, "' not found. Run setup_python_env() first.")
  }

  tryCatch({
    reticulate::use_virtualenv(envname, required = TRUE)

    pdf_module <- reticulate::import_from_path(
      "pdf_extraction",
      path = system.file("python", package = "TextAnalysisR")
    )

    pages_list <- if (!is.null(pages)) as.list(as.integer(pages)) else NULL

    result <- pdf_module$extract_tables_from_pdf(file_path, pages = pages_list)

    if (!result$success) {
      message(result$message)
      return(NULL)
    }

    df <- as.data.frame(result$data, stringsAsFactors = FALSE)

    if (nrow(df) == 0) {
      message("Table extraction resulted in empty data frame")
      return(NULL)
    }

    attr(df, "pdf_type") <- "tabular"
    attr(df, "file_name") <- basename(file_path)
    attr(df, "num_tables") <- result$num_tables

    return(df)

  }, error = function(e) {
    warning(paste("Error extracting tables from PDF:", e$message))
    return(NULL)
  })
}


#' Detect PDF Content Type using Python
#'
#' @description
#' Analyzes PDF to determine if it contains primarily tabular data or text.
#'
#' @param file_path Character string path to PDF file
#' @param envname Character string, name of Python virtual environment
#'   (default: "textanalysisr-env")
#'
#' @return Character string: "tabular", "text", or "unknown"
#'
#' @family pdf
#' @export
#'
#' @examples
#' \dontrun{
#' setup_python_env()
#'
#' pdf_path <- "path/to/document.pdf"
#' content_type <- detect_pdf_content_type_py(pdf_path)
#' print(content_type)
#' }
detect_pdf_content_type_py <- function(file_path, envname = "textanalysisr-env") {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required.")
  }

  status <- check_python_env(envname)
  if (!status$available) {
    return("unknown")
  }

  tryCatch({
    reticulate::use_virtualenv(envname, required = TRUE)

    pdf_module <- reticulate::import_from_path(
      "pdf_extraction",
      path = system.file("python", package = "TextAnalysisR")
    )

    result <- pdf_module$detect_pdf_content_type(file_path)

    return(result$content_type)

  }, error = function(e) {
    return("unknown")
  })
}


#' Process PDF File using Python
#'
#' @description
#' Main function to process PDF files using pdfplumber (Python).
#' Automatically detects content type and extracts data accordingly.
#' No Java required.
#'
#' @param file_path Character string path to PDF file
#' @param content_type Character string: "auto", "text", or "tabular"
#'   If "auto", will detect content type automatically
#' @param envname Character string, name of Python virtual environment
#'   (default: "textanalysisr-env")
#'
#' @return List with:
#'   - data: Data frame with extracted content
#'   - type: Character string indicating content type
#'   - success: Logical indicating success
#'   - message: Character string with status message
#'
#' @details
#' This function uses Python's pdfplumber library which:
#' - Handles both text and tables
#' - No Java dependency
#' - Better accuracy than tabulizer for complex tables
#' - Uses TextAnalysisR Python environment
#'
#' @family pdf
#' @export
#'
#' @examples
#' \dontrun{
#' setup_python_env()
#'
#' pdf_path <- "path/to/document.pdf"
#' result <- process_pdf_file_py(pdf_path)
#'
#' if (result$success) {
#'   print(head(result$data))
#' } else {
#'   print(result$message)
#' }
#' }
process_pdf_file_py <- function(file_path, content_type = "auto", envname = "textanalysisr-env") {
  if (!file.exists(file_path)) {
    return(list(
      data = NULL,
      type = "error",
      success = FALSE,
      message = "File not found"
    ))
  }

  if (!requireNamespace("reticulate", quietly = TRUE)) {
    return(list(
      data = NULL,
      type = "error",
      success = FALSE,
      message = "Package 'reticulate' is required"
    ))
  }

  status <- check_python_env(envname)
  if (!status$available) {
    return(list(
      data = NULL,
      type = "error",
      success = FALSE,
      message = paste("Python environment", envname, "not found. Run setup_python_env() first.")
    ))
  }

  tryCatch({
    reticulate::use_virtualenv(envname, required = TRUE)

    pdf_module <- reticulate::import_from_path(
      "pdf_extraction",
      path = system.file("python", package = "TextAnalysisR")
    )

    result <- pdf_module$process_pdf_file(file_path, content_type = content_type)

    if (!result$success) {
      return(list(
        data = NULL,
        type = result$type,
        success = FALSE,
        message = result$message
      ))
    }

    if (result$type == "tabular") {
      df <- as.data.frame(result$data, stringsAsFactors = FALSE)
    } else {
      df <- do.call(rbind, lapply(result$data, as.data.frame))
      df$page <- as.integer(df$page)
      df$text <- as.character(df$text)

      if (!"category" %in% names(df)) {
        df$category <- "Uploaded"
      }
    }

    return(list(
      data = df,
      type = result$type,
      success = TRUE,
      message = result$message
    ))

  }, error = function(e) {
    return(list(
      data = NULL,
      type = "error",
      success = FALSE,
      message = paste("Error processing PDF:", e$message)
    ))
  })
}




#' Check Multimodal Prerequisites
#'
#' @description
#' Checks all prerequisites for multimodal PDF extraction.
#' Uses R-native pdftools for rendering (no Python required).
#'
#' @param vision_provider Character: "ollama", "openai", or "gemini"
#' @param vision_model Character: Model name (optional)
#' @param api_key Character: API key for OpenAI/Gemini (if using cloud provider)
#' @param envname Character: Kept for backward compatibility, ignored
#'
#' @return List with:
#'   - ready: Logical - TRUE if all prerequisites met
#'   - missing: Character vector of missing components
#'   - instructions: Character - Detailed setup instructions
#'   - details: List with component-specific status
#'
#' @keywords internal
check_multimodal_prerequisites <- function(
  vision_provider = "ollama",
  vision_model = NULL,
  api_key = NULL,
  envname = "textanalysisr-env"
) {
  missing <- character(0)
  details <- list()
  instructions <- character(0)

  pdftools_ok <- requireNamespace("pdftools", quietly = TRUE)
  if (!pdftools_ok) {
    missing <- c(missing, "pdftools package")
    instructions <- c(instructions,
      "pdftools package required.\nInstall: install.packages('pdftools')"
    )
  }
  details$pdftools <- list(available = pdftools_ok)

  if (vision_provider == "ollama") {
    ollama_ok <- check_ollama(verbose = FALSE)

    if (!ollama_ok) {
      missing <- c(missing, "Ollama")
      instructions <- c(instructions, paste0(
        "Ollama not running.\n",
        "1. Start Ollama application\n",
        "2. Pull vision model: ollama pull llava"
      ))
    } else if (!is.null(vision_model)) {
      models_available <- tryCatch(list_ollama_models(), error = function(e) character(0))
      if (!vision_model %in% models_available) {
        missing <- c(missing, paste("Vision model:", vision_model))
        instructions <- c(instructions, paste0(
          "Vision model '", vision_model, "' not found.\n",
          "Pull model: ollama pull ", vision_model
        ))
      }
    }
    details$ollama <- list(available = ollama_ok)

  } else if (vision_provider == "openai") {
    if (is.null(api_key) || nchar(api_key) == 0) {
      missing <- c(missing, "OpenAI API key")
      instructions <- c(instructions, .missing_api_key_message("openai", "shiny"))
    }
    details$openai <- list(api_key_provided = !is.null(api_key) && nchar(api_key) > 0)

  } else if (vision_provider == "gemini") {
    if (is.null(api_key) || nchar(api_key) == 0) {
      missing <- c(missing, "Gemini API key")
      instructions <- c(instructions, .missing_api_key_message("gemini", "shiny"))
    }
    details$gemini <- list(api_key_provided = !is.null(api_key) && nchar(api_key) > 0)
  }

  ready <- length(missing) == 0

  if (!ready) {
    full_instructions <- paste0(
      "Multimodal extraction requires:\n\n",
      paste(paste0(seq_along(instructions), ". ", instructions), collapse = "\n\n"),
      if (vision_provider == "ollama") "\n\nNote: Pull the vision model using terminal/command prompt (not R code): ollama pull llava" else ""
    )
  } else {
    full_instructions <- "All prerequisites met"
  }

  return(list(
    ready = ready,
    missing = missing,
    instructions = full_instructions,
    details = details
  ))
}
#' Extract PDF with Multimodal Analysis
#'
#' @description
#' Extract both text and visual content from PDFs using R-native pdftools
#' and vision LLM APIs. No Python required.
#'
#' @param file_path Character string path to PDF file
#' @param vision_provider Character: "ollama" (local, default), "openai", or "gemini"
#' @param vision_model Character: Model name
#'   - For Ollama: "llava", "llava:13b", "bakllava"
#'   - For OpenAI: "gpt-4.1", "gpt-4.1-mini"
#'   - For Gemini: "gemini-2.5-flash", "gemini-2.5-pro"
#' @param api_key Character: API key (required for openai/gemini providers)
#' @param describe_images Logical: Convert page images to text descriptions (default: TRUE)
#' @param envname Character: Kept for backward compatibility, ignored
#'
#' @return List with:
#'   - success: Logical
#'   - combined_text: Character string with all content for text analysis
#'   - text_content: List of text chunks
#'   - image_descriptions: List of image descriptions
#'   - num_images: Integer count of described pages
#'   - vision_provider: Character indicating provider used
#'   - message: Character status message
#'
#' @details
#' **Workflow:**
#' 1. Extracts text using pdftools (R-native)
#' 2. Renders each page as an image
#' 3. Sends sparse-text pages to vision LLM for description
#' 4. Merges text + descriptions into a single text corpus
#'
#' @family pdf
#' @export
#'
#' @examples
#' \dontrun{
#' result <- extract_pdf_multimodal("research_paper.pdf")
#' text_for_analysis <- result$combined_text
#'
#' result <- extract_pdf_multimodal(
#'   "paper.pdf",
#'   vision_provider = "gemini",
#'   api_key = Sys.getenv("GEMINI_API_KEY")
#' )
#' }
extract_pdf_multimodal <- function(
  file_path,
  vision_provider = "ollama",
  vision_model = NULL,
  api_key = NULL,
  describe_images = TRUE,
  envname = "textanalysisr-env"
) {
  if (!file.exists(file_path)) {
    return(list(success = FALSE, message = "File not found"))
  }

  if (is.null(vision_model)) {
    vision_model <- switch(vision_provider,
      "ollama" = "llava",
      "openai" = "gpt-4.1",
      "gemini" = "gemini-2.5-flash",
      "llava"
    )
  }

  if (vision_provider == "ollama") {
    if (!check_ollama(verbose = FALSE)) {
      return(list(
        success = FALSE,
        message = paste(
          "Ollama not available. Please:",
          "1. Install Ollama from https://ollama.com",
          "2. Pull a vision model: ollama pull llava",
          sep = "\n"
        )
      ))
    }
  } else if (vision_provider == "openai") {
    if (is.null(api_key) || !nzchar(api_key)) {
      return(list(success = FALSE, message = .missing_api_key_message("openai", "shiny")))
    }
  } else if (vision_provider == "gemini") {
    if (is.null(api_key) || !nzchar(api_key)) {
      return(list(success = FALSE, message = .missing_api_key_message("gemini", "shiny")))
    }
  }

  tryCatch({
    text_pages <- pdftools::pdf_text(file_path)
    text_content <- list(paste(trimws(text_pages), collapse = "\n\n"))

    image_descriptions <- list()
    num_described <- 0

    if (describe_images) {
      page_images <- render_pdf_pages_to_base64(file_path)

      for (i in seq_along(page_images)) {
        page_text_len <- if (i <= length(text_pages)) nchar(trimws(text_pages[i])) else 0
        if (page_text_len > 500) next

        desc <- describe_image(
          image_base64 = page_images[[i]],
          provider = vision_provider,
          model = vision_model,
          api_key = api_key
        )
        if (!is.null(desc)) {
          image_descriptions[[length(image_descriptions) + 1]] <- desc
          num_described <- num_described + 1
        }
      }
    }

    combined_text <- paste(trimws(text_pages), collapse = "\n\n")
    if (length(image_descriptions) > 0) {
      combined_text <- paste0(combined_text, "\n\n", paste(image_descriptions, collapse = "\n\n"))
    }

    return(list(
      success = TRUE,
      text_content = text_content,
      image_descriptions = image_descriptions,
      combined_text = combined_text,
      total_pages = length(text_pages),
      num_images = num_described,
      vision_provider = vision_provider,
      message = paste("Extracted text and", num_described, "page descriptions")
    ))

  }, error = function(e) {
    return(list(success = FALSE, message = paste("Error:", e$message)))
  })
}


#' Smart PDF Extraction with Auto-Detection
#'
#' @description
#' Extracts text and visual content from PDFs using R-native pdftools
#' and vision LLM APIs. Routes directly to multimodal extraction.
#'
#' @param file_path Character string path to PDF file
#' @param doc_type Character: "auto" (default), "academic", or "general" (kept for compatibility)
#' @param vision_provider Character: "ollama" (default), "openai", or "gemini"
#' @param vision_model Character: Model name for vision analysis
#' @param api_key Character: API key for cloud providers
#' @param envname Character: Kept for backward compatibility, ignored
#'
#' @return List with extracted content ready for text analysis
#'
#' @family pdf
#' @export
#'
#' @examples
#' \dontrun{
#' result <- extract_pdf_smart("document.pdf")
#' corpus <- prep_texts(result$combined_text)
#' }
extract_pdf_smart <- function(
  file_path,
  doc_type = "auto",
  vision_provider = "ollama",
  vision_model = NULL,
  api_key = NULL,
  envname = "textanalysisr-env"
) {
  if (!file.exists(file_path)) {
    return(list(success = FALSE, message = "File not found"))
  }

  extract_pdf_multimodal(
    file_path = file_path,
    vision_provider = vision_provider,
    vision_model = vision_model,
    api_key = api_key,
    describe_images = TRUE
  )
}


#' Check Vision Model Availability
#'
#' @description
#' Check if required vision models are available for multimodal processing.
#'
#' @param provider Character: "ollama", "openai", or "gemini"
#' @param api_key Character: API key (for OpenAI/Gemini)
#'
#' @return List with availability status and recommendations
#'
#' @family pdf
#' @export
#'
#' @examples
#' \dontrun{
#' status <- check_vision_models("ollama")
#' status <- check_vision_models("gemini", api_key = Sys.getenv("GEMINI_API_KEY"))
#' }
check_vision_models <- function(provider = "ollama", api_key = NULL) {
  if (provider == "ollama") {
    if (!check_ollama(verbose = FALSE)) {
      return(list(
        available = FALSE,
        models = character(0),
        message = "Ollama not running. Install from https://ollama.com"
      ))
    }

    models <- list_ollama_models(verbose = FALSE)
    vision_models <- grep("llava|bakllava|llava-phi3", models, value = TRUE)

    if (length(vision_models) == 0) {
      return(list(
        available = FALSE,
        models = character(0),
        message = paste(
          "No vision models found. Pull one with:",
          "  ollama pull llava",
          "  ollama pull bakllava",
          "  ollama pull llava-phi3",
          sep = "\n"
        )
      ))
    }

    return(list(
      available = TRUE,
      models = vision_models,
      message = paste("Found", length(vision_models), "vision model(s)")
    ))

  } else if (provider == "openai") {
    if (is.null(api_key) || !nzchar(api_key)) {
      return(list(available = FALSE, message = "OpenAI API key required"))
    }

    valid <- nchar(api_key) > 20 && grepl("^sk-", api_key)
    return(list(
      available = valid,
      message = if (valid) "API key format valid" else "Invalid API key format"
    ))

  } else if (provider == "gemini") {
    if (is.null(api_key) || !nzchar(api_key)) {
      return(list(available = FALSE, message = "Gemini API key required"))
    }

    valid <- nchar(api_key) > 10
    return(list(
      available = valid,
      message = if (valid) "API key provided" else "Invalid API key"
    ))
  }

  return(list(
    available = FALSE,
    message = paste("Unknown provider:", provider)
  ))
}
