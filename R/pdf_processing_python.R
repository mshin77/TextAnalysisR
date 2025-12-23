#' Extract Text from PDF using Python
#'
#' @description
#' Extracts text content from a PDF file using pdfplumber (Python).
#' No Java required - uses Python environment.
#'
#' @param file_path Character string path to PDF file
#' @param envname Character string, name of Python virtual environment
#'   (default: "langgraph-env")
#'
#' @return Data frame with columns: page (integer), text (character)
#'   Returns NULL if extraction fails or PDF is empty
#'
#' @details
#' Uses pdfplumber Python library through reticulate.
#' Requires Python environment setup. See \code{setup_langgraph_env()}.
#'
#' @family pdf
#' @export
#'
#' @examples
#' \dontrun{
#' setup_langgraph_env()
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
    stop("Python environment '", envname, "' not found. Run setup_langgraph_env() first.")
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
#'   (default: "langgraph-env")
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
#' setup_langgraph_env()
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
    stop("Python environment '", envname, "' not found. Run setup_langgraph_env() first.")
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
#'   (default: "langgraph-env")
#'
#' @return Character string: "tabular", "text", or "unknown"
#'
#' @family pdf
#' @export
#'
#' @examples
#' \dontrun{
#' setup_langgraph_env()
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
#'   (default: "langgraph-env")
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
#' - Uses same Python environment as LangGraph
#'
#' @family pdf
#' @export
#'
#' @examples
#' \dontrun{
#' setup_langgraph_env()
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
      message = paste("Python environment", envname, "not found. Run setup_langgraph_env() first.")
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
