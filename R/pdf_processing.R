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
