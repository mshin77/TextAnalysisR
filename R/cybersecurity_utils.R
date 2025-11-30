#' Cybersecurity Utility Functions
#'
#' @description Functions for input validation, sanitization, and security logging
#'
#' @section NIST Compliance:
#' This package follows NIST security standards (based on NIST SP 800-53):
#' - SC-8: Transmission Confidentiality and Integrity (HTTPS encryption)
#' - SC-28: Protection of Information at Rest (secure API key storage)
#' - IA-5: Authenticator Management (API key validation and format checking)
#' - AC-3: Access Enforcement (rate limiting, input validation, file type restrictions)
#' - SI-10: Information Input Validation (malicious content detection)
#' - AU-2: Audit Events (security logging and monitoring)

#' Validate File Upload
#'
#' @param file_info File info object from Shiny fileInput
#' @return TRUE if valid, stops with error message if invalid
#' @keywords internal
validate_file_upload <- function(file_info) {
  if (is.null(file_info)) {
    stop("No file provided")
  }

  allowed_extensions <- c(".csv", ".xlsx", ".txt", ".rds", ".pdf", ".docx")
  ext <- tools::file_ext(file_info$name)

  if (!paste0(".", tolower(ext)) %in% allowed_extensions) {
    stop("Invalid file type. Allowed types: CSV, XLSX, TXT, RDS, PDF, DOCX")
  }

  max_size <- 50 * 1024 * 1024
  if (file_info$size > max_size) {
    stop("File size exceeds maximum limit of 50MB")
  }

  if (ext %in% c("csv", "txt")) {
    content <- tryCatch({
      suppressWarnings(readLines(file_info$datapath, n = 10, warn = FALSE))
    }, error = function(e) {
      stop("Unable to read file contents")
    })

    suspicious_patterns <- c("<script", "javascript:", "onerror=", "onclick=")
    if (any(grepl(paste(suspicious_patterns, collapse = "|"), content, ignore.case = TRUE))) {
      stop("File contains potentially malicious content")
    }
  }

  return(TRUE)
}

#' Sanitize Text Input
#'
#' @param text Text input from user
#' @return Sanitized text
#' @keywords internal
sanitize_text_input <- function(text) {
  if (is.null(text) || nchar(text) == 0) {
    return(text)
  }

  text <- gsub("<script.*?>.*?</script>", "", text, ignore.case = TRUE)
  text <- gsub("javascript:", "", text, ignore.case = TRUE)
  text <- gsub("on\\w+\\s*=", "", text, ignore.case = TRUE)

  max_chars <- 1000000
  if (nchar(text) > max_chars) {
    stop("Text input exceeds maximum length of 1 million characters")
  }

  return(text)
}

#' Check Rate Limit
#'
#' @param session_token Shiny session token
#' @param user_requests Reactive value storing request history
#' @param max_requests Maximum requests allowed in time window
#' @param window_seconds Time window in seconds (default: 3600 = 1 hour)
#' @return TRUE if within limit, stops with error if exceeded
#' @keywords internal
check_rate_limit <- function(session_token, user_requests, max_requests = 100, window_seconds = 3600) {
  current_time <- Sys.time()
  requests <- user_requests()

  if (is.null(requests[[session_token]])) {
    requests[[session_token]] <- list()
  }

  requests[[session_token]] <- Filter(function(x) {
    difftime(current_time, x, units = "secs") < window_seconds
  }, requests[[session_token]])

  if (length(requests[[session_token]]) >= max_requests) {
    stop("Rate limit exceeded. You have made too many requests. Please wait before trying again.")
  }

  requests[[session_token]] <- c(requests[[session_token]], current_time)
  user_requests(requests)

  return(TRUE)
}

#' Log Security Event
#'
#' @param event_type Type of security event
#' @param details Additional details about the event
#' @param session_info Shiny session object
#' @param level Log level (INFO, WARNING, ERROR)
#' @keywords internal
log_security_event <- function(event_type, details, session_info, level = "INFO") {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")

  session_token <- if (!is.null(session_info$token)) {
    substr(session_info$token, 1, 8)
  } else {
    "unknown"
  }

  log_message <- paste0(
    "[", timestamp, "] ",
    "[", level, "] ",
    "SECURITY: ", event_type, " | ",
    "Session: ", session_token, "... | ",
    "Details: ", details
  )

  log_file <- "security.log"
  tryCatch({
    cat(log_message, "\n", file = log_file, append = TRUE)
  }, error = function(e) {
    warning("Failed to write security log: ", e$message)
  })

  if (Sys.getenv("ENVIRONMENT") != "production") {
    message(log_message)
  }

  return(invisible(NULL))
}

#' Validate OpenAI API Key Format
#'
#' @description
#' Validates OpenAI API key format according to NIST IA-5(1) authenticator management.
#' Checks key prefix, length, and basic format requirements.
#'
#' @param api_key Character string containing the API key
#' @param strict Logical, if TRUE performs additional validation checks
#'
#' @return Logical TRUE if valid, FALSE with warnings if invalid
#' @keywords internal
#'
#' @section NIST Compliance:
#' Implements NIST IA-5(1): Authenticator Management - Password-Based Authentication.
#' Validates format, length, and character composition to prevent weak or malformed keys.
#'
#' @examples
#' \dontrun{
#' validate_api_key("sk-proj...")
#' }
validate_api_key <- function(api_key, strict = TRUE) {
  if (is.null(api_key) || !is.character(api_key) || !nzchar(api_key)) {
    warning("API key is NULL, empty, or not a character string")
    return(FALSE)
  }

  if (!grepl("^sk-", api_key)) {
    warning("API key format appears invalid: OpenAI keys should start with 'sk-'")
    return(FALSE)
  }

  if (nchar(api_key) < 40) {
    warning("API key appears too short: OpenAI keys are typically 48+ characters")
    return(FALSE)
  }

  if (strict) {
    if (grepl("\\s", api_key)) {
      warning("API key contains whitespace characters")
      return(FALSE)
    }

    if (grepl("[^A-Za-z0-9_-]", api_key)) {
      warning("API key contains unexpected special characters")
      return(FALSE)
    }
  }

  return(TRUE)
}

#' Validate Column Name
#'
#' @description
#' Validates column names to prevent code injection through formula construction.
#' Ensures column names follow R naming conventions and contain no malicious patterns.
#'
#' @param col_name Character string containing the column name
#'
#' @return TRUE if valid, stops with error if invalid
#' @keywords internal
#'
#' @section Security:
#' Protects against formula injection attacks where malicious column names could
#' execute arbitrary code when used in model formulas. Part of NIST SI-10 input validation.
#'
#' @examples
#' \dontrun{
#' validate_column_name("age")
#' validate_column_name("my_variable")
#' }
validate_column_name <- function(col_name) {
  if (is.null(col_name) || !is.character(col_name) || !nzchar(col_name)) {
    stop("Column name is NULL, empty, or not a character string")
  }

  if (length(col_name) != 1) {
    stop("Column name must be a single value, not a vector")
  }

  if (!grepl("^[A-Za-z][A-Za-z0-9_\\.]*$", col_name)) {
    stop("Invalid column name format. Column names must start with a letter and contain only letters, numbers, underscores, or periods.")
  }

  if (nchar(col_name) > 255) {
    stop("Column name exceeds maximum length of 255 characters")
  }

  backticks <- grepl("`", col_name, fixed = TRUE)
  if (backticks) {
    stop("Column name contains backticks which are not allowed")
  }

  return(TRUE)
}
