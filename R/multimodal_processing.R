#' Check Multimodal Prerequisites
#'
#' @description
#' Checks all prerequisites for multimodal PDF extraction and returns
#' detailed status with setup instructions.
#'
#' @param vision_provider Character: "ollama" or "openai"
#' @param vision_model Character: Model name (optional)
#' @param api_key Character: API key for OpenAI (if using openai provider)
#' @param envname Character: Python environment name
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
  envname = "langgraph-env"
) {
  missing <- character(0)
  details <- list()
  instructions <- character(0)

  py_available <- FALSE
  py_packages_ok <- FALSE

  tryCatch({
    if (requireNamespace("reticulate", quietly = TRUE)) {
      reticulate::use_condaenv(envname, required = FALSE)
      py_available <- reticulate::py_available(initialize = FALSE)

      if (py_available) {
        required_packages <- c("pdf2image", "PIL", "requests")
        py_packages_ok <- all(sapply(required_packages, function(pkg) {
          tryCatch({
            reticulate::py_module_available(pkg)
          }, error = function(e) FALSE)
        }))
      }
    }
  }, error = function(e) {
    py_available <- FALSE
  })

  if (!py_available) {
    missing <- c(missing, "Python environment")
    instructions <- c(instructions, paste0(
      "Python environment '", envname, "' not found.\n",
      "Setup: library(TextAnalysisR); setup_python_env()"
    ))
  } else if (!py_packages_ok) {
    missing <- c(missing, "Python packages")
    instructions <- c(instructions, paste0(
      "Required Python packages missing.\n",
      "Setup: library(TextAnalysisR); setup_python_env()"
    ))
  }

  details$python <- list(
    available = py_available,
    packages_ok = py_packages_ok
  )

  if (vision_provider == "ollama") {
    ollama_ok <- check_ollama(verbose = FALSE)

    if (!ollama_ok) {
      missing <- c(missing, "Ollama")
      instructions <- c(instructions, paste0(
        "Ollama not running.\n",
        "1. Start Ollama application\n",
        "2. Pull vision model: ollama pull llava"
      ))
    } else {
      if (!is.null(vision_model)) {
        models_available <- tryCatch({
          list_ollama_models()
        }, error = function(e) character(0))

        if (!vision_model %in% models_available) {
          missing <- c(missing, paste("Vision model:", vision_model))
          instructions <- c(instructions, paste0(
            "Vision model '", vision_model, "' not found.\n",
            "Pull model: ollama pull ", vision_model
          ))
        }
      }
    }

    details$ollama <- list(available = ollama_ok)

  } else if (vision_provider == "openai") {
    if (is.null(api_key) || nchar(api_key) == 0) {
      missing <- c(missing, "OpenAI API key")
      instructions <- c(instructions, paste0(
        "OpenAI API key required.\n",
        "Provide your API key in the 'OpenAI API Key' field"
      ))
    }

    details$openai <- list(api_key_provided = !is.null(api_key))
  }

  ready <- length(missing) == 0

  if (!ready) {
    full_instructions <- paste0(
      "Multimodal extraction requires:\n\n",
      paste(paste0(seq_along(instructions), ". ", instructions), collapse = "\n\n"),
      "\n\nNote: Pull the vision model using terminal/command prompt (not R code): ollama pull llava"
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
#' Extract both text and visual content from PDFs, converting everything
#' to text for downstream analysis in your existing workflow.
#'
#' @param file_path Character string path to PDF file
#' @param vision_provider Character: "ollama" (local, default) or "openai" (cloud)
#' @param vision_model Character: Model name
#'   - For Ollama: "llava", "llava:13b", "bakllava"
#'   - For OpenAI: "gpt-4-vision-preview", "gpt-4o"
#' @param api_key Character: OpenAI API key (required if vision_provider="openai")
#' @param describe_images Logical: Convert images to text descriptions (default: TRUE)
#' @param envname Character: Python environment name (default: "langgraph-env")
#'
#' @return List with:
#'   - success: Logical
#'   - combined_text: Character string with all content for text analysis
#'   - text_content: List of text chunks
#'   - image_descriptions: List of image descriptions
#'   - num_images: Integer count of processed images
#'   - vision_provider: Character indicating provider used
#'   - message: Character status message
#'
#' @details
#' **Workflow Integration:**
#' 1. Extracts text using Marker (preserves equations, tables, structure)
#' 2. Detects images/charts/diagrams in PDF
#' 3. Uses vision LLM to describe visual content as text
#' 4. Merges text + descriptions → single text corpus
#' 5. Feed to existing text analysis pipeline
#'
#' **Vision Provider Options:**
#'
#' **Ollama (Default - Local & Free):**
#' - Privacy: Everything runs locally
#' - Cost: Free
#' - Setup: Requires Ollama installed + vision model pulled
#' - Models: llava, bakllava, llava-phi3
#'
#' **OpenAI (Optional - Cloud):**
#' - Privacy: Data sent to OpenAI
#' - Cost: Paid (user's API key)
#' - Setup: Just provide API key
#' - Models: gpt-4-vision-preview, gpt-4o
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Local analysis with Ollama (free, private)
#' result <- extract_pdf_multimodal("research_paper.pdf")
#'
#' # Access combined text for analysis
#' text_for_analysis <- result$combined_text
#'
#' # Use in existing workflow
#' corpus <- prep_texts(text_for_analysis)
#' topics <- fit_semantic_model(corpus, k = 5)
#'
#' # Optional: Use OpenAI for better accuracy
#' result <- extract_pdf_multimodal(
#'   "paper.pdf",
#'   vision_provider = "openai",
#'   vision_model = "gpt-4o",
#'   api_key = Sys.getenv("OPENAI_API_KEY")
#' )
#' }
extract_pdf_multimodal <- function(
  file_path,
  vision_provider = "ollama",
  vision_model = NULL,
  api_key = NULL,
  describe_images = TRUE,
  envname = "langgraph-env"
) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required.")
  }

  if (!file.exists(file_path)) {
    return(list(
      success = FALSE,
      message = "File not found"
    ))
  }

  # Set default models
  if (is.null(vision_model)) {
    vision_model <- if (vision_provider == "ollama") "llava" else "gpt-4-vision-preview"
  }

  # Check prerequisites
  if (vision_provider == "ollama") {
    if (!check_ollama(verbose = FALSE)) {
      return(list(
        success = FALSE,
        message = paste(
          "Ollama not available. Please:",
          "1. Install Ollama from https://ollama.ai",
          "2. Pull a vision model: ollama pull llava",
          sep = "\n"
        )
      ))
    }
  } else if (vision_provider == "openai") {
    if (is.null(api_key)) {
      return(list(
        success = FALSE,
        message = "OpenAI API key required for vision_provider='openai'"
      ))
    }
  }

  tryCatch({
    reticulate::use_condaenv(envname, required = FALSE)

    pdf_module <- reticulate::import_from_path(
      "multimodal_extraction",
      path = system.file("python", package = "TextAnalysisR")
    )

    result <- pdf_module$extract_pdf_with_images(
      file_path = file_path,
      vision_provider = vision_provider,
      vision_model = vision_model,
      api_key = api_key,
      describe_images = describe_images
    )

    return(result)

  }, error = function(e) {
    return(list(
      success = FALSE,
      message = paste("Error:", e$message)
    ))
  })
}


#' Smart PDF Extraction with Auto-Detection
#'
#' @description
#' Automatically detects document type and chooses best extraction method:
#' - Academic papers → Nougat (equations)
#' - Documents with visuals → Multimodal extraction
#' - General documents → Marker
#'
#' @param file_path Character string path to PDF file
#' @param doc_type Character: "auto" (default), "academic", or "general"
#' @param vision_provider Character: "ollama" (default) or "openai"
#' @param vision_model Character: Model name for vision analysis
#' @param api_key Character: API key for cloud providers
#' @param envname Character: Python environment name
#'
#' @return List with extracted content ready for text analysis
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Auto-detect and extract
#' result <- extract_pdf_smart("document.pdf")
#'
#' # Feed to text analysis
#' corpus <- prep_texts(result$combined_text)
#' topics <- fit_semantic_model(corpus, k = 10)
#'
#' # Force academic extraction (with equations)
#' result <- extract_pdf_smart("paper.pdf", doc_type = "academic")
#' }
extract_pdf_smart <- function(
  file_path,
  doc_type = "auto",
  vision_provider = "ollama",
  vision_model = NULL,
  api_key = NULL,
  envname = "langgraph-env"
) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required.")
  }

  if (!file.exists(file_path)) {
    return(list(
      success = FALSE,
      message = "File not found"
    ))
  }

  if (is.null(vision_model)) {
    vision_model <- if (vision_provider == "ollama") "llava" else "gpt-4-vision-preview"
  }

  tryCatch({
    reticulate::use_condaenv(envname, required = FALSE)

    pdf_module <- reticulate::import_from_path(
      "multimodal_extraction",
      path = system.file("python", package = "TextAnalysisR")
    )

    result <- pdf_module$extract_pdf_smart(
      file_path = file_path,
      doc_type = doc_type,
      vision_provider = vision_provider,
      vision_model = vision_model,
      api_key = api_key
    )

    return(result)

  }, error = function(e) {
    return(list(
      success = FALSE,
      message = paste("Error:", e$message)
    ))
  })
}


#' Check Vision Model Availability
#'
#' @description
#' Check if required vision models are available for multimodal processing.
#'
#' @param provider Character: "ollama" or "openai"
#' @param api_key Character: API key (for OpenAI)
#'
#' @return List with availability status and recommendations
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Check Ollama vision models
#' status <- check_vision_models("ollama")
#' print(status$message)
#'
#' # Check OpenAI access
#' status <- check_vision_models("openai", api_key = Sys.getenv("OPENAI_API_KEY"))
#' }
check_vision_models <- function(provider = "ollama", api_key = NULL) {
  if (provider == "ollama") {
    if (!check_ollama(verbose = FALSE)) {
      return(list(
        available = FALSE,
        models = character(0),
        message = "Ollama not running. Install from https://ollama.ai"
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
    if (is.null(api_key)) {
      return(list(
        available = FALSE,
        message = "OpenAI API key required"
      ))
    }

    # Simple API key validation
    valid <- nchar(api_key) > 20 && grepl("^sk-", api_key)

    return(list(
      available = valid,
      message = if (valid) "API key format valid" else "Invalid API key format"
    ))
  }

  return(list(
    available = FALSE,
    message = paste("Unknown provider:", provider)
  ))
}
