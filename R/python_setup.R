#' Setup Python Environment
#'
#' @description
#' Intelligently sets up Python virtual environment with required packages.
#' Detects existing Python installations and guides users if Python is missing.
#'
#' @param envname Character string name for the virtual environment
#'   (default: "textanalysisr-env")
#' @param force Logical, whether to recreate environment if it exists
#'   (default: FALSE)
#'
#' @return Invisible TRUE if successful, stops with error message if failed
#'
#' @details
#' This function:
#' - Automatically detects if Python is already installed
#' - Offers to install Miniconda if no Python found
#' - Creates an isolated virtual environment (does NOT modify system Python)
#' - Installs ONLY 6 core packages (minimal installation):
#'   * langchain-core (core LangChain functionality)
#'   * langchain-ollama (Ollama integration)
#'   * langgraph (workflow graphs)
#'   * langgraph-checkpoint (workflow state management)
#'   * ollama (Ollama client)
#'   * pdfplumber (PDF table extraction)
#' - Dependencies installed automatically by pip
#' - Avoids heavy packages (no marker-pdf, nougat-ocr, torch)
#'
#' The virtual environment approach means:
#' - No conflicts with other Python projects
#' - Easy to remove (just delete the environment)
#' - System Python remains untouched
#' - Much smaller download (~100MB vs 5GB+)
#'
#' After setup, restart R session to activate enhanced features.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # First time setup (auto-detects Python)
#' setup_python_env()
#'
#' # Recreate environment
#' setup_python_env(force = TRUE)
#' }
setup_python_env <- function(envname = "textanalysisr-env", force = FALSE) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required. Install it with: install.packages('reticulate')")
  }

  message("\nPython Environment Setup")


  # Check if Python is available
  python_available <- tryCatch({
    py_config <- reticulate::py_discover_config()
    !is.null(py_config$python)
  }, error = function(e) FALSE)

  if (!python_available) {
    message("No Python found. Install Miniconda? (y/n): ")
    response <- readline(prompt = "")

    if (tolower(trimws(response)) == "y") {
      message("Installing Miniconda...")
      reticulate::install_miniconda()
      message("Done.")
    } else {
      stop("Python required. Install from python.org and retry.")
    }
  } else {
    py_info <- reticulate::py_discover_config()
    message("Python: ", py_info$python, " (v", py_info$version, ")")
  }

  tryCatch({
    env_exists <- envname %in% reticulate::virtualenv_list()

    if (env_exists && !force) {
      message("Environment '", envname, "' exists. Use force=TRUE to recreate.")
      reticulate::use_virtualenv(envname, required = TRUE)
      return(invisible(TRUE))
    }

    if (env_exists && force) {
      message("Removing existing environment...")
      reticulate::virtualenv_remove(envname, confirm = FALSE)
    }

    message("Creating environment '", envname, "'...")
    reticulate::virtualenv_create(envname, python = NULL)
    reticulate::use_virtualenv(envname, required = TRUE)

    requirements_file <- system.file("python", "requirements.txt", package = "TextAnalysisR")

    message("Installing packages...")
    if (file.exists(requirements_file)) {
      req_packages <- readLines(requirements_file)
      req_packages <- req_packages[!grepl("^#|^\\s*$", req_packages)]
      reticulate::virtualenv_install(envname = envname, packages = req_packages, ignore_installed = FALSE)
    } else {
      packages <- c(
        "langchain-core>=0.3.0",
        "langchain-ollama>=0.2.0",
        "langgraph>=0.2.0",
        "langgraph-checkpoint>=2.0.0",
        "ollama>=0.3.0",
        "pdfplumber>=0.10.0"
      )
      reticulate::virtualenv_install(envname = envname, packages = packages, ignore_installed = FALSE)
    }

    message("Testing imports...")
    test_result <- tryCatch({
      reticulate::py_run_string("import langgraph")
      reticulate::py_run_string("from langchain_core import prompts")
      reticulate::py_run_string("from langchain_ollama import ChatOllama")
      reticulate::py_run_string("import ollama")
      reticulate::py_run_string("import pdfplumber")
      TRUE
    }, error = function(e) {
      message("Import failed: ", e$message)
      FALSE
    })

    if (test_result) {
      message("Setup complete. Restart R session to activate.")
      return(invisible(TRUE))
    } else {
      stop("Package imports failed. Check Python logs.")
    }

  }, error = function(e) {
    stop("Failed to set up Python environment: ", e$message)
  })
}


#' Check Python Environment Status
#'
#' @description
#' Checks if Python environment is available and properly configured.
#'
#' @param envname Character string name of the virtual environment
#'   (default: "textanalysisr-env")
#'
#' @return List with status information:
#'   - available: Logical, TRUE if environment exists
#'   - active: Logical, TRUE if environment is currently active
#'   - packages: List of installed package versions
#'
#' @export
#'
#' @examples
#' \dontrun{
#' status <- check_python_env()
#' print(status)
#' }
check_python_env <- function(envname = "textanalysisr-env") {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required.")
  }

  env_list <- reticulate::virtualenv_list()
  available <- envname %in% env_list

  if (!available) {
    return(list(
      available = FALSE,
      active = FALSE,
      packages = NULL,
      message = paste("Environment", envname, "not found. Run setup_python_env() to create it.")
    ))
  }

  tryCatch({
    reticulate::use_virtualenv(envname, required = TRUE)

    packages <- tryCatch({
      langgraph_version <- reticulate::py_run_string("import langgraph; print(langgraph.__version__)")
      langchain_version <- reticulate::py_run_string("import langchain; print(langchain.__version__)")
      ollama_version <- reticulate::py_run_string("import ollama; print(ollama.__version__)")

      list(
        langgraph = langgraph_version,
        langchain = langchain_version,
        ollama = ollama_version
      )
    }, error = function(e) NULL)

    ollama_check <- tryCatch({
      reticulate::py_run_string("import ollama; ollama.list()")
      TRUE
    }, error = function(e) FALSE)

    return(list(
      available = TRUE,
      active = TRUE,
      packages = packages,
      ollama_available = ollama_check,
      message = "Python environment is ready"
    ))

  }, error = function(e) {
    return(list(
      available = TRUE,
      active = FALSE,
      packages = NULL,
      ollama_available = FALSE,
      message = paste("Failed to activate environment:", e$message)
    ))
  })
}


#' Initialize LangGraph for Current Session
#'
#' @description
#' Initializes LangGraph/LangChain/Ollama modules for current R session.
#' Use only for LangGraph workflows. PDF/embeddings load automatically.
#'
#' @param envname Character string name of the virtual environment
#'   (default: "textanalysisr-env")
#'
#' @return Invisible list with LangGraph/LangChain/Ollama modules
#'
#' @export
#'
#' @examples
#' \dontrun{
#' lg <- init_langgraph()
#' }
init_langgraph <- function(envname = "textanalysisr-env") {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required.")
  }

  status <- check_python_env(envname)

  if (!status$available) {
    stop("Python environment not found. Run setup_python_env() first.")
  }

  message("Initializing LangGraph modules...")
  reticulate::use_virtualenv(envname, required = TRUE)

  modules <- tryCatch({
    list(
      langgraph = reticulate::import("langgraph"),
      langchain = reticulate::import("langchain"),
      langchain_ollama = reticulate::import("langchain_ollama"),
      ollama = reticulate::import("ollama")
    )
  }, error = function(e) {
    stop("Failed to import LangGraph modules: ", e$message)
  })

  message("LangGraph initialized successfully")

  if (!status$ollama_available) {
    warning("Ollama connection not available. Make sure Ollama is running.")
  }

  return(invisible(modules))
}
