#' Check if Ollama is Available
#'
#' @description Checks if Ollama is installed and running on the local machine.
#'
#' @param verbose Logical, if TRUE, prints status messages.
#'
#' @return Logical indicating whether Ollama is available.
#'
#' @family ai
#' @export
#'
#' @examples
#' \dontrun{
#' if (check_ollama()) {
#'   message("Ollama is ready!")
#' }
#' }
check_ollama <- function(verbose = FALSE) {
  tryCatch({
    response <- httr::GET(
      "http://localhost:11434/api/tags",
      httr::timeout(2)
    )

    is_available <- httr::status_code(response) == 200

    if (verbose) {
      if (is_available) {
        message("Ollama is available and running")
      } else {
        message("Ollama server responded but returned error status")
      }
    }

    return(is_available)

  }, error = function(e) {
    if (verbose) {
      message("Ollama is not available: ", e$message)
      message("To use Ollama, please install it from https://ollama.ai")
    }
    return(FALSE)
  })
}

#' List Available Ollama Models
#'
#' @description Lists all models currently installed in Ollama.
#'
#' @param verbose Logical, if TRUE, prints status messages.
#'
#' @return Character vector of model names, or NULL if Ollama is unavailable.
#'
#' @family ai
#' @export
#'
#' @examples
#' \dontrun{
#' models <- list_ollama_models()
#' print(models)
#' }
list_ollama_models <- function(verbose = FALSE) {
  if (!check_ollama(verbose = FALSE)) {
    if (verbose) {
      message("Ollama is not available")
    }
    return(NULL)
  }

  tryCatch({
    response <- httr::GET(
      "http://localhost:11434/api/tags",
      httr::timeout(5)
    )

    if (httr::status_code(response) == 200) {
      content <- jsonlite::fromJSON(httr::content(response, "text", encoding = "UTF-8"))

      if (!is.null(content$models) && length(content$models) > 0) {
        model_names <- content$models$name
        if (verbose) {
          message("Found ", length(model_names), " Ollama models:")
          for (model in model_names) {
            message("  - ", model)
          }
        }
        return(model_names)
      } else {
        if (verbose) {
          message("No Ollama models found. Please pull a model:")
          message("  ollama pull phi3:mini")
        }
        return(character(0))
      }
    }

    return(NULL)

  }, error = function(e) {
    if (verbose) {
      message("Error listing Ollama models: ", e$message)
    }
    return(NULL)
  })
}

#' Call Ollama for Text Generation
#'
#' @description Sends a prompt to Ollama and returns the generated text.
#'
#' @param prompt Character string containing the prompt.
#' @param model Character string specifying the Ollama model (default: "phi3:mini").
#' @param temperature Numeric value controlling randomness (default: 0.3).
#' @param max_tokens Maximum number of tokens to generate (default: 512).
#' @param timeout Timeout in seconds for the request (default: 60).
#' @param verbose Logical, if TRUE, prints progress messages.
#'
#' @return Character string with the generated text, or NULL if failed.
#'
#' @family ai
#' @export
#'
#' @examples
#' \dontrun{
#' response <- call_ollama(
#'   prompt = "Summarize these keywords: machine learning, neural networks, AI",
#'   model = "phi3:mini"
#' )
#' print(response)
#' }
call_ollama <- function(prompt,
                       model = "phi3:mini",
                       temperature = 0.3,
                       max_tokens = 512,
                       timeout = 60,
                       verbose = FALSE) {

  if (!check_ollama(verbose = verbose)) {
    stop("Ollama is not available. Please ensure Ollama is installed and running.")
  }

  if (verbose) {
    message("Calling Ollama with model: ", model)
  }

  tryCatch({
    body <- list(
      model = model,
      prompt = prompt,
      stream = FALSE,
      options = list(
        temperature = temperature,
        num_predict = max_tokens
      )
    )

    response <- httr::POST(
      "http://localhost:11434/api/generate",
      body = jsonlite::toJSON(body, auto_unbox = TRUE),
      httr::content_type_json(),
      httr::timeout(timeout)
    )

    if (httr::status_code(response) == 200) {
      content <- jsonlite::fromJSON(httr::content(response, "text", encoding = "UTF-8"))

      if (!is.null(content$response)) {
        if (verbose) {
          message("Ollama response received successfully")
        }
        return(trimws(content$response))
      } else {
        warning("Ollama response was empty")
        return(NULL)
      }
    } else {
      warning("Ollama API returned status code: ", httr::status_code(response))
      return(NULL)
    }

  }, error = function(e) {
    warning("Error calling Ollama: ", e$message)
    return(NULL)
  })
}

#' Get Recommended Ollama Model
#'
#' @description Returns a recommended Ollama model based on what's available.
#'
#' @param preferred_models Character vector of preferred models in priority order.
#' @param verbose Logical, if TRUE, prints status messages.
#'
#' @return Character string of recommended model, or NULL if none available.
#'
#' @family ai
#' @export
#'
#' @examples
#' \dontrun{
#' model <- get_recommended_ollama_model()
#' print(model)
#' }
get_recommended_ollama_model <- function(preferred_models = c("phi3:mini", "llama3.1:8b", "mistral:7b", "tinyllama"),
                                        verbose = FALSE) {

  available_models <- list_ollama_models(verbose = FALSE)

  if (is.null(available_models) || length(available_models) == 0) {
    if (verbose) {
      message("No Ollama models available. Recommended models:")
      message("  1. phi3:mini (2.3GB, best balance)")
      message("  2. llama3.1:8b (4.7GB, strong reasoning)")
      message("  3. mistral:7b (4.1GB, high quality)")
      message("\nTo install: ollama pull phi3:mini")
    }
    return(NULL)
  }

  for (preferred in preferred_models) {
    if (preferred %in% available_models) {
      if (verbose) {
        message("Using model: ", preferred)
      }
      return(preferred)
    }
  }

  if (verbose) {
    message("Using first available model: ", available_models[1])
  }
  return(available_models[1])
}
