#' Generate Topic Labels with LLM Assistance
#'
#' @description
#' Uses LangGraph workflow to generate multiple label candidates for topics
#' using local LLM (Ollama). Provides human-in-the-loop review of suggestions.
#'
#' @param topic_terms List of character vectors, where each vector contains
#'   the top terms for a topic (from STM or other topic model)
#' @param num_topics Integer, number of topics
#' @param ollama_model Character string, name of Ollama model to use
#'   (default: "llama3")
#' @param ollama_base_url Character string, base URL for Ollama API
#'   (default: "http://localhost:11434")
#' @param envname Character string, name of Python virtual environment
#'   (default: "langgraph-env")
#'
#' @return List with:
#'   - success: Logical, TRUE if workflow completed successfully
#'   - label_candidates: List of label candidate objects for each topic
#'   - validation_metrics: Validation metrics (if available)
#'   - error: Error message (if failed)
#'
#' @details
#' This function:
#' 1. Initializes LangGraph Python environment
#' 2. Calls Python workflow to generate label candidates
#' 3. Returns structured results for display in Shiny UI
#' 4. Allows human review and selection of labels
#'
#' The workflow uses a StateGraph with nodes for:
#' - Label generation (LLM)
#' - Validation (LLM)
#' - Conditional revision based on quality metrics
#'
#' @export
#'
#' @examples
#' \dontrun{
#' topic_terms <- list(
#'   c("education", "student", "learning", "teacher", "school"),
#'   c("health", "medical", "patient", "doctor", "treatment"),
#'   c("environment", "climate", "carbon", "emissions", "energy")
#' )
#'
#' result <- generate_topic_labels_langgraph(
#'   topic_terms = topic_terms,
#'   num_topics = 3,
#'   ollama_model = "llama3"
#' )
#'
#' if (result$success) {
#'   print(result$label_candidates)
#' }
#' }
generate_topic_labels_langgraph <- function(
  topic_terms,
  num_topics,
  ollama_model = "llama3",
  ollama_base_url = "http://localhost:11434",
  envname = "textanalysisr-env"
) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required.")
  }

  status <- check_python_env(envname)

  if (!status$available) {
    stop("LangGraph environment not found. Run setup_python_env() first.")
  }

  tryCatch({
    reticulate::use_virtualenv(envname, required = TRUE)

    workflow_module <- reticulate::import_from_path(
      "topic_modeling_workflow",
      path = system.file("python", package = "TextAnalysisR")
    )

    result <- workflow_module$run_topic_label_generation(
      topic_terms = topic_terms,
      num_topics = as.integer(num_topics),
      ollama_model = ollama_model,
      ollama_base_url = ollama_base_url
    )

    if (is.null(result$success) || !result$success) {
      return(list(
        success = FALSE,
        error = result$error %||% "Unknown error in workflow",
        label_candidates = NULL
      ))
    }

    return(list(
      success = TRUE,
      label_candidates = result$label_candidates,
      validation_metrics = result$validation_metrics,
      needs_revision = result$needs_revision %||% FALSE
    ))

  }, error = function(e) {
    return(list(
      success = FALSE,
      error = paste("LangGraph workflow error:", e$message),
      label_candidates = NULL
    ))
  })
}


#' Validate User-Selected Topic Labels
#'
#' @description
#' Uses LangGraph workflow to validate user-selected topic labels using LLM.
#'
#' @param user_labels Character vector of user-selected labels for each topic
#' @param topic_terms List of character vectors with top terms for each topic
#' @param ollama_model Character string, Ollama model name (default: "llama3")
#' @param ollama_base_url Character string, Ollama API URL
#'   (default: "http://localhost:11434")
#' @param envname Character string, Python virtual environment name
#'   (default: "langgraph-env")
#'
#' @return List with:
#'   - success: Logical, TRUE if validation completed
#'   - validation_metrics: List with coherence and distinctiveness scores
#'   - error: Error message (if failed)
#'
#' @details
#' Validation metrics include:
#' - coherence_scores: How well labels match term distributions (0-10 scale)
#' - distinctiveness_scores: How unique/specific labels are (0-10 scale)
#' - overall_quality: Average of coherence and distinctiveness
#'
#' @export
#'
#' @examples
#' \dontrun{
#' user_labels <- c("Education Policy", "Healthcare Services", "Climate Action")
#' topic_terms <- list(
#'   c("education", "student", "learning"),
#'   c("health", "medical", "patient"),
#'   c("environment", "climate", "carbon")
#' )
#'
#' validation <- validate_topic_labels_langgraph(
#'   user_labels = user_labels,
#'   topic_terms = topic_terms
#' )
#'
#' print(validation$validation_metrics$overall_quality)
#' }
validate_topic_labels_langgraph <- function(
  user_labels,
  topic_terms,
  ollama_model = "llama3",
  ollama_base_url = "http://localhost:11434",
  envname = "textanalysisr-env"
) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required.")
  }

  status <- check_python_env(envname)

  if (!status$available) {
    stop("LangGraph environment not found. Run setup_python_env() first.")
  }

  tryCatch({
    reticulate::use_virtualenv(envname, required = TRUE)

    workflow_module <- reticulate::import_from_path(
      "topic_modeling_workflow",
      path = system.file("python", package = "TextAnalysisR")
    )

    result <- workflow_module$validate_user_labels(
      user_labels = user_labels,
      topic_terms = topic_terms,
      ollama_model = ollama_model,
      ollama_base_url = ollama_base_url
    )

    if (is.null(result$success) || !result$success) {
      return(list(
        success = FALSE,
        error = result$error %||% "Unknown error in validation",
        validation_metrics = NULL
      ))
    }

    return(list(
      success = TRUE,
      validation_metrics = result$validation_metrics
    ))

  }, error = function(e) {
    return(list(
      success = FALSE,
      error = paste("Validation error:", e$message),
      validation_metrics = NULL
    ))
  })
}


#' Format Label Candidates for Display
#'
#' @description
#' Helper function to format LangGraph label candidates for display in Shiny UI.
#'
#' @param label_candidates List of label candidate objects from
#'   generate_topic_labels_langgraph()
#'
#' @return Data frame with columns:
#'   - topic_index: Integer, topic number
#'   - top_terms: Character, comma-separated top terms
#'   - label: Character, suggested label
#'   - reasoning: Character, LLM explanation
#'   - candidate_number: Integer, candidate rank (1-3)
#'
#' @export
#'
#' @examples
#' \dontrun{
#' result <- generate_topic_labels_langgraph(...)
#' df <- format_label_candidates(result$label_candidates)
#' print(df)
#' }
format_label_candidates <- function(label_candidates) {
  if (is.null(label_candidates) || length(label_candidates) == 0) {
    return(data.frame(
      topic_index = integer(0),
      top_terms = character(0),
      label = character(0),
      reasoning = character(0),
      candidate_number = integer(0),
      stringsAsFactors = FALSE
    ))
  }

  rows <- list()

  for (topic_data in label_candidates) {
    topic_idx <- topic_data$topic_index + 1
    top_terms_str <- paste(topic_data$top_terms[1:min(5, length(topic_data$top_terms))],
                           collapse = ", ")

    candidates <- topic_data$candidates

    for (i in seq_along(candidates)) {
      candidate <- candidates[[i]]

      rows[[length(rows) + 1]] <- data.frame(
        topic_index = topic_idx,
        top_terms = top_terms_str,
        label = candidate$label %||% "",
        reasoning = candidate$reasoning %||% "",
        candidate_number = i,
        stringsAsFactors = FALSE
      )
    }
  }

  if (length(rows) == 0) {
    return(data.frame(
      topic_index = integer(0),
      top_terms = character(0),
      label = character(0),
      reasoning = character(0),
      candidate_number = integer(0),
      stringsAsFactors = FALSE
    ))
  }

  do.call(rbind, rows)
}


#' Create Label Selection UI Data
#'
#' @description
#' Creates a structured list for rendering label selection UI in Shiny.
#'
#' @param label_candidates List from generate_topic_labels_langgraph()
#'
#' @return List of topic objects, each with:
#'   - topic_number: Integer
#'   - top_terms: Character vector
#'   - candidates: List of candidate objects
#'
#' @export
#'
#' @examples
#' \dontrun{
#' result <- generate_topic_labels_langgraph(...)
#' ui_data <- create_label_selection_data(result$label_candidates)
#' }
create_label_selection_data <- function(label_candidates) {
  if (is.null(label_candidates) || length(label_candidates) == 0) {
    return(list())
  }

  lapply(label_candidates, function(topic_data) {
    list(
      topic_number = topic_data$topic_index + 1,
      top_terms = topic_data$top_terms,
      candidates = topic_data$candidates
    )
  })
}


#' RAG-Enhanced Semantic Search
#'
#' @description
#' Uses LangGraph multi-agent workflow for Retrieval Augmented Generation.
#' Provides question-answering over document corpus with source attribution.
#'
#' @param query Character string, user question
#' @param documents Character vector, corpus to search
#' @param ollama_model Character string, LLM model (default: "llama3")
#' @param ollama_base_url Character string, Ollama API endpoint
#' @param embedding_model Character string, embedding model (default: "nomic-embed-text")
#' @param top_k Integer, number of documents to retrieve (default: 5)
#' @param envname Character string, Python environment name
#'
#' @return List with:
#'   - success: Logical
#'   - answer: Generated answer
#'   - confidence: Confidence score (0-1)
#'   - sources: Vector of source document IDs
#'   - retrieved_docs: Retrieved document chunks
#'   - scores: Similarity scores
#'
#' @details
#' Multi-agent workflow:
#' 1. Retrieval Agent: Find relevant documents via embeddings
#' 2. Generation Agent: Create answer from context
#' 3. Validation Agent: Assess answer quality
#' 4. Conditional retry if confidence < 0.4
#'
#' Requires Ollama with embedding model:
#' ```
#' ollama pull llama3
#' ollama pull nomic-embed-text
#' ```
#'
#' @export
#'
#' @examples
#' \dontrun{
#' documents <- c(
#'   "Assistive technology helps students with disabilities access curriculum.",
#'   "Universal Design for Learning provides multiple means of engagement.",
#'   "Response to Intervention uses tiered support systems."
#' )
#'
#' result <- run_rag_search(
#'   query = "How does assistive technology support learning?",
#'   documents = documents
#' )
#'
#' if (result$success) {
#'   cat("Answer:", result$answer, "\n")
#'   cat("Confidence:", result$confidence, "\n")
#'   cat("Sources:", paste(result$sources, collapse = ", "), "\n")
#' }
#' }
run_rag_search <- function(
  query,
  documents,
  ollama_model = "llama3",
  ollama_base_url = "http://localhost:11434",
  embedding_model = "nomic-embed-text",
  top_k = 5,
  envname = "textanalysisr-env"
) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required.")
  }

  status <- check_python_env(envname)
  if (!status$available) {
    stop("LangGraph environment not found. Run setup_python_env() first.")
  }

  if (length(documents) < 1) {
    return(list(
      success = FALSE,
      error = "No documents provided",
      answer = "",
      confidence = 0.0,
      sources = c()
    ))
  }

  tryCatch({
    reticulate::use_virtualenv(envname, required = TRUE)

    rag_module <- reticulate::import_from_path(
      "rag_search_workflow",
      path = system.file("python", package = "TextAnalysisR")
    )

    result <- rag_module$run_rag_search(
      query = query,
      documents = documents,
      ollama_model = ollama_model,
      ollama_base_url = ollama_base_url,
      embedding_model = embedding_model,
      top_k = as.integer(top_k)
    )

    return(result)
  }, error = function(e) {
    return(list(
      success = FALSE,
      error = paste("Error in RAG search:", e$message),
      answer = "",
      confidence = 0.0,
      sources = c()
    ))
  })
}
