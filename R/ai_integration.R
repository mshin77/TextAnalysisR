#' AI-Assisted Content Generation from Topics
#'
#' Functions for generating various types of content from topic model terms
#' using Large Language Models (LLMs). Supports both OpenAI and Ollama backends.
#'
#' @name ai_content_generation
NULL

#' Get Default System Prompt for Content Type
#'
#' Returns the default system prompt for a given content type.
#'
#' @param content_type Type of content to generate.
#'
#' @return Character string with the system prompt.
#'
#' @family ai
#' @export
get_content_type_prompt <- function(content_type) {
  prompts <- list(
    survey_item = "
You are a survey design expert specializing in creating Likert-scale items for research.
Your task is to generate clear, concise survey statements that can be rated on a 5-point scale (1=Strongly Disagree to 5=Strongly Agree).

Guidelines:
1. Create statements focused on a single concept
2. Use active voice and present tense
3. Avoid double-barreled questions
4. Use simple, direct language
5. Ensure items work well on an agree-disagree scale
6. Frame items to capture the essence of the provided keywords

Return ONLY the survey item text, without numbering, quotes, or explanations.",

    research_question = "
You are a research methodology expert specializing in formulating research questions.
Your task is to generate clear, focused research questions based on topic keywords.

Guidelines:
1. Create questions that are specific and answerable
2. Use appropriate question words (How, What, Why, To what extent)
3. Ensure questions are neither too broad nor too narrow
4. Frame questions to guide empirical investigation
5. Avoid yes/no questions - aim for open-ended inquiry

Return ONLY the research question, without numbering, quotes, or explanations.",

    theme_description = "
You are a qualitative research expert specializing in thematic analysis.
Your task is to generate descriptive summaries of themes based on topic keywords.

Guidelines:
1. Write in third person, academic style
2. Describe what the theme encompasses
3. Use language like 'This theme captures...', 'Participants discussed...'
4. Be concise but comprehensive
5. Avoid interpretation - focus on description

Return ONLY the theme description, without numbering, quotes, or explanations.",

    policy_recommendation = "
You are a policy analysis expert specializing in evidence-based recommendations.
Your task is to generate actionable policy recommendations based on topic keywords.

Guidelines:
1. Begin with action verbs (Implement, Establish, Develop, Ensure)
2. Be specific and actionable
3. Consider feasibility and impact
4. Use clear, professional language
5. Focus on one recommendation per topic

Return ONLY the policy recommendation, without numbering, quotes, or explanations.",

    interview_question = "
You are a qualitative research expert specializing in interview methodology.
Your task is to generate open-ended interview questions based on topic keywords.

Guidelines:
1. Create questions that encourage detailed responses
2. Use open-ended phrasing (Can you describe..., Tell me about..., How do you...)
3. Avoid leading or biased questions
4. Make questions conversational yet focused
5. Ensure questions are appropriate for semi-structured interviews

Return ONLY the interview question, without numbering, quotes, or explanations.",

    custom = "
You are an expert content generator.
Your task is to generate content based on the provided keywords.

Return ONLY the requested content, without numbering, quotes, or explanations."
  )

  prompts[[content_type]] %||% prompts[["custom"]]
}


#' Get Default User Prompt Template for Content Type
#'
#' Returns the default user prompt template for a given content type.
#'
#' @param content_type Type of content to generate.
#'
#' @return Character string with the user prompt template containing \code{\{terms\}} placeholder.
#'
#' @family ai
#' @export
get_content_type_user_template <- function(content_type) {
  templates <- list(
    survey_item = "Generate a single survey item based on these keywords (ordered by importance): {terms}

The survey item should:
- Capture the main concept from these keywords
- Be rateable on a 5-point Likert scale
- Be clear and concise

Survey item:",

    research_question = "Generate a research question based on these keywords (ordered by importance): {terms}

The research question should:
- Be specific and empirically answerable
- Capture the key concepts from these keywords
- Guide meaningful investigation

Research question:",

    theme_description = "Generate a theme description based on these keywords (ordered by importance): {terms}

The theme description should:
- Summarize what this theme encompasses
- Be written in academic style
- Capture the essence of the keywords

Theme description:",

    policy_recommendation = "Generate a policy recommendation based on these keywords (ordered by importance): {terms}

The recommendation should:
- Be specific and actionable
- Address the key concepts from these keywords
- Be feasible to implement

Policy recommendation:",

    interview_question = "Generate an interview question based on these keywords (ordered by importance): {terms}

The interview question should:
- Be open-ended and encourage detailed responses
- Explore the concepts represented by these keywords
- Be appropriate for a semi-structured interview

Interview question:",

    custom = "Generate content based on these keywords (ordered by importance): {terms}

Content:"
  )

  templates[[content_type]] %||% templates[["custom"]]
}


#' Generate Content from Topic Terms
#'
#' Uses Large Language Models (LLMs) to generate various types of content
#' based on topic model terms. Supports multiple content types with optimized
#' default prompts, or fully custom prompts.
#'
#' @param topic_terms_df A data frame with topic terms, containing columns for
#'   topic identifier, term, and optionally term weight (beta).
#' @param content_type Type of content to generate. One of:
#'   \describe{
#'     \item{"survey_item"}{Likert-scale survey items for scale development}
#'     \item{"research_question"}{Research questions for literature review}
#'     \item{"theme_description"}{Theme descriptions for qualitative analysis}
#'     \item{"policy_recommendation"}{Policy recommendations for policy analysis}
#'     \item{"interview_question"}{Interview questions for qualitative research}
#'     \item{"custom"}{Custom content using user-provided prompts}
#'   }
#' @param topic_var Name of the column containing topic identifiers (default: "topic").
#' @param term_var Name of the column containing terms (default: "term").
#' @param weight_var Name of the column containing term weights (default: "beta").
#' @param provider LLM provider: "openai" or "ollama" (default: "openai").
#' @param model Model name. For OpenAI: "gpt-3.5-turbo", "gpt-4", etc.
#'   For Ollama: "llama3", "mistral", etc.
#' @param temperature Sampling temperature (0-2). Lower = more deterministic (default: 0).
#' @param system_prompt Custom system prompt. If NULL, uses default for content_type.
#' @param user_prompt_template Custom user prompt template with \{terms\} placeholder.
#'   If NULL, uses default for content_type.
#' @param max_tokens Maximum tokens for response (default: 150).
#' @param api_key OpenAI API key. If NULL, reads from OPENAI_API_KEY environment variable.
#' @param output_var Name of the output column (default: based on content_type).
#' @param verbose Logical, if TRUE, prints progress messages.
#'
#' @return A data frame with generated content joined to original topic terms.
#'
#' @details
#' The function generates one piece of content per unique topic. Each content type
#' has optimized default prompts, but these can be overridden with custom prompts.
#'
#' For OpenAI, requires an API key set via the `api_key` parameter or
#' OPENAI_API_KEY environment variable (can be loaded from .env file).
#'
#' For Ollama, requires a local Ollama installation with the specified model.
#'
#' @family ai
#' @export
#'
#' @examples
#' \dontrun{
#' # Generate survey items
#' survey_items <- generate_topic_content(
#'   topic_terms_df = top_terms,
#'   content_type = "survey_item",
#'   provider = "openai",
#'   model = "gpt-3.5-turbo"
#' )
#'
#' # Generate research questions
#' research_qs <- generate_topic_content(
#'   topic_terms_df = top_terms,
#'   content_type = "research_question",
#'   provider = "ollama",
#'   model = "llama3"
#' )
#'
#' # Generate with custom prompt
#' custom_content <- generate_topic_content(
#'   topic_terms_df = top_terms,
#'   content_type = "custom",
#'   system_prompt = "You are an expert in educational policy...",
#'   user_prompt_template = "Based on {terms}, generate a learning objective:"
#' )
#' }
generate_topic_content <- function(topic_terms_df,
                                    content_type = c("survey_item", "research_question",
                                                     "theme_description", "policy_recommendation",
                                                     "interview_question", "custom"),
                                    topic_var = "topic",
                                    term_var = "term",
                                    weight_var = "beta",
                                    provider = c("openai", "ollama"),
                                    model = "gpt-3.5-turbo",
                                    temperature = 0,
                                    system_prompt = NULL,
                                    user_prompt_template = NULL,
                                    max_tokens = 150,
                                    api_key = NULL,
                                    output_var = NULL,
                                    verbose = TRUE) {

  content_type <- match.arg(content_type)
  provider <- match.arg(provider)

  if (!topic_var %in% names(topic_terms_df)) {
    stop("topic_var '", topic_var, "' not found in topic_terms_df")
  }
  if (!term_var %in% names(topic_terms_df)) {
    stop("term_var '", term_var, "' not found in topic_terms_df")
  }

  # Set default output variable name based on content type
  if (is.null(output_var)) {
    output_var <- switch(content_type,
      "survey_item" = "survey_item",
      "research_question" = "research_question",
      "theme_description" = "theme_description",
      "policy_recommendation" = "policy_recommendation",
      "interview_question" = "interview_question",
      "custom" = "generated_content"
    )
  }

  # Get default prompts if not provided
  if (is.null(system_prompt)) {
    system_prompt <- get_content_type_prompt(content_type)
  }
  if (is.null(user_prompt_template)) {
    user_prompt_template <- get_content_type_user_template(content_type)
  }

  # Prepare topic data
 has_weights <- weight_var %in% names(topic_terms_df)

  if (has_weights) {
    top_terms <- topic_terms_df %>%
      dplyr::group_by(.data[[topic_var]]) %>%
      dplyr::arrange(dplyr::desc(.data[[weight_var]])) %>%
      dplyr::ungroup()
  } else {
    top_terms <- topic_terms_df %>%
      dplyr::group_by(.data[[topic_var]]) %>%
      dplyr::ungroup()
  }

  unique_topics <- top_terms %>%
    dplyr::distinct(.data[[topic_var]]) %>%
    dplyr::arrange(.data[[topic_var]])

  unique_topics[[output_var]] <- NA_character_

  # Setup for provider
  if (provider == "openai") {
    if (is.null(api_key)) {
      if (file.exists(".env")) {
        if (requireNamespace("dotenv", quietly = TRUE)) {
          dotenv::load_dot_env()
        }
      }
      api_key <- Sys.getenv("OPENAI_API_KEY")
      if (api_key == "") {
        stop("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
      }
    }
  } else if (provider == "ollama") {
    if (!check_ollama(verbose = FALSE)) {
      stop("Ollama is not available. Please install and start Ollama from https://ollama.ai")
    }
  }

  # Progress bar
  if (verbose && requireNamespace("progress", quietly = TRUE)) {
    pb <- progress::progress_bar$new(
      format = paste0(" Generating ", content_type, "s [:bar] :percent (:current/:total) ETA: :eta"),
      total = nrow(unique_topics),
      clear = FALSE, width = 60
    )
  }

  for (i in seq_len(nrow(unique_topics))) {
    if (verbose && exists("pb")) {
      pb$tick()
    }

    current_topic <- unique_topics[[topic_var]][i]

    # Get terms for this topic
    topic_data <- top_terms %>%
      dplyr::filter(.data[[topic_var]] == current_topic)

    terms_text <- paste(topic_data[[term_var]], collapse = ", ")

    # Create user prompt
    user_prompt <- gsub("\\{terms\\}", terms_text, user_prompt_template)

    # Call LLM
    generated_content <- tryCatch({
      if (provider == "openai") {
        call_openai_chat(
          system_prompt = system_prompt,
          user_prompt = user_prompt,
          model = model,
          temperature = temperature,
          max_tokens = max_tokens,
          api_key = api_key
        )
      } else {
        call_ollama(
          prompt = paste(system_prompt, "\n\n", user_prompt),
          model = model,
          temperature = temperature
        )
      }
    }, error = function(e) {
      if (verbose) {
        warning(sprintf("Error generating content for topic %s: %s", current_topic, e$message))
      }
      NA_character_
    })

    # Clean up response
    if (!is.na(generated_content)) {
      generated_content <- trimws(generated_content)
      generated_content <- gsub('^["\'](.*)["\']$', '\\1', generated_content)
    }

    unique_topics[[output_var]][i] <- generated_content

    # Rate limiting
    if (provider == "openai") {
      Sys.sleep(0.5)
    }
  }

  # Join generated content back to original data
  result <- top_terms %>%
    dplyr::left_join(
      unique_topics %>% dplyr::select(dplyr::all_of(c(topic_var, output_var))),
      by = topic_var
    )

  if (verbose) {
    n_generated <- sum(!is.na(unique_topics[[output_var]]))
    message(sprintf("Generated %d/%d %ss", n_generated, nrow(unique_topics), content_type))
  }

  result
}


#' Generate Survey Items from Topic Terms
#'
#' Convenience wrapper for \code{\link{generate_topic_content}} with
#' \code{content_type = "survey_item"}. Generates Likert-scale survey items
#' for scale development.
#'
#' @inheritParams generate_topic_content
#'
#' @return A data frame with generated survey items joined to original topic terms.
#'
#' @seealso \code{\link{generate_topic_content}}
#'
#' @family ai
#' @export
#'
#' @examples
#' \dontrun{
#' survey_items <- generate_survey_items(
#'   topic_terms_df = top_terms,
#'   provider = "openai",
#'   model = "gpt-3.5-turbo"
#' )
#' }
generate_survey_items <- function(topic_terms_df,
                                   topic_var = "topic",
                                   term_var = "term",
                                   weight_var = "beta",
                                   provider = c("openai", "ollama"),
                                   model = "gpt-3.5-turbo",
                                   temperature = 0,
                                   system_prompt = NULL,
                                   user_prompt_template = NULL,
                                   max_tokens = 150,
                                   api_key = NULL,
                                   verbose = TRUE) {

  generate_topic_content(
    topic_terms_df = topic_terms_df,
    content_type = "survey_item",
    topic_var = topic_var,
    term_var = term_var,
    weight_var = weight_var,
    provider = provider,
    model = model,
    temperature = temperature,
    system_prompt = system_prompt,
    user_prompt_template = user_prompt_template,
    max_tokens = max_tokens,
    api_key = api_key,
    output_var = "survey_item",
    verbose = verbose
  )
}


#' Call OpenAI Chat Completion API
#'
#' Internal function to call OpenAI's chat completion API.
#'
#' @param system_prompt System message for the chat.
#' @param user_prompt User message/query.
#' @param model Model to use (default: "gpt-3.5-turbo").
#' @param temperature Sampling temperature (default: 0).
#' @param max_tokens Maximum tokens in response (default: 150).
#' @param api_key OpenAI API key.
#'
#' @return Character string with the model's response.
#'
#' @keywords internal
call_openai_chat <- function(system_prompt,
                              user_prompt,
                              model = "gpt-3.5-turbo",
                              temperature = 0,
                              max_tokens = 150,
                              api_key) {

  if (!requireNamespace("httr", quietly = TRUE)) {
    stop("httr package is required for OpenAI API calls")
  }
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("jsonlite package is required for OpenAI API calls")
  }

  body_list <- list(
    model = model,
    messages = list(
      list(role = "system", content = system_prompt),
      list(role = "user", content = user_prompt)
    ),
    temperature = temperature,
    max_tokens = max_tokens
  )

  response <- httr::POST(
    url = "https://api.openai.com/v1/chat/completions",
    httr::add_headers(
      `Content-Type` = "application/json",
      `Authorization` = paste("Bearer", api_key)
    ),
    body = jsonlite::toJSON(body_list, auto_unbox = TRUE),
    encode = "json"
  )

  if (httr::status_code(response) != 200) {
    error_content <- httr::content(response, "text", encoding = "UTF-8")
    stop(sprintf("OpenAI API error (status %d): %s",
                 httr::status_code(response), error_content))
  }

  res_json <- jsonlite::fromJSON(httr::content(response, "text", encoding = "UTF-8"))

  if (!is.null(res_json$choices) && length(res_json$choices) > 0) {
    return(res_json$choices$message$content[1])
  }

  stop("Unexpected response structure from OpenAI API")
}


#' Analyze Contrastive Similarity (Alias)
#'
#' Alias for \code{\link{analyze_similarity_gaps}}. Identifies unique items,
#' missing content, and cross-category opportunities based on similarity thresholds.
#'
#' @inheritParams analyze_similarity_gaps
#'
#' @return A list containing unique_items, missing_items, cross_policy, and summary_stats.
#'
#' @seealso \code{\link{analyze_similarity_gaps}}
#'
#' @family ai
#' @export
analyze_contrastive_similarity <- function(similarity_data,
                                            ref_var = "ref_id",
                                            other_var = "other_id",
                                            similarity_var = "similarity",
                                            category_var = "other_category",
                                            ref_label_var = NULL,
                                            other_label_var = NULL,
                                            unique_threshold = 0.6,
                                            cross_policy_min = 0.6,
                                            cross_policy_max = 0.8) {
  analyze_similarity_gaps(
    similarity_data = similarity_data,
    ref_var = ref_var,
    other_var = other_var,
    similarity_var = similarity_var,
    category_var = category_var,
    ref_label_var = ref_label_var,
    other_label_var = other_label_var,
    unique_threshold = unique_threshold,
    cross_policy_min = cross_policy_min,
    cross_policy_max = cross_policy_max
  )
}


################################################################################
# LANGGRAPH WORKFLOWS
################################################################################

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
#' @family ai
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
#' @family ai
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
#' @family ai
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
#' @family ai
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
#' @family ai
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
