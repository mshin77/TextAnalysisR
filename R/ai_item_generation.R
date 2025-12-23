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
