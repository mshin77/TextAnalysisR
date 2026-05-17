topic_modeling_ui_content <- function() {
      sidebarLayout(
        sidebarPanel(
          width = 3,
          class = "sidebar-panel",

          tags$h5(HTML("<strong>Topic modeling approach</strong>"), style = "color: #4269BF; margin-bottom: 10px;"),
          tags$div(
            id = "topic_modeling_path",
            class = "shiny-input-radiogroup shiny-input-container",
            role = "radiogroup",
            tags$div(
              class = "radio",
              style = "display: flex; align-items: center; margin-bottom: 8px;",
              tags$label(
                style = "margin-bottom: 0; display: flex; align-items: center; width: 100%;",
                tags$input(
                  type = "radio",
                  name = "topic_modeling_path",
                  value = "probability",
                  checked = "checked"
                ),
                tags$span("Structural Topic Model", style = "margin-left: 5px;"),
                actionLink("showSTMInfo", icon("info-circle"),
                          style = "color: #337ab7; font-size: 16px; margin-left: 8px;",
                          title = "Learn about STM")
              )
            ),
            tags$div(
              class = "radio",
              style = "display: flex; align-items: center; margin-bottom: 8px;",
              tags$label(
                style = "margin-bottom: 0; display: flex; align-items: center; width: 100%;",
                tags$input(
                  type = "radio",
                  name = "topic_modeling_path",
                  value = "embedding"
                ),
                tags$span("Embedding-based Topic Model", style = "margin-left: 5px;"),
                actionLink("showEmbeddingTopicsInfo", icon("info-circle"),
                          style = "color: #337ab7; font-size: 16px; margin-left: 8px;",
                          title = "Learn about Embedding-based Topics")
              )
            )
          ),
          hr(),

          conditionalPanel(
            condition = "input.topic_modeling_path == 'probability' && input.conditioned3 == 4 && input.searchKSubtabs != 'ai_rec'",
            tags$h5(
              HTML("<strong>Evaluate optimal topic number (K)</strong> <a href='https://github.com/bstewart/stm' target='_blank' rel='noopener noreferrer' onclick='window.open(this.href); return false;' style='font-size: 16px;'>Source</a>"),
              style = "color: #4269BF; margin-bottom: 10px;"
            ),
            sliderInput(
              "stm_K_range",
              "Range of topic numbers",
              value = c(5, 10),
              min = 2,
              max = 50
            ),
            selectizeInput(
              "stm_categorical_var",
              "Categorical covariate(s)",
              choices = NULL,
              multiple = TRUE,
              options = list(placeholder = "Optional")
            ),
            selectizeInput(
              "stm_continuous_var",
              "Continuous covariate(s)",
              choices = NULL,
              multiple = TRUE,
              options = list(placeholder = "Optional")
            ),
            selectInput("stm_init_type_search", "Initialization type",
              choices = c("Spectral", "LDA", "Random", "Custom"),
              selected = "Spectral"
            ),
            radioButtons("stm_gamma_prior_search", "Gamma prior",
              choices = c("Pooled", "L1"),
              selected = "Pooled"
            ),
            radioButtons("stm_kappa_prior_search", "Kappa prior",
              choices = c("L1", "Jeffreys"),
              selected = "L1"
            ),
            numericInput("stm_max_em_its_search", "Max EM iterations",
              value = 500, min = 100, max = 2000, step = 50
            ),
            div(
              style = "margin-bottom: 15px;",
              actionButton("stm_search", "Search K", class = "btn-primary btn-block")
            )
          ),
          conditionalPanel(
            condition = "input.topic_modeling_path == 'probability' && input.conditioned3 == 5",
            tags$h5(strong("Structural topic model"), style = "color: #4269BF; margin-bottom: 10px;"),
            uiOutput("stm_k_selector_uiOutput"),
            selectInput(
              "stm_topic_measure",
              "Topic term measure",
              choices = c(
                "FREX" = "frex",
                "Lift" = "lift",
                "Score" = "score",
                "Probability" = "beta"
              ),
              selected = "frex"
            ),
            conditionalPanel(
              condition = "input.stm_topic_measure == 'frex'",
              sliderInput(
                "stm_top_term_number_frex",
                "Top terms (FREX)",
                value = 5,
                min = 0,
                max = 50
              )
            ),
            conditionalPanel(
              condition = "input.stm_topic_measure == 'lift'",
              sliderInput(
                "stm_top_term_number_lift",
                "Top terms (Lift)",
                value = 5,
                min = 0,
                max = 50
              )
            ),
            conditionalPanel(
              condition = "input.stm_topic_measure == 'score'",
              sliderInput(
                "stm_top_term_number_score",
                "Top terms (Score)",
                value = 5,
                min = 0,
                max = 50
              )
            ),
            conditionalPanel(
              condition = "input.stm_topic_measure == 'beta'",
              sliderInput(
                "stm_top_term_number_beta",
                "Top terms (Probability)",
                value = 5,
                min = 0,
                max = 50
              )
            ),
            sliderInput(
              "stm_ncol_top_terms",
              "Column numbers",
              value = 2,
              min = 1,
              max = 10
            ),
            selectizeInput(
              "stm_categorical_var_2",
              "Categorical covariate(s)",
              choices = NULL,
              multiple = TRUE,
              options = list(placeholder = "Optional")
            ),
            selectizeInput(
              "stm_continuous_var_2",
              "Continuous covariate(s)",
              choices = NULL,
              multiple = TRUE,
              options = list(placeholder = "Optional")
            ),
            selectInput("stm_init_type_K", "Initialization type",
              choices = c("Spectral", "LDA", "Random", "Custom"),
              selected = "Spectral"
            ),
            radioButtons("stm_gamma_prior_K", "Gamma prior",
              choices = c("Pooled", "L1"),
              selected = "Pooled"
            ),
            radioButtons("stm_kappa_prior_K", "Kappa prior",
              choices = c("L1", "Jeffreys"),
              selected = "L1"
            ),
            numericInput("stm_max_em_its_K", "Max EM iterations",
              value = 500, min = 100, max = 2000, step = 50
            ),
            div(
              style = "margin-bottom: 15px;",
              actionButton("stm_run", "Display", class = "btn-primary btn-block")
            ),
            tags$hr(),
            tags$h5(strong("Generate topic labels using AI"), style = "color: #4269BF; margin-bottom: 10px;"),
            sliderInput(
              "stm_top_term_number_labeling",
              "Top terms for labeling",
              value = 7,
              min = 3,
              max = 15
            ),
            radioButtons(
              "stm_label_provider",
              "AI Provider:",
              choices = c(
                "Local (Ollama - Free, Private)" = "ollama",
                "OpenAI (API Key Required)" = "openai",
                "Gemini (API Key Required)" = "gemini"
              ),
              selected = "ollama",
              inline = FALSE
            ),
            conditionalPanel(
              condition = "input.stm_label_provider == 'ollama'",
              selectizeInput(
                "stm_label_ollama_model",
                "Ollama Model:",
                choices = c("Llama 3.2" = "llama3.2", "Gemma 3" = "gemma3", "Mistral" = "mistral"),
                selected = NULL,
                options = list(create = TRUE, placeholder = "Type your model...", onInitialize = I('function() { this.setValue(""); }'))
              ),
              tags$p(
                style = "font-size: 16px; color: #666;",
                "Requires Ollama. Get it from ",
                tags$a(href = "https://ollama.com", target = "_blank", "ollama.com")
              )
            ),
            conditionalPanel(
              condition = "input.stm_label_provider == 'openai'",
              selectizeInput(
                "stm_label_openai_model",
                "OpenAI Model:",
                choices = c("GPT-4.1 Mini (Default, fast)" = "gpt-4.1-mini", "GPT-4.1 (Accurate)" = "gpt-4.1", "GPT-4" = "gpt-4"),
                selected = NULL,
                options = list(create = TRUE, placeholder = "Type your model...", onInitialize = I('function() { this.setValue(""); }'))
              ),
              passwordInput("stm_label_openai_api_key", "API Key:", placeholder = "sk-..."),
              conditionalPanel(
                condition = "output.has_openai_key",
                tags$div(style = "color: #0C795A; font-size: 12px; margin-top: -8px; margin-bottom: 8px;",
                  icon("check-circle"), " Key stored. Enter new key to override.")
              )
            ),
            conditionalPanel(
              condition = "input.stm_label_provider == 'gemini'",
              selectizeInput(
                "stm_label_gemini_model",
                "Gemini Model:",
                choices = c("Gemini 2.5 Flash Lite (Default, economy)" = "gemini-2.5-flash-lite", "Gemini 2.5 Flash" = "gemini-2.5-flash", "Gemini 2.5 Pro (Accurate)" = "gemini-2.5-pro"),
                selected = NULL,
                options = list(create = TRUE, placeholder = "Type your model...", onInitialize = I('function() { this.setValue(""); }'))
              ),
              passwordInput("stm_label_gemini_api_key", "API Key:", placeholder = "AIza..."),
              conditionalPanel(
                condition = "output.has_gemini_key",
                tags$div(style = "color: #0C795A; font-size: 12px; margin-top: -8px; margin-bottom: 8px;",
                  icon("check-circle"), " Key stored. Enter new key to override.")
              )
            ),
            textAreaInput(
              "stm_system_prompt",
              "System prompt",
              value = "
You are a highly skilled data scientist specializing in generating concise and descriptive topic labels based on provided top terms for each topic.
Each topic consists of a list of terms ordered from most to least significant (by beta scores).

Your objective is to create precise labels that capture the essence of each topic by following these guidelines:

1. Use Person-First Language
   - Prioritize respectful and inclusive language.
   - Avoid terms that may be considered offensive or stigmatizing.
   - For example, use 'students with learning disabilities' instead of 'disabled students'.
   - Use 'students with visual impairments' instead of 'impaired students'
   - Use 'students with blindness' instead of 'blind students'.

1. Analyze Top Terms' Significance
   - Primary Focus: Emphasize high beta-score terms as they strongly define the topic.
   - Secondary Consideration: Include lower-scoring terms if they add essential context.

2. Synthesize the Topic Label
   - Clarity: Make sure the label is clear and easily understandable.
   - Conciseness: Aim for a short phrase of about 5-7 words.
   - Relevance: Reflect the collective meaning of the most influential terms.
   - Intelligent interpretation: Use your understanding to create meaningful labels that capture the topic's essence.

3. Maintain Consistency
   - Capitalize the first word of all topic labels.
   - Keep formatting and terminology uniform across all labels.
   - Avoid ambiguity or generic wording that does not fit the provided top terms.

4. Adhere to Style Guidelines
   - Capitalization: Use title case for labels.
   - Avoid Jargon: Maintain accessibility; only use technical terms if absolutely necessary.
   - Uniqueness: Ensure each label is distinct and does not overlap significantly with others.

5. Handle Edge Cases
   - Conflicting Top Terms: If the terms suggest different directions, prioritize those with higher beta scores.
   - Low-Scoring Terms: Include them only if they add meaningful context.

6. Iterative Improvement
   - If the generated label is insufficiently representative, re-check term significance and revise accordingly.
   - Always adhere to these guidelines.

Example
----------
Top Terms (highest to lowest beta score):
virtual manipulatives (.035)
manipulatives (.022)
mathematical (.014)
app (.013)
solving (.013)
learning disability (.012)
algebra (.012)
area (.011)
tool (.010)
concrete manipulatives (.010)

Generated Topic Label:
Mathematical learning tools for students with disabilities

Focus on incorporating the most significant keywords while following the guidelines above to produce a concise, descriptive topic label.
",
              rows = 10
            ),
            textAreaInput(
              "stm_user_prompt",
              "User prompt:",
              value = "You have a topic with keywords listed from most to least significant: [terms will be inserted here]. Please create a concise and descriptive label (5-7 words) that: 1. Reflects the collective meaning of these keywords. 2. Gives higher priority to the most significant terms. 3. Adheres to the style guidelines provided in the system message.",
              rows = 5
            ),
            sliderInput(
              "stm_temperature",
              "Temperature (creativity level)",
              min = 0, max = 1, value = 0.5, step = 0.1
            ),
            actionButton("topic_generate_labels", HTML('<i class="fas fa-wand-magic-sparkles"></i> Generate Labels'), class = "btn-primary btn-block"),
            tags$hr(),
            textInput(
              "stm_label_topics",
              "Manually label topics",
              value = "",
              placeholder = "Type labels, use comma to separate"
            )
          ),
          conditionalPanel(
            condition = "input.topic_modeling_path == 'probability' && input.conditioned3 == 6",
            uiOutput("topic_number_uiOutput"),
            sliderInput(
              "stm_top_term_number_2",
              "Top terms per topic",
              value = 5,
              min = 1,
              max = 20
            ),
            actionButton("stm_display", "Display", class = "btn-primary btn-block")
          ),
          conditionalPanel(
            condition = "input.topic_modeling_path == 'probability' && input.conditioned3 == 7",
            tags$h5(strong("Explore example documents"), style = "color: #4269BF; margin-bottom: 10px;"),
            uiOutput("quote_topic_number_uiOutput"),
            selectInput("stm_topic_texts", "Example quotes to display", choices = NULL),
            actionButton("stm_quote", "Quote", class = "btn-primary btn-block")
          ),
          conditionalPanel(
            condition = "input.topic_modeling_path == 'probability' && input.conditioned3 == 8",
            tags$h5(
              strong("Estimate Covariate Effects on Topics"),
              style = "color: #4269BF; margin-bottom: 10px;"
            ),
            tags$div(
              class = "status-step-purple",
              style = "margin-bottom: 15px;",
              tags$i(class = "fa fa-check-circle status-icon status-icon-purple"),
              "Choose covariates in the Word-Topic tab"
            ),
            div(
              style = "display: flex; gap: 10px; margin-bottom: 15px;",
              div(
                style = "flex: 1;",
                actionButton("stm_effect", "Estimate", class = "btn-primary btn-block")
              ),
              div(
                style = "flex: 1;",
                downloadButton("stm_effect_download_table", class = "btn-secondary btn-block")
              )
            )
          ),
          conditionalPanel(
            condition = "input.topic_modeling_path == 'probability' && input.conditioned3 == 4 && input.searchKSubtabs == 'ai_rec'",
            tags$h5(strong("AI Configuration for K Selection"), style = "color: #4269BF; margin-bottom: 10px;"),

            # AI Provider selection
            radioButtons(
              "k_rec_provider",
              "AI Provider:",
              choices = c(
                "Local (Ollama - Free, Private)" = "ollama",
                "OpenAI (API Key Required)" = "openai",
                "Gemini (API Key Required)" = "gemini"
              ),
              selected = "ollama",
              inline = FALSE
            ),

            conditionalPanel(
              condition = "input.k_rec_provider == 'ollama'",
              selectizeInput(
                "k_rec_ollama_model",
                "Ollama Model:",
                choices = c("Llama 3.2" = "llama3.2", "Gemma 3" = "gemma3", "Mistral" = "mistral"),
                selected = NULL,
                options = list(create = TRUE, placeholder = "Type your model...", onInitialize = I('function() { this.setValue(""); }'))
              ),
              tags$p(
                style = "font-size: 16px; color: #666;",
                "Requires Ollama. Get it from ",
                tags$a(href = "https://ollama.com", target = "_blank", "ollama.com")
              )
            ),

            conditionalPanel(
              condition = "input.k_rec_provider == 'openai'",
              selectizeInput(
                "k_rec_openai_model",
                "OpenAI Model:",
                choices = c(
                  "GPT-4.1 Mini (Default, fast)" = "gpt-4.1-mini",
                  "GPT-4.1 (Accurate)" = "gpt-4.1",
                  "GPT-4" = "gpt-4"
                ),
                selected = NULL,
                options = list(create = TRUE, placeholder = "Type your model...", onInitialize = I('function() { this.setValue(""); }'))
              ),
              passwordInput("k_rec_openai_api_key", "API Key:", placeholder = "sk-..."),
              conditionalPanel(
                condition = "output.has_openai_key",
                tags$div(style = "color: #0C795A; font-size: 12px; margin-top: -8px; margin-bottom: 8px;",
                  icon("check-circle"), " Key stored. Enter new key to override.")
              )
            ),

            conditionalPanel(
              condition = "input.k_rec_provider == 'gemini'",
              selectizeInput(
                "k_rec_gemini_model",
                "Gemini Model:",
                choices = c(
                  "Gemini 2.5 Flash Lite (Default, economy)" = "gemini-2.5-flash-lite",
                  "Gemini 2.5 Flash" = "gemini-2.5-flash",
                  "Gemini 2.5 Pro (Accurate)" = "gemini-2.5-pro"
                ),
                selected = NULL,
                options = list(create = TRUE, placeholder = "Type your model...", onInitialize = I('function() { this.setValue(""); }'))
              ),
              passwordInput("k_rec_gemini_api_key", "API Key:", placeholder = "AIza..."),
              conditionalPanel(
                condition = "output.has_gemini_key",
                tags$div(style = "color: #0C795A; font-size: 12px; margin-top: -8px; margin-bottom: 8px;",
                  icon("check-circle"), " Key stored. Enter new key to override.")
              )
            ),

            textAreaInput(
              "ai_system_search",
              "System prompt",
              value = "You are an expert in Structural Topic Modeling (STM) and topic model selection. Analyze the provided metrics to recommend the optimal number of topics (K).\n\nEvaluation criteria in order of importance:\n1. **Held-out likelihood**: Higher values indicate better generalization (most important for model selection)\n2. **Semantic coherence**: Higher values mean topics are more interpretable\n3. **Exclusivity**: Higher values mean topics are more distinct\n4. **Residuals**: Lower values indicate better model fit\n5. **Overall score**: A balanced metric combining all factors\n\nBest practice: Look for K values where:\n- Held-out likelihood plateaus or peaks\n- Semantic coherence and exclusivity are both reasonably high (upper-right quadrant)\n- There's an \"elbow\" in the residuals plot\n- The rate of improvement diminishes (diminishing returns)\n\nProvide a specific K recommendation with brief justification based on these metrics.",
              rows = 8
            ),
            textAreaInput(
              "ai_user_search",
              "User prompt",
              value = "Based on the Search K diagnostic plots and metrics table, what is the optimal number of topics (K)?\n\nKey observations:\n- Models with high coherence but low exclusivity may be too general\n- Models with high exclusivity but low coherence may be too specific\n- The ideal model balances both metrics while maximizing held-out likelihood\n- Consider practical interpretability: too few topics may oversimplify, too many may fragment meaningful themes\n\nPlease recommend a specific K value and explain which metrics most influenced your decision.",
              rows = 6
            ),
            actionButton(
              "generate_k_recommendation",
              HTML('<i class="fas fa-wand-magic-sparkles"></i> Generate Recommendation'),
              class = "btn-primary btn-block"
            )
          ),
          conditionalPanel(
            condition = "input.topic_modeling_path == 'probability' && input.conditioned3 == 9",
            tags$h5(strong("Plot Topic Effects by Categorical Covariates"), style = "color: #4269BF; margin-bottom: 10px;"),
            tags$div(
              class = "status-step-purple",
              tags$i(class = "fa fa-check-circle status-icon status-icon-purple"),
              tags$strong("Step 1:"), " Select categorical covariates in Word-Topic tab"
            ),
            tags$div(
              class = "status-step-blue",
              tags$i(class = "fa fa-check-circle status-icon status-icon-info"),
              tags$strong("Step 2:"), " Estimate coefficients in Estimated Effects tab"
            ),
            selectizeInput(
              "stm_effect_cat_btn",
              "Categorical covariate",
              choices = NULL,
              multiple = FALSE
            ),
            sliderInput(
              "stm_ncol_cat",
              "Column numbers",
              value = 2,
              min = 1,
              max = 10
            ),
            actionButton("stm_display_cat", "Display", class = "btn-primary btn-block")
          ),
          conditionalPanel(
            condition = "input.topic_modeling_path == 'probability' && input.conditioned3 == 10",
            tags$h5(strong("Plot Topic Effects by Continuous Covariates"), style = "color: #4269BF; margin-bottom: 10px;"),
            tags$div(
              class = "status-step-purple",
              tags$i(class = "fa fa-check-circle status-icon status-icon-purple"),
              tags$strong("Step 1:"), " Select continuous covariates in Word-Topic tab"
            ),
            tags$div(
              class = "status-step-blue",
              tags$i(class = "fa fa-check-circle status-icon status-icon-info"),
              tags$strong("Step 2:"), " Estimate coefficients in Estimated Effects tab"
            ),
            selectizeInput(
              "stm_effect_con_btn",
              "Continuous covariate",
              choices = NULL,
              multiple = FALSE
            ),
            sliderInput(
              "stm_ncol_con",
              "Column numbers",
              value = 2,
              min = 1,
              max = 10
            ),
            actionButton("stm_display_con", "Display", class = "btn-primary btn-block")
          ),

          conditionalPanel(
            condition = "input.conditioned3 == 'ai_content'",
            tags$h5(HTML("<strong>Generate content using AI</strong>"), style = "color: #4269BF; margin-bottom: 15px;"),
            tags$div(
              class = "status-main-info",
              tags$i(class = "fas fa-robot status-icon status-icon-info"),
              "Generate survey items, research questions, or other content from topic terms."
            ),
            selectInput(
              "content_type",
              "Content type",
              choices = c(
                "Survey Item" = "survey_item",
                "Research Question" = "research_question",
                "Theme Description" = "theme_description",
                "Policy Recommendation" = "policy_recommendation",
                "Interview Question" = "interview_question",
                "Custom" = "custom"
              ),
              selected = "survey_item"
            ),
            sliderInput(
              "content_top_terms",
              "Top terms for content",
              value = 7,
              min = 3,
              max = 15
            ),
            sliderInput(
              "content_max_tokens",
              "Max tokens",
              min = 50, max = 500, value = 150, step = 25
            ),
            radioButtons(
              "content_provider",
              "AI Provider:",
              choices = c(
                "Local (Ollama - Free, Private)" = "ollama",
                "OpenAI (API Key Required)" = "openai",
                "Gemini (API Key Required)" = "gemini"
              ),
              selected = "ollama",
              inline = FALSE
            ),
            conditionalPanel(
              condition = "input.content_provider == 'ollama'",
              selectizeInput(
                "content_ollama_model",
                "Ollama Model:",
                choices = c("Llama 3.2" = "llama3.2", "Gemma 3" = "gemma3", "Mistral" = "mistral"),
                selected = NULL,
                options = list(create = TRUE, placeholder = "Type your model...", onInitialize = I('function() { this.setValue(""); }'))
              ),
              tags$p(
                style = "font-size: 16px; color: #666;",
                "Requires Ollama. Get it from ",
                tags$a(href = "https://ollama.com", target = "_blank", "ollama.com")
              )
            ),
            conditionalPanel(
              condition = "input.content_provider == 'openai'",
              selectizeInput(
                "content_openai_model",
                "OpenAI Model:",
                choices = c("GPT-4.1 Mini (Default, fast)" = "gpt-4.1-mini", "GPT-4.1 (Accurate)" = "gpt-4.1", "GPT-4" = "gpt-4"),
                selected = NULL,
                options = list(create = TRUE, placeholder = "Type your model...", onInitialize = I('function() { this.setValue(""); }'))
              ),
              passwordInput("content_openai_api_key", "API Key:", placeholder = "sk-..."),
              conditionalPanel(
                condition = "output.has_openai_key",
                tags$div(style = "color: #0C795A; font-size: 12px; margin-top: -8px; margin-bottom: 8px;",
                  icon("check-circle"), " Key stored. Enter new key to override.")
              )
            ),
            conditionalPanel(
              condition = "input.content_provider == 'gemini'",
              selectizeInput(
                "content_gemini_model",
                "Gemini Model:",
                choices = c("Gemini 2.5 Flash Lite (Default, economy)" = "gemini-2.5-flash-lite", "Gemini 2.5 Flash" = "gemini-2.5-flash", "Gemini 2.5 Pro (Accurate)" = "gemini-2.5-pro"),
                selected = NULL,
                options = list(create = TRUE, placeholder = "Type your model...", onInitialize = I('function() { this.setValue(""); }'))
              ),
              passwordInput("content_gemini_api_key", "API Key:", placeholder = "AIza..."),
              conditionalPanel(
                condition = "output.has_gemini_key",
                tags$div(style = "color: #0C795A; font-size: 12px; margin-top: -8px; margin-bottom: 8px;",
                  icon("check-circle"), " Key stored. Enter new key to override.")
              )
            ),
            textAreaInput(
              "content_system_prompt",
              "System prompt",
              value = "",
              rows = 8
            ),
            textAreaInput(
              "content_user_prompt",
              "User prompt",
              value = "",
              rows = 5
            ),
            sliderInput(
              "content_temperature",
              "Temperature (creativity level)",
              min = 0, max = 1, value = 0.5, step = 0.1
            ),
            actionButton(
              "generate_topic_content",
              HTML('<i class="fas fa-pen-fancy"></i> Generate Content'),
              class = "btn-primary btn-block",
              style = "margin-top: 15px;"
            )
          ),
          conditionalPanel(
            condition = "input.topic_modeling_path == 'embedding' && input.conditioned3 == 4",
            tags$h5(HTML("<strong>Configure Embedding-based Topic Modeling</strong>"), style = "color: #4269BF; margin-bottom: 10px;"),

            selectInput(
              "embedding_backend",
              "Backend:",
              choices = c(
                "Python (BERTopic, recommended)" = "python",
                "R (no Python required)" = "r"
              ),
              selected = "python"
            ),

            uiOutput("embedding_topic_provider_status"),

            conditionalPanel(
              condition = "input.embedding_backend == 'python'",
              uiOutput("embedding_topic_backend_status")
            ),

            radioButtons(
              "topic_embedding_provider",
              "AI Provider:",
              choices = c(
                "Local (Ollama - Free, Private)" = "ollama",
                "Sentence Transformers (Python)" = "sentence-transformers",
                "OpenAI (API Key Required)" = "openai",
                "Gemini (API Key Required)" = "gemini"
              ),
              selected = "ollama",
              inline = FALSE
            ),

            conditionalPanel(
              condition = "input.topic_embedding_provider == 'ollama'",
              selectizeInput(
                "topic_embedding_ollama_model",
                "Ollama Model:",
                choices = c(
                  "Nomic Embed Text (Default)" = "nomic-embed-text",
                  "MxBai Embed Large (Higher Quality)" = "mxbai-embed-large",
                  "All-MiniLM (Lightweight)" = "all-minilm"
                ),
                selected = NULL,
                options = list(create = TRUE, placeholder = "Type your model...", onInitialize = I('function() { this.setValue(""); }'))
              ),
              tags$p(
                style = "font-size: 16px; color: #666;",
                "Requires Ollama. Get it from ",
                tags$a(href = "https://ollama.com", target = "_blank", "ollama.com")
              )
            ),

            conditionalPanel(
              condition = "input.topic_embedding_provider == 'sentence-transformers'",
              selectizeInput(
                "topic_embedding_st_model",
                "Model:",
                choices = c(
                  "all-MiniLM-L6-v2 (Fast)" = "all-MiniLM-L6-v2",
                  "all-mpnet-base-v2 (Higher Quality)" = "all-mpnet-base-v2",
                  "BGE Small EN v1.5 (Fast + Strong Retrieval)" = "BAAI/bge-small-en-v1.5",
                  "BGE Base EN v1.5 (BERTopic Optimized)" = "BAAI/bge-base-en-v1.5",
                  "E5 Base v2 (Instructor-tuned)" = "intfloat/e5-base-v2",
                  "Nomic Embed Text v2 (Multilingual)" = "nomic-ai/nomic-embed-text-v2-moe",
                  "GTE Multilingual Base (Fast, Multilingual)" = "Alibaba-NLP/gte-multilingual-base"
                ),
                selected = NULL,
                options = list(create = TRUE, placeholder = "Type your model...", onInitialize = I('function() { this.setValue(""); }'))
              ),
              tags$p(
                style = "font-size: 16px; color: #666;",
                "Requires Python + sentence-transformers"
              )
            ),

            conditionalPanel(
              condition = "input.topic_embedding_provider == 'openai'",
              selectizeInput(
                "topic_embedding_openai_model",
                "OpenAI Model:",
                choices = c(
                  "Text Embedding 3 Small (Default)" = "text-embedding-3-small",
                  "Text Embedding 3 Large (Higher Quality)" = "text-embedding-3-large"
                ),
                selected = NULL,
                options = list(create = TRUE, placeholder = "Type your model...", onInitialize = I('function() { this.setValue(""); }'))
              ),
              passwordInput("topic_embedding_openai_api_key", "API Key:", placeholder = "sk-..."),
              conditionalPanel(
                condition = "output.has_openai_key",
                tags$div(style = "color: #0C795A; font-size: 12px; margin-top: -8px; margin-bottom: 8px;",
                  icon("check-circle"), " Key stored. Enter new key to override.")
              )
            ),

            conditionalPanel(
              condition = "input.topic_embedding_provider == 'gemini'",
              selectizeInput(
                "topic_embedding_gemini_model",
                "Gemini Model:",
                choices = c("Gemini Embedding 001" = "gemini-embedding-001"),
                selected = NULL,
                options = list(create = TRUE, placeholder = "Type your model...", onInitialize = I('function() { this.setValue(""); }'))
              ),
              passwordInput("topic_embedding_gemini_api_key", "API Key:", placeholder = "AIza..."),
              conditionalPanel(
                condition = "output.has_gemini_key",
                tags$div(style = "color: #0C795A; font-size: 12px; margin-top: -8px; margin-bottom: 8px;",
                  icon("check-circle"), " Key stored. Enter new key to override.")
              )
            ),

            conditionalPanel(
              condition = "input.embedding_backend == 'python'",
              sliderInput(
                "embedding_umap_neighbors",
                "Neighbors:",
                value = 15,
                min = 5,
                max = 50
              ),
              sliderInput(
                "embedding_umap_n_components",
                "UMAP dimensions:",
                value = 5,
                min = 2,
                max = 50
              ),
              sliderInput(
                "embedding_umap_min_dist",
                "Min distance (0.0 = tight clusters for topic modeling):",
                value = 0.0,
                min = 0.0,
                max = 0.99,
                step = 0.01
              ),
              selectInput(
                "embedding_umap_metric",
                "Distance metric:",
                choices = c(
                  "Cosine (recommended for text)" = "cosine",
                  "Euclidean" = "euclidean"
                ),
                selected = "cosine"
              ),
              sliderInput(
                "embedding_min_topic_size",
                "Min cluster size:",
                value = 10,
                min = 2,
                max = 50
              ),
              selectInput(
                "embedding_cluster_selection",
                "Cluster selection method:",
                choices = c(
                  "EOM (broader topics, default)" = "eom",
                  "Leaf (smaller, focused topics)" = "leaf"
                ),
                selected = "eom"
              )
            ),

            conditionalPanel(
              condition = "input.embedding_backend == 'r'",
              selectInput(
                "embedding_dimred_method",
                "Dimensionality reduction:",
                choices = c(
                  "UMAP" = "umap",
                  "t-SNE" = "tsne",
                  "PCA" = "pca"
                ),
                selected = "umap"
              ),

              conditionalPanel(
                condition = "input.embedding_dimred_method == 'umap'",
                sliderInput(
                  "embedding_r_umap_neighbors",
                  "Neighbors:",
                  value = 15,
                  min = 5,
                  max = 50
                ),
                sliderInput(
                  "embedding_r_umap_n_components",
                  "UMAP dimensions:",
                  value = 5,
                  min = 2,
                  max = 50
                ),
                sliderInput(
                  "embedding_r_umap_min_dist",
                  "Min distance (0.0 = tight clusters for topic modeling):",
                  value = 0.0,
                  min = 0.0,
                  max = 0.5,
                  step = 0.01
                ),
                selectInput(
                  "embedding_r_umap_metric",
                  "Distance metric:",
                  choices = c(
                    "Cosine (recommended for text)" = "cosine",
                    "Euclidean" = "euclidean"
                  ),
                  selected = "cosine"
                )
              ),

              conditionalPanel(
                condition = "input.embedding_dimred_method == 'tsne'",
                sliderInput(
                  "embedding_tsne_perplexity",
                  "Perplexity:",
                  value = 30,
                  min = 5,
                  max = 100
                )
              ),

              conditionalPanel(
                condition = "input.embedding_dimred_method == 'pca'",
                sliderInput(
                  "embedding_pca_dims",
                  "Components:",
                  value = 50,
                  min = 10,
                  max = 200
                )
              ),

              selectInput(
                "embedding_method_r",
                "Clustering method:",
                choices = c(
                  "DBSCAN (density-based)" = "dbscan",
                  "K-means" = "kmeans",
                  "Hierarchical" = "hierarchical",
                  "HDBSCAN" = "hdbscan"
                ),
                selected = "dbscan"
              ),

              conditionalPanel(
                condition = "input.embedding_method_r == 'kmeans' || input.embedding_method_r == 'hierarchical'",
                numericInput(
                  "embedding_r_n_topics",
                  "Number of clusters:",
                  value = 5,
                  min = 2,
                  max = 50
                )
              ),

              conditionalPanel(
                condition = "input.embedding_method_r == 'dbscan'",
                sliderInput(
                  "embedding_dbscan_eps",
                  "Epsilon:",
                  value = 0.5,
                  min = 0.1,
                  max = 2.0,
                  step = 0.1
                ),
                sliderInput(
                  "embedding_dbscan_minpts",
                  "Min points:",
                  value = 5,
                  min = 2,
                  max = 20
                )
              ),

              conditionalPanel(
                condition = "input.embedding_method_r == 'hdbscan'",
                sliderInput(
                  "embedding_r_min_cluster_size",
                  "Min cluster size:",
                  value = 5,
                  min = 2,
                  max = 50
                )
              ),

              checkboxInput(
                "embedding_r_reduce_outliers",
                "Reduce outliers (for DBSCAN/HDBSCAN)",
                value = TRUE
              )
            ),

            actionButton("embedding_run", "Run Model", class = "btn-primary btn-block")
          ),

          conditionalPanel(
            condition = "input.topic_modeling_path == 'embedding' && input.conditioned3 == 5",
            uiOutput("embedding_topics_info"),
            actionButton("embedding_display", "Display", class = "btn-primary btn-block")
          ),

          conditionalPanel(
            condition = "input.topic_modeling_path == 'embedding' && input.conditioned3 == 6",
            selectInput(
              "embedding_viz_type",
              "Visualization type",
              choices = c(
                "Topic Distribution" = "distribution",
                "Document Clusters" = "documents",
                "Topic Similarity Heatmap" = "heatmap",
                "Intertopic Distance" = "distance"
              ),
              selected = "distribution"
            )
          ),

          conditionalPanel(
            condition = "input.topic_modeling_path == 'embedding' && input.conditioned3 == 7",
            sliderInput(
              "embedding_quote_number",
              "Quotes to display",
              value = 3,
              min = 1,
              max = 10
            ),
            selectInput(
              "embedding_quote_topic",
              "Topic",
              choices = NULL
            ),
            actionButton("embedding_quote", "Quote", class = "btn-primary btn-block")
          ),






        ),
        mainPanel(
          width = 9,
          tabsetPanel(
            id = "conditioned3",
            tabPanel(
              "1. Model Configuration",
              value = 4,
              bsCollapse(
                open = 0,
                bsCollapsePanel(
                  p(strong("Click to set plot dimensions"),
                    class = "plot-dimensions-text"
                  ),
                  value = 4,
                  style = "success",
                  p(strong("Dimensions of the plots")),
                  div(
                    style = "display:inline-block",
                    sliderInput(
                      inputId = "height_search_k",
                      post = " px",
                      label = "height",
                      min = 200,
                      max = 4000,
                      step = 5,
                      value = 600
                    )
                  ),
                  div(
                    style = "display:inline-block",
                    sliderInput(
                      inputId = "width_search_k",
                      post = " px",
                      label = "width",
                      min = 500,
                      max = 3000,
                      step = 5,
                      value = 1000
                    )
                  )
                )
              ),
              conditionalPanel(
                condition = "output.has_search_k_results == true",
                tabsetPanel(
                  id = "searchKSubtabs",
                  tabPanel(
                    "Diagnostic Plots",
                    value = "diagnostic",
                    br(),
                    uiOutput("topic_search_message"),
                    br(),
                    fluidRow(
                      column(6, uiOutput("quality_metrics_semcoh_uiOutput")),
                      column(6, uiOutput("quality_metrics_residual_uiOutput"))
                    ),
                    fluidRow(
                      column(6, uiOutput("quality_metrics_heldout_uiOutput")),
                      column(6, uiOutput("quality_metrics_lbound_uiOutput"))
                    )
                  ),
                  tabPanel(
                    "Quality Comparison",
                    value = "quality",
                    br(),
                    wellPanel(
                      style = "background-color: #f0f8ff; border: 1px solid #4682b4;",
                      p("Overall Score = Coherence(z) + Exclusivity(z) - Residual(z) + Heldout(z)",
                        style = "font-family: monospace; font-size: 16px; font-weight: bold;"),
                      tags$div(
                        style = "font-size: 16px; margin-top: 10px; line-height: 1.6;",
                        p(tags$b("Coherence:"), " How interpretable topics are based on co-occurring words", style = "margin: 5px 0;"),
                        p(tags$b("Exclusivity:"), " How distinctive topics are from each other", style = "margin: 5px 0;"),
                        p(tags$b("Residual:"), " Model fit to data (lower is better)", style = "margin: 5px 0;"),
                        p(tags$b("Heldout:"), " Model's ability to generalize to new data", style = "margin: 5px 0;")
                      )
                    ),
                    br(),
                    DT::dataTableOutput("quality_summary_table")
                  ),
                  tabPanel(
                    "Coherence vs Exclusivity",
                    value = "comparison",
                    br(),
                    uiOutput("model_comparison_plot_uiOutput")
                  ),
                  tabPanel(
                    "AI Recommendation",
                    value = "ai_rec",
                    br(),
                    uiOutput("ai_recommendation_output")
                  )
                )
              ),
              conditionalPanel(
                condition = "output.has_search_k_results == false && input.topic_modeling_path == 'probability'",
                div(
                  style = "padding: 60px 40px; text-align: center;",
                  div(
                    style = "max-width: 400px; margin: 0 auto;",
                    tags$i(class = "fa fa-search-plus", style = "font-size: 48px; color: #CBD5E1; margin-bottom: 20px; display: block;", "aria-hidden" = "true"),
                    tags$p(
                      "Configure K range and click ",
                      tags$strong("'Search K'", style = "color: #4269BF;"),
                      " to find optimal topic numbers",
                      style = "font-size: 18px; font-weight: 400; line-height: 1.7; color: #64748B; margin: 0;"
                    )
                  )
                )
              ),
              conditionalPanel(
                condition = "input.topic_modeling_path == 'embedding'",
                div(
                  style = "padding: 60px 40px; text-align: center;",
                  div(
                    style = "max-width: 400px; margin: 0 auto;",
                    tags$i(class = "fa fa-cogs", style = "font-size: 48px; color: #CBD5E1; margin-bottom: 20px; display: block;", "aria-hidden" = "true"),
                    tags$p(
                      "Configure settings and click ",
                      tags$strong("'Run Model'", style = "color: #4269BF;"),
                      " to discover topics",
                      style = "font-size: 18px; font-weight: 400; line-height: 1.7; color: #64748B; margin: 0;"
                    )
                  )
                )
              ),
            ),
            tabPanel(
              "2. Word-Topic",
              value = 5,
              bsCollapse(
                open = 0,
                bsCollapsePanel(
                  p(strong("Click to set plot dimensions"),
                    class = "plot-dimensions-text"
                  ),
                  value = 1,
                  style = "success",
                  p(strong("Dimensions of the plot")),
                  div(
                    style = "display:inline-block",
                    sliderInput(
                      inputId = "height",
                      post = " px",
                      label = "height",
                      min = 200,
                      max = 4000,
                      step = 5,
                      value = 1000
                    )
                  ),
                  div(
                    style = "display:inline-block",
                    sliderInput(
                      inputId = "width",
                      post = " px",
                      label = "width",
                      min = 500,
                      max = 3000,
                      step = 5,
                      value = 1000
                    )
                  )
                )
              ),
              tags$style(
                HTML(
                  ".plot-container {
                                max-height: 4000px;
                                max-width: 3000px;
                                overflow: auto; }"
                )
              ),
              conditionalPanel(
                condition = "output.has_word_topic_results == true",
                uiOutput("topic_term_message"),
                uiOutput("topic_term_plot_uiOutput"),
                br(),
                uiOutput("topic_term_table_uiOutput"),
                br()
              ),
              conditionalPanel(
                condition = "output.has_word_topic_results == false && input.topic_modeling_path == 'probability'",
                div(
                  style = "padding: 60px 40px; text-align: center;",
                  div(
                    style = "max-width: 400px; margin: 0 auto;",
                    tags$i(class = "fa fa-project-diagram", style = "font-size: 48px; color: #CBD5E1; margin-bottom: 20px; display: block;", "aria-hidden" = "true"),
                    tags$p(
                      "Search K, and then click ",
                      tags$strong("'Display'", style = "color: #4269BF;"),
                      " to view word-topic distributions",
                      style = "font-size: 18px; font-weight: 400; line-height: 1.7; color: #64748B; margin: 0;"
                    )
                  )
                )
              ),
              conditionalPanel(
                condition = "output.has_word_topic_results == false && input.topic_modeling_path == 'embedding'",
                div(
                  style = "padding: 60px 40px; text-align: center;",
                  div(
                    style = "max-width: 400px; margin: 0 auto;",
                    tags$i(class = "fa fa-project-diagram", style = "font-size: 48px; color: #CBD5E1; margin-bottom: 20px; display: block;", "aria-hidden" = "true"),
                    tags$p(
                      "Run model, then click ",
                      tags$strong("'Display'", style = "color: #4269BF;"),
                      " to view topic keywords",
                      style = "font-size: 18px; font-weight: 400; line-height: 1.7; color: #64748B; margin: 0;"
                    )
                  )
                )
              ),
            ),
            tabPanel(
              "3. Content Generation",
              value = "ai_content",
              div(
                style = "padding: 20px;",

                conditionalPanel(
                  condition = "output.has_generated_content == true",
                  DT::dataTableOutput("generated_content_table")
                ),
                conditionalPanel(
                  condition = "output.has_generated_content == false",
                  div(
                    style = "padding: 60px 40px; text-align: center;",
                    tags$i(class = "fas fa-file-alt", style = "font-size: 48px; color: #CBD5E1; margin-bottom: 20px; display: block;"),
                    tags$p(
                      "Run topic modeling first, then configure settings in the sidebar and click ",
                      tags$strong("'Generate Content'", style = "color: #4269BF;"),
                      " to create content from your topics.",
                      style = "font-size: 18px; font-weight: 400; line-height: 1.7; color: #64748B; margin: 0;"
                    )
                  )
                )
              )
            ),

            tabPanel(
              "4. Document-Topic",
              value = 6,
              bsCollapse(
                open = 0,
                bsCollapsePanel(
                  p(strong("Click to set plot dimensions"),
                    class = "plot-dimensions-text"
                  ),
                  value = 1,
                  style = "success",
                  p(strong("Dimensions of the plot")),
                  div(
                    style = "display:inline-block",
                    sliderInput(
                      inputId = "height_topic_prevalence",
                      post = " px",
                      label = "height",
                      min = 200,
                      max = 4000,
                      step = 5,
                      value = 500
                    )
                  ),
                  div(
                    style = "display:inline-block",
                    sliderInput(
                      inputId = "width_topic_prevalence",
                      post = " px",
                      label = "width",
                      min = 500,
                      max = 3000,
                      step = 5,
                      value = 1000
                    )
                  )
                )
              ),
              tags$style(
                HTML(
                  ".plot-container {
                                max-height: 4000px;
                                max-width: 3000px;
                                overflow: auto; }"
                )
              ),
              conditionalPanel(
                condition = "output.has_document_topic_results == true",
                uiOutput("topic_prevalence_plot_uiOutput"),
                br(),
                uiOutput("topic_prevalence_table_uiOutput")
              ),
              conditionalPanel(
                condition = "output.has_document_topic_results == false && input.topic_modeling_path == 'probability'",
                div(
                  style = "padding: 60px 40px; text-align: center;",
                  div(
                    style = "max-width: 400px; margin: 0 auto;",
                    tags$i(class = "fa fa-file-alt", style = "font-size: 48px; color: #CBD5E1; margin-bottom: 20px; display: block;", "aria-hidden" = "true"),
                    tags$p(
                      "Complete ",
                      tags$strong("Word-Topic tab", style = "color: #4269BF;"),
                      " to view document-topic distributions",
                      style = "font-size: 18px; font-weight: 400; line-height: 1.7; color: #64748B; margin: 0;"
                    )
                  )
                )
              ),
              conditionalPanel(
                condition = "output.has_document_topic_results == false && input.topic_modeling_path == 'embedding'",
                div(
                  style = "padding: 60px 40px; text-align: center;",
                  div(
                    style = "max-width: 400px; margin: 0 auto;",
                    tags$i(class = "fa fa-file-alt", style = "font-size: 48px; color: #CBD5E1; margin-bottom: 20px; display: block;", "aria-hidden" = "true"),
                    tags$p(
                      "Complete ",
                      tags$strong("Word-Topic tab", style = "color: #4269BF;"),
                      " to view document-topic distributions",
                      style = "font-size: 18px; font-weight: 400; line-height: 1.7; color: #64748B; margin: 0;"
                    )
                  )
                )
              ),
            ),
            tabPanel(
              "5. Quotes",
              value = 7,
              conditionalPanel(
                condition = "output.has_quotes == true",
                DT::dataTableOutput("quote_table")
              ),
              conditionalPanel(
                condition = "output.has_quotes == false",
                div(
                  style = "padding: 60px 40px; text-align: center;",
                  div(
                    style = "max-width: 400px; margin: 0 auto;",
                    tags$i(class = "fa fa-quote-right", style = "font-size: 48px; color: #CBD5E1; margin-bottom: 20px; display: block;", "aria-hidden" = "true"),
                    tags$p(
                      "Run topic model and select a topic to view ",
                      tags$strong("representative quotes", style = "color: #4269BF;"),
                      style = "font-size: 18px; font-weight: 400; line-height: 1.7; color: #64748B; margin: 0;"
                    )
                  )
                )
              )
            ),
            tabPanel(
              "5. Estimated Effects",
              value = 8,
              conditionalPanel(
                condition = "output.has_effect_estimates == true",
                DT::dataTableOutput("effect_table")
              ),
              conditionalPanel(
                condition = "output.has_effect_estimates == false",
                div(
                  style = "padding: 60px 40px; text-align: center;",
                  div(
                    style = "max-width: 400px; margin: 0 auto;",
                    tags$i(class = "fa fa-calculator", style = "font-size: 48px; color: #CBD5E1; margin-bottom: 20px; display: block;"),
                    tags$p(
                      "Click ",
                      tags$strong("'Estimate'", style = "color: #4269BF;"),
                      " button to generate effect estimates",
                      style = "font-size: 18px; font-weight: 400; line-height: 1.7; color: #64748B; margin: 0;"
                    )
                  )
                )
              )
            ),
            tabPanel(
              "6. Categorical Covariates",
              value = 9,
              bsCollapse(
                open = 0,
                bsCollapsePanel(
                  p(strong("Click to set plot dimensions"),
                    class = "plot-dimensions-text"
                  ),
                  value = 1,
                  style = "success",
                  p(strong("Dimensions of the plot")),
                  div(
                    style = "display:inline-block",
                    sliderInput(
                      inputId = "height_cat_plot",
                      post = " px",
                      label = "height",
                      min = 200,
                      max = 4000,
                      step = 5,
                      value = 500
                    )
                  ),
                  div(
                    style = "display:inline-block",
                    sliderInput(
                      inputId = "width_cat_plot",
                      post = " px",
                      label = "width",
                      min = 500,
                      max = 3000,
                      step = 5,
                      value = 1000
                    )
                  )
                )
              ),
              tags$style(
                HTML(
                  ".plot-container {
                                max-height: 4000px;
                                max-width: 3000px;
                                overflow: auto; }"
                )
              ),
              conditionalPanel(
                condition = "output.has_categorical_plot == true",
                uiOutput("cat_plot_uiOutput"),
                br(),
                uiOutput("cat_table_uiOutput")
              ),
              conditionalPanel(
                condition = "output.has_categorical_plot == false",
                div(
                  style = "padding: 60px 40px; text-align: center;",
                  div(
                    style = "max-width: 400px; margin: 0 auto;",
                    tags$i(class = "fa fa-chart-bar", style = "font-size: 48px; color: #CBD5E1; margin-bottom: 20px; display: block;", "aria-hidden" = "true"),
                    tags$p(
                      "Estimate effects, select categorical covariate, then click ",
                      tags$strong("'Display'", style = "color: #4269BF;"),
                      " to visualize topic prevalence by categories",
                      style = "font-size: 18px; font-weight: 400; line-height: 1.7; color: #64748B; margin: 0;"
                    )
                  )
                )
              )
            ),
            tabPanel(
              "7. Continuous Covariates",
              value = 10,
              bsCollapse(
                open = 0,
                bsCollapsePanel(
                  p(strong("Click to set plot dimensions"),
                    class = "plot-dimensions-text"
                  ),
                  value = 1,
                  style = "success",
                  p(strong("Dimensions of the plot")),
                  div(
                    style = "display:inline-block",
                    sliderInput(
                      inputId = "height_con_plot",
                      post = " px",
                      label = "height",
                      min = 200,
                      max = 4000,
                      step = 5,
                      value = 500
                    )
                  ),
                  div(
                    style = "display:inline-block",
                    sliderInput(
                      inputId = "width_con_plot",
                      post = " px",
                      label = "width",
                      min = 500,
                      max = 3000,
                      step = 5,
                      value = 1000
                    )
                  )
                )
              ),
              tags$style(
                HTML(
                  ".plot-container {
                                max-height: 4000px;
                                max-width: 3000px;
                                overflow: auto; }"
                )
              ),
              conditionalPanel(
                condition = "output.has_continuous_plot == true",
                uiOutput("con_plot_uiOutput"),
                br(),
                uiOutput("con_table_uiOutput")
              ),
              conditionalPanel(
                condition = "output.has_continuous_plot == false",
                div(
                  style = "padding: 60px 40px; text-align: center;",
                  div(
                    style = "max-width: 400px; margin: 0 auto;",
                    tags$i(class = "fa fa-chart-line", style = "font-size: 48px; color: #CBD5E1; margin-bottom: 20px; display: block;", "aria-hidden" = "true"),
                    tags$p(
                      "Estimate effects, select continuous covariate, then click ",
                      tags$strong("'Display'", style = "color: #4269BF;"),
                      " to visualize topic prevalence trends",
                      style = "font-size: 18px; font-weight: 400; line-height: 1.7; color: #64748B; margin: 0;"
                    )
                  )
                )
              )
            )
          )
        )
      )
}
