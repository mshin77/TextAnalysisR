suppressPackageStartupMessages({
  library(shiny)
  library(shinyBS)
  library(shinyjs)
})

# Detect web/Docker deployment (hide GPU option in these environments)
is_web <- tryCatch({
  TextAnalysisR:::check_web_deployment()
}, error = function(e) FALSE)
is_docker <- tryCatch({
  TextAnalysisR:::check_docker_deployment()
}, error = function(e) FALSE)
is_remote_ui <- is_web || is_docker


ui <- fluidPage(
  lang = "en",
  useShinyjs(),
  shinybusy::add_busy_spinner(
    spin = "fading-circle",
    position = "full-page",
    color = "#337ab7",
    height = "100px",
    width = "100px"
  ),
  tags$head(
    tags$meta(charset = "UTF-8"),
    tags$meta(name = "viewport", content = "width=device-width, initial-scale=1.0"),
    tags$meta(name = "description", content = "TextAnalysisR: A text mining workflow tool"),
    tags$link(rel = "icon", type = "image/png", href = "logo.png"),
    tags$script(HTML(
      "document.querySelectorAll('title').forEach(function(t, i) {
         if (i > 0 || !t.textContent.trim()) t.remove();
       });"
    )),
    tags$meta(name = "keywords", content = "text mining, topic modeling, semantic analysis, R Shiny"),

    tags$meta(
      `http-equiv` = "Content-Security-Policy",
      content = paste(
        "default-src 'self';",
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net https://cdn.plot.ly https://translate.google.com https://translate.googleapis.com https://translate.googleusercontent.com https://translate-pa.googleapis.com https://static.cloudflareinsights.com;",
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://translate.googleapis.com https://translate.googleusercontent.com https://www.gstatic.com https://use.fontawesome.com https://cdnjs.cloudflare.com https://ka-f.fontawesome.com;",
        "font-src 'self' https://fonts.gstatic.com https://use.fontawesome.com https://cdnjs.cloudflare.com https://ka-f.fontawesome.com data:;",
        "img-src 'self' data: https: https://www.gstatic.com https://translate.google.com;",
        "connect-src 'self' http://127.0.0.1:* http://localhost:* https://translate.googleapis.com https://translate-pa.googleapis.com https://cloudflareinsights.com;",
        "frame-src 'self' http: https: https://translate.google.com https://translate.googleusercontent.com;"
      )
    ),
    tags$meta(`http-equiv` = "X-Content-Type-Options", content = "nosniff"),
    tags$meta(`http-equiv` = "X-XSS-Protection", content = "1; mode=block"),
    tags$meta(`http-equiv` = "Referrer-Policy", content = "no-referrer-when-downgrade"),

    tags$meta(name = "theme-color", content = "#337ab7"),

    # Font Awesome icons (preload to avoid render-blocking)
    tags$link(
      rel = "preload",
      href = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css",
      as = "style",
      onload = "this.onload=null;this.rel='stylesheet'",
      integrity = "sha512-1ycn6IcaQQ40/MKBW2W4Rhis/DbILU74C1vSrLJxCq57o941Ym01SwNsOMqvEBFlcgUa6xLiPY/NS5R+E6ztJQ==",
      crossorigin = "anonymous",
      referrerpolicy = "no-referrer"
    ),

    # Google Translate API loads on demand the first time the user opens the
    # language picker. Avoids ~600 ms of render-blocking 3rd-party JS for every
    # session, since most researchers never touch the translation widget.
    tags$script(HTML("
      window.loadGoogleTranslate = function() {
        if (window.__gtLoaded) return;
        window.__gtLoaded = true;
        var s = document.createElement('script');
        s.src = 'https://translate.google.com/translate_a/element.js?cb=googleTranslateElementInit';
        s.async = true;
        document.head.appendChild(s);
      };
    ")),

    tags$style(HTML("
      .password-wrapper { position: relative; }
      .password-wrapper input[type='password'],
      .password-wrapper input[type='text'] { padding-right: 36px; }
      .password-toggle {
        position: absolute; right: 8px; top: 50%; transform: translateY(-50%);
        background: none; border: none; cursor: pointer; color: #888;
        padding: 2px 4px; font-size: 14px; line-height: 1; z-index: 2;
      }
      .password-toggle:hover { color: #333; }
    ")),
    tags$script(HTML("
      $(document).on('shiny:value shiny:inputchanged shiny:bound', function() {
        setTimeout(function() {
          $('input[type=\"password\"]').each(function() {
            if ($(this).parent().hasClass('password-wrapper')) return;
            $(this).wrap('<div class=\"password-wrapper\"></div>');
            var btn = $('<button type=\"button\" class=\"password-toggle\" title=\"Show/hide\" aria-label=\"Toggle visibility\"><i class=\"fa fa-eye\"></i></button>');
            $(this).after(btn);
          });
        }, 100);
      });
      $(document).on('click', '.password-toggle', function(e) {
        e.preventDefault();
        var input = $(this).siblings('input');
        var icon = $(this).find('i');
        if (input.attr('type') === 'password') {
          input.attr('type', 'text');
          icon.removeClass('fa-eye').addClass('fa-eye-slash');
        } else {
          input.attr('type', 'password');
          icon.removeClass('fa-eye-slash').addClass('fa-eye');
        }
      });
      $(document).ready(function() {
        $('input[type=\"password\"]').each(function() {
          if ($(this).parent().hasClass('password-wrapper')) return;
          $(this).wrap('<div class=\"password-wrapper\"></div>');
          var btn = $('<button type=\"button\" class=\"password-toggle\" title=\"Show/hide\" aria-label=\"Toggle visibility\"><i class=\"fa fa-eye\"></i></button>');
          $(this).after(btn);
        });
      });
    "))
  ),

  tags$div(
    class = "top-right-controls",
    tags$a(
      id = "tts_toggle",
      href = "javascript:void(0);",
      style = "cursor: pointer; text-decoration: none; color: #5F7088; font-size: 26px; user-select: none;",
      `aria-label` = "Text to speech",
      title = "Text to Speech (Alt+S)",
      tags$i(class = "fa fa-volume-up", id = "tts_icon", `aria-hidden` = "true", style = "pointer-events: none;")
    ),
    tags$a(
      id = "dark_mode_toggle",
      href = "javascript:void(0);",
      onclick = "toggleDarkMode()",
      style = "cursor: pointer; text-decoration: none; color: #5F7088; font-size: 30px;",
      `aria-label` = "Toggle dark mode",
      title = "Toggle dark mode",
      tags$i(class = "fa fa-moon", `aria-hidden` = "true")
    ),
    tags$div(
      style = "position: relative; display: flex; align-items: center; gap: 8px;",
      tags$a(
        id = "translate_icon",
        href = "javascript:void(0);",
        style = "cursor: pointer; text-decoration: none; color: #5F7088; font-size: 26px; user-select: none;",
        `aria-label` = "Select language",
        title = "Select Language",
        tags$i(class = "fa fa-globe", `aria-hidden` = "true", style = "pointer-events: none;")
      ),
      # Hidden Google Translate element (accessed programmatically)
      tags$div(
        id = "google_translate_element",
        style = "position: absolute; left: -9999px; top: -9999px;"
      ),
      # Custom language menu dropdown
      tags$div(
        id = "translate_dropdown",
        class = "notranslate",
        translate = "no",
        style = "display: none; position: absolute; top: 100%; right: 0; margin-top: 8px; background: white; border: 1px solid #e5e7eb; border-radius: 8px; box-shadow: 0 4px 16px rgba(0,0,0,0.15); z-index: 10000; min-width: 200px; max-height: 360px; overflow-y: auto;",
        # Language buttons
        tags$button(class = "lang-btn active", `data-lang` = "en", lang = "en", tags$i(class = "fa fa-check"), " English"),
        tags$button(class = "lang-btn", `data-lang` = "de", lang = "de", tags$i(class = "fa fa-check"), " Deutsch"),
        tags$button(class = "lang-btn", `data-lang` = "es", lang = "es", tags$i(class = "fa fa-check"), " Español"),
        tags$button(class = "lang-btn", `data-lang` = "fr", lang = "fr", tags$i(class = "fa fa-check"), " Français"),
        tags$button(class = "lang-btn", `data-lang` = "hi", lang = "hi", tags$i(class = "fa fa-check"), " हिन्दी"),
        tags$button(class = "lang-btn", `data-lang` = "id", lang = "id", tags$i(class = "fa fa-check"), " Bahasa Indonesia"),
        tags$button(class = "lang-btn", `data-lang` = "it", lang = "it", tags$i(class = "fa fa-check"), " Italiano"),
        tags$button(class = "lang-btn", `data-lang` = "ja", lang = "ja", tags$i(class = "fa fa-check"), " 日本語"),
        tags$button(class = "lang-btn", `data-lang` = "ko", lang = "ko", tags$i(class = "fa fa-check"), " 한국어"),
        tags$button(class = "lang-btn", `data-lang` = "nl", lang = "nl", tags$i(class = "fa fa-check"), " Nederlands"),
        tags$button(class = "lang-btn", `data-lang` = "pl", lang = "pl", tags$i(class = "fa fa-check"), " Polski"),
        tags$button(class = "lang-btn", `data-lang` = "pt", lang = "pt", tags$i(class = "fa fa-check"), " Português"),
        tags$button(class = "lang-btn", `data-lang` = "ru", lang = "ru", tags$i(class = "fa fa-check"), " Русский"),
        tags$button(class = "lang-btn", `data-lang` = "zh-CN", lang = "zh-Hans", tags$i(class = "fa fa-check"), " 中文 (简体)"),
        tags$button(class = "lang-btn", `data-lang` = "zh-TW", lang = "zh-Hant", tags$i(class = "fa fa-check"), " 中文 (繁體)"),
        # Attribution at bottom
        tags$div(
          class = "translate-attribution",
          style = "display: flex; align-items: center; justify-content: center; gap: 4px; font-size: 16px; color: #9ca3af; padding: 8px 12px; border-top: 1px solid #e5e7eb; background: #f9fafb; border-radius: 0 0 8px 8px; white-space: nowrap; flex-wrap: nowrap;",
          tags$span("Powered by"),
          tags$img(src = "https://www.gstatic.com/images/branding/googlelogo/1x/googlelogo_color_42x16dp.png", alt = "Google", height = "12"),
          tags$span("Translate")
        )
      )
    )
  ),
  includeCSS("www/styles.css"),
  includeCSS("www/mobile.css"),
  includeScript("www/script.js"),

  tags$script(HTML(
    "$(document).on('shiny:connected', function(){",
    "  $('.navbar-brand').css({cursor:'pointer', 'user-select':'none'}).on('click', function(e){",
    "    e.preventDefault();",
    "    $('a[data-value=\"Home\"]').tab('show');",
    "  });",
    "});"
  )),

  tags$a(
    href = "#main-content",
    class = "skip-link",
    "Skip to main content"
  ),

  tags$div(
    role = "status",
    `aria-live` = "polite",
    `aria-atomic` = "true",
    id = "status_region",
    class = "sr-only"
  ),

  tags$head(
    tags$title("TextAnalysisR")
  ),
  tags$main(
    id = "main-content",
    role = "main",
    tabindex = "-1",
    tags$h1("TextAnalysisR", class = "sr-only"),
    navbarPage(
      "TextAnalysisR",
      id = "main_navbar",
      collapsible = TRUE,
      tabPanel(
        "Home",
        fluidRow(
          column(
            width = 3,
            wellPanel(
              style = "padding: 0; border: none; box-shadow: none; background: transparent;",
              tags$ul(
                class = "nav nav-pills nav-stacked",
                id = "home_nav_menu",
                tags$li(tags$a(href = "#about-tab", "data-toggle" = "tab", "About")),
                tags$li(tags$a(href = "#semantic-tab", "data-toggle" = "tab", "Semantic Analysis")),
                tags$li(tags$a(href = "#lexicon-tab", "data-toggle" = "tab", "Sentiment Lexicons")),
                tags$li(tags$a(href = "#cyber-tab", "data-toggle" = "tab", "Cybersecurity")),
                tags$li(tags$a(href = "#access-tab", "data-toggle" = "tab", "Web Accessibility")),
                tags$li(tags$a(href = "#resources-tab", "data-toggle" = "tab", "Resources")),
                tags$li(tags$a(href = "#support-tab", "data-toggle" = "tab", "Support"))
              )
            )
          ),
          column(
            width = 9,
            tags$div(
              class = "tab-content",
              tags$div(id = "about-tab", class = "tab-pane", div(id = "about-content", class = "markdown-content", uiOutput("about_content"))),
              tags$div(id = "semantic-tab", class = "tab-pane", div(id = "installation-semantic-content", class = "markdown-content", uiOutput("installation_semantic_content"))),
              tags$div(id = "lexicon-tab", class = "tab-pane", div(id = "installation-lexical-content", class = "markdown-content", uiOutput("installation_lexical_content"))),
              tags$div(id = "cyber-tab", class = "tab-pane", div(id = "cybersecurity-content", class = "markdown-content", uiOutput("cybersecurity_content"))),
              tags$div(id = "access-tab", class = "tab-pane", div(id = "web-accessibility-content", class = "markdown-content", uiOutput("web_accessibility_content"))),
              tags$div(id = "resources-tab", class = "tab-pane", div(id = "links-content", class = "markdown-content", uiOutput("links_content"))),
              tags$div(id = "support-tab", class = "tab-pane", div(id = "support-content", class = "markdown-content", uiOutput("support_content")))
            )
          )
        )
      ),
    tabPanel(
      "AI Setup",
      sidebarLayout(
        sidebarPanel(
          width = 3,
          class = "sidebar-panel",
          tags$h5(
            HTML("<strong>AI Configuration</strong>"),
            tags$span("OPTIONAL", style = "background-color: #6c757d; color: white; padding: 2px 8px; border-radius: 3px; font-size: 13px; margin-left: 8px;"),
            style = "color: #4269BF; margin-bottom: 10px;"
          ),
          tags$p(style = "font-size: 16px; color: #666;", "API keys entered here apply to all AI features. You can also enter keys per-feature."),
          .password_input("global_openai_api_key", "OpenAI API Key:", placeholder = "sk-..."),
          .password_input("global_gemini_api_key", "Gemini API Key:", placeholder = "AIza..."),
          tags$hr(),
          tags$h5(
            HTML("<strong>Usage Log</strong>"),
            style = "color: #4269BF; margin-bottom: 10px;"
          ),
          tags$p(style = "font-size: 16px; color: #666;", "Track which AI models were used (for reproducibility reporting)."),
          downloadButton("download_ai_log", "Download Log as CSV", class = "btn-secondary btn-block")
        ),
        mainPanel(
          width = 9,
          conditionalPanel(
            condition = "output.has_ai_usage_log",
            DT::DTOutput("ai_usage_log_table")
          ),
          conditionalPanel(
            condition = "!output.has_ai_usage_log",
            div(
              style = "padding: 30px 20px;",
              tags$h5(HTML("<strong>AI Features Overview</strong>"), style = "color: #4269BF; margin-bottom: 16px;"),
              tags$p(style = "font-size: 16px; color: #475569; margin-bottom: 20px;",
                "Configure API keys in the sidebar, then use AI features across the app. Usage is logged here for reproducibility."),
              tags$table(
                class = "table table-bordered",
                style = "font-size: 16px;",
                tags$thead(
                  tags$tr(
                    tags$th("Feature", style = "width: 30%;"),
                    tags$th("Providers"),
                    tags$th("Tab", style = "width: 22%;")
                  )
                ),
                tags$tbody(
                  tags$tr(
                    tags$td("Document Similarity"),
                    tags$td("Sentence Transformers, OpenAI, Gemini"),
                    tags$td("Semantic Analysis")
                  ),
                  tags$tr(
                    tags$td("Semantic Search"),
                    tags$td("Sentence Transformers, OpenAI, Gemini"),
                    tags$td("Semantic Analysis")
                  ),
                  tags$tr(
                    tags$td("RAG Q&A"),
                    tags$td("OpenAI, Gemini"),
                    tags$td("Semantic Analysis")
                  ),
                  tags$tr(
                    tags$td("LLM Sentiment"),
                    tags$td("OpenAI, Gemini"),
                    tags$td("Semantic Analysis")
                  ),
                  tags$tr(
                    tags$td("Topic Modeling Embeddings"),
                    tags$td("Sentence Transformers, OpenAI, Gemini"),
                    tags$td("Topic Modeling")
                  ),
                  tags$tr(
                    tags$td("Topic Labels & Content"),
                    tags$td("OpenAI, Gemini"),
                    tags$td("Topic Modeling")
                  ),
                  tags$tr(
                    tags$td("Vision OCR"),
                    tags$td("OpenAI, Gemini"),
                    tags$td("Upload")
                  )
                )
              ),
              tags$div(
                style = "margin-top: 20px; padding: 12px 16px; background-color: #F1F5F9; border-radius: 6px;",
                tags$p(style = "margin: 0 0 6px 0; font-size: 16px; color: #5C6E88;",
                  tags$strong("Sentence Transformers"), " \u2014 Free, runs locally. Requires Python + sentence-transformers."),
                tags$p(style = "margin: 0 0 6px 0; font-size: 16px; color: #5C6E88;",
                  tags$strong("OpenAI"), " \u2014 Cloud API. Enter key above or set ", tags$code("OPENAI_API_KEY"), " in .Renviron."),
                tags$p(style = "margin: 0; font-size: 16px; color: #5C6E88;",
                  tags$strong("Gemini"), " \u2014 Cloud API. Enter key above or set ", tags$code("GEMINI_API_KEY"), " in .Renviron.")
              )
            )
          )
        )
      )
    ),
    tabPanel(
      "Upload",
      sidebarLayout(
        sidebarPanel(
          width = 3,
          class = "sidebar-panel",
          selectizeInput(
            "dataset_choice",
            "Dataset",
            choices = c(
              "Select a dataset" = "",
              "Upload an Example Dataset",
              "Upload Your File",
              "Copy and Paste Text"
            ),
            selected = "",
            options = list(placeholder = "Select dataset", loadThrottle = 0)
          ),
          fileInput("file", "File upload",
            multiple = FALSE,
            accept = c(".xlsx", ".xls", ".xlsm", ".csv", ".pdf", ".docx", ".txt")
          ),
          conditionalPanel(
            condition = "input.dataset_choice == 'Upload Your File'",
            uiOutput("pdf_status_indicator")
          ),
          conditionalPanel(
            condition = "input.dataset_choice == 'Upload Your File'",
            uiOutput("multimodal_options_ui")
          ),
          conditionalPanel(
            condition = "input.dataset_choice == 'Copy and Paste Text'",
            tags$div(
              class = "text-input-white-placeholder",
              textAreaInput("text_input", "Text input", "",
                rows = 10, placeholder = "Paste your text here...

Supports:
• Plain text
• Tabular data (Excel, CSV, web tables)"
              )
            )
          ),
          tags$div(
            class = "limits-info-box status-step-purple",
            style = "margin-top: 0;",
            tags$i(class = "fa fa-info-circle limits-icon status-icon status-icon-purple"),
            tags$strong("Limits:", class = "limits-title"), " Max 100MB file upload, 50MB paste. Optimal: 1K-5K documents"
          )
        ),
        mainPanel(
          width = 9,
          DT::dataTableOutput("data_table")
        )
      )
    ),
    tabPanel(
      "Preprocess",
      sidebarLayout(
        sidebarPanel(
          width = 3,
          class = "sidebar-panel",
          conditionalPanel(
            condition = "input.conditioned == 1",
            tags$h5(
              HTML("<strong>Select columns</strong> <a href='https://tidyr.tidyverse.org/reference/unite.html' target='_blank' rel='noopener noreferrer' onclick='window.open(this.href); return false;' style='font-size: 16px;'>Source</a>"),
              tags$span("REQUIRED", style = "background-color: #dc3545; color: white; padding: 2px 8px; border-radius: 3px; font-size: 13px; margin-left: 8px;"),
              style = "color: #4269BF; margin-bottom: 10px;"
            ),
            div(class = "checkbox-margin", checkboxGroupInput("show_vars",
              label = NULL,
              choices = NULL
            )),
            actionButton("apply", "Apply", class = "btn-primary btn-block")
          ),
          conditionalPanel(
            condition = "input.conditioned == 2",
            tags$h5(
              HTML("<strong>Segment corpus into tokens</strong> <a href='https://quanteda.io/reference/tokens.html' target='_blank' rel='noopener noreferrer' onclick='window.open(this.href); return false;' style='font-size: 16px;'>Source</a>"),
              tags$span("OPTIONAL", style = "background-color: #6c757d; color: white; padding: 2px 8px; border-radius: 3px; font-size: 13px; margin-left: 8px;"),
              style = "color: #4269BF; margin-bottom: 10px;"
            ),
            tags$div(
              class = "warning-box status-sidebar-warning",
              style = "padding: 0 12px; margin-top: 0;",
              tags$div(
                style = "margin: 0; padding: 0;",
                checkboxInput("math_mode",
                  HTML("<strong>Math Mode:</strong> Keep numbers, symbols, and punctuation"),
                  value = FALSE)
              )
            ),
            div(class = "checkbox-margin", checkboxGroupInput("segment_options",
              label = NULL,
              choices = list(
                "Convert to lowercase" = "lowercase",
                "Remove punctuation" = "remove_punct",
                "Remove symbols" = "remove_symbols",
                "Remove numbers" = "remove_numbers",
                "Remove URLs" = "remove_url",
                "Remove separators" = "remove_separators",
                "Split hyphens" = "split_hyphens",
                "Split tags" = "split_tags",
                "Include document variables" = "include_docvars",
                "Keep acronyms" = "keep_acronyms",
                "Keep padding" = "padding"
              ),
              selected = c("lowercase", "remove_punct", "remove_symbols", "remove_numbers", "remove_url",
                          "remove_separators", "split_hyphens", "split_tags", "include_docvars")
            )),
            numericInput("min_char",
              label = "Minimum characters per token",
              value = 2,
              min = 1,
              max = 10,
              step = 1),
            div(
              style = "display: flex; gap: 10px; margin-bottom: 15px;",
              div(
                style = "flex: 1;",
                actionButton("preprocess", "Apply", class = "btn-primary btn-block")
              ),
              div(
                style = "flex: 1;",
                actionButton("skip_segment", "Skip", class = "btn-secondary btn-block")
              )
            )
          ),
          conditionalPanel(
            condition = "input.conditioned == 4",
            tags$h5(
              HTML("<strong>Detect multi-words</strong> <a href='https://www.tidytextmining.com/ngrams' target='_blank' rel='noopener noreferrer' onclick='window.open(this.href); return false;' style='font-size: 16px;'>Source</a>"),
              tags$span("OPTIONAL", style = "background-color: #6c757d; color: white; padding: 2px 8px; border-radius: 3px; font-size: 13px; margin-left: 8px;"),
              style = "color: #4269BF; margin-bottom: 10px;"
            ),
            checkboxGroupInput(
              "ngram_sizes",
              "N-gram sizes",
              choices = list(
                "Bigrams (2 words)" = "2",
                "Trigrams (3 words)" = "3",
                "4-grams (4 words)" = "4",
                "5-grams (5 words)" = "5"
              ),
              selected = c("2", "3")
            ),
            sliderInput(
              "ngram_min_count",
              "Minimum frequency",
              min = 2,
              max = 20,
              value = 3,
              step = 1
            ),
            sliderInput(
              "ngram_min_lambda",
              "Minimum lambda (collocation strength)",
              min = 0,
              max = 10,
              value = 3,
              step = 0.5
            ),
            div(
              style = "display: flex; gap: 10px; margin-bottom: 15px;",
              div(
                style = "flex: 1;",
                actionButton("detect_ngrams", "Apply", class = "btn-primary btn-block")
              ),
              div(
                style = "flex: 1;",
                actionButton("skip_ngram_detection", "Skip", class = "btn-secondary btn-block")
              )
            ),
            tags$hr(),
            tags$h5(strong("Compound selected n-grams"), style = "color: #4269BF; margin-bottom: 10px;"),
            selectizeInput(
              "multi_word_expressions",
              label = "N-grams to compound",
              choices = NULL,
              multiple = TRUE,
              options = list(
                create = TRUE,
                placeholder = "Select from detected n-grams"
              )
            ),
            div(class = "checkbox-margin", checkboxGroupInput("stopword_options",
              label = "Clean stopwords from compound edges",
              choices = list(
                "Remove leading stopwords" = "leading_stopwords",
                "Remove trailing stopwords" = "trailing_stopwords"
              ),
              selected = c("leading_stopwords", "trailing_stopwords")
            )),
            div(
              style = "display: flex; gap: 10px; margin-bottom: 15px;",
              div(
                style = "flex: 1;",
                actionButton("dictionary", "Apply", class = "btn-primary btn-block")
              ),
              div(
                style = "flex: 1;",
                actionButton("skip_dictionary", "Skip", class = "btn-secondary btn-block")
              )
            )
          ),
          conditionalPanel(
            condition = "input.conditioned == 5",
            tags$div(
              style = "display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;",
              tags$h5(
                HTML("<strong>Document-feature matrix</strong> <a href='https://quanteda.io/reference/dfm.html' target='_blank' rel='noopener noreferrer' onclick='window.open(this.href); return false;' style='font-size: 16px;'>Source</a>"),
                tags$span("REQUIRED", style = "background-color: #dc3545; color: white; padding: 2px 8px; border-radius: 3px; font-size: 13px; margin-left: 8px;"),
                style = "color: #4269BF; margin: 0;"
              ),
              actionLink("showDFMInfo", tags$i(class = "fas fa-info-circle"),
                        style = "color: #337ab7; font-size: 16px;",
                        title = "Click for DFM guide")
            ),
            div(
              style = "display: flex; gap: 10px; margin-bottom: 15px;",
              div(
                style = "flex: 1;",
                actionButton("dfm_btn", "Process", class = "btn-primary btn-block")
              ),
              div(
                style = "flex: 1;",
                downloadButton("download_preprocessing_report", "Report", class = "btn-secondary btn-block")
              )
            )
          ),
          conditionalPanel(
            condition = "input.conditioned == 3",
            tags$h5(
              strong("Remove stopwords"),
              tags$span("OPTIONAL", style = "background-color: #6c757d; color: white; padding: 2px 8px; border-radius: 3px; font-size: 13px; margin-left: 8px;"),
              style = "color: #4269BF; margin-bottom: 10px;"
            ),
            selectizeInput(
              "common_words",
              label = "Top 10 frequent words pre-selected",
              choices = NULL,
              multiple = TRUE,
              options = list(
                create = TRUE,
                placeholder = "Type to add more or modify"
              )
            ),
            div(
              class = "stopwords-container",
              style = "max-height: 400px; overflow-y: auto;",
              selectizeInput(
                "custom_stopwords",
                label = HTML("<strong>Predefined stopwords</strong> <a href='https://search.r-project.org/CRAN/refmans/stopwords/html/stopwords.html' target='_blank' rel='noopener noreferrer' onclick='window.open(this.href); return false;' style='font-size: 16px;'>Source</a>"),
                choices = stopwords::stopwords("en", source = "snowball"),
                selected = stopwords::stopwords("en", source = "snowball"),
                multiple = TRUE,
                options = list(create = TRUE)
              )
            ),
            br(),
            div(
              style = "display: flex; gap: 10px; margin-bottom: 15px;",
              div(
                style = "flex: 1;",
                actionButton("remove", "Apply", class = "btn-primary btn-block")
              ),
              div(
                style = "flex: 1;",
                actionButton("skip_stopwords", "Skip", class = "btn-secondary btn-block")
              )
            )
          )
        ),
        mainPanel(
          width = 9,
          tabsetPanel(
            id = "conditioned",
            tabPanel(
              "1. Unite Texts",
              value = 1,
              conditionalPanel(
                condition = "output.has_united_table_results",
                br(),
                DT::dataTableOutput("united_table")
              ),
              conditionalPanel(
                condition = "!output.has_united_table_results",
                div(
                  style = "padding: 60px 40px; text-align: center;",
                  div(
                    style = "max-width: 400px; margin: 0 auto;",
                    tags$i(class = "fa fa-columns", style = "font-size: 48px; color: #CBD5E1; margin-bottom: 20px; display: block;", "aria-hidden" = "true"),
                    tags$p(
                      "Select columns and click ",
                      tags$strong("'Apply'", style = "color: #4269BF;"),
                      " to unite text columns",
                      style = "font-size: 18px; font-weight: 400; line-height: 1.7; color: #475569; margin: 0;"
                    )
                  )
                )
              )
            ),
            tabPanel(
              "2. Segment Texts",
              value = 2,
              conditionalPanel(
                condition = "output.has_preprocess_results",
                shiny::verbatimTextOutput("dict_print_preprocess")
              ),
              conditionalPanel(
                condition = "!output.has_preprocess_results",
                div(
                  style = "padding: 60px 40px; text-align: center;",
                  div(
                    style = "max-width: 400px; margin: 0 auto;",
                    tags$i(class = "fa fa-cut", style = "font-size: 48px; color: #CBD5E1; margin-bottom: 20px; display: block;", "aria-hidden" = "true"),
                    tags$p(
                      "Configure options and click ",
                      tags$strong("'Apply'", style = "color: #4269BF;"),
                      " to tokenize texts",
                      style = "font-size: 18px; font-weight: 400; line-height: 1.7; color: #475569; margin: 0;"
                    )
                  )
                )
              )
            ),
            tabPanel(
              "3. Remove Stopwords",
              value = 3,
              conditionalPanel(
                condition = "output.step_3_outdated == true",
                div(
                  class = "status-outdated",
                  tags$i(class = "fa fa-exclamation-triangle status-icon-lg status-icon-warning"),
                  tags$span(
                    tags$strong("Outdated Results: "),
                    "Based on previous Step 2 settings. Click ",
                    tags$strong("Apply"),
                    " to update with latest Step 2 output."
                  )
                )
              ),
              conditionalPanel(
                condition = "output.has_stopword_results",
                br(),
                div(
                  style = "margin-bottom: 20px; overflow: visible;",
                  plotly::plotlyOutput("stopword_plot", height = 500, width = "100%")
                ),
                br(),
                DT::dataTableOutput("stopword_table")
              ),
              conditionalPanel(
                condition = "!output.has_stopword_results",
                div(
                  style = "padding: 60px 40px; text-align: center;",
                  div(
                    style = "max-width: 400px; margin: 0 auto;",
                    tags$i(class = "fa fa-filter", style = "font-size: 48px; color: #CBD5E1; margin-bottom: 20px; display: block;", "aria-hidden" = "true"),
                    tags$p(
                      "Select stopwords and click ",
                      tags$strong("'Apply'", style = "color: #4269BF;"),
                      " to remove common words",
                      style = "font-size: 18px; font-weight: 400; line-height: 1.7; color: #475569; margin: 0;"
                    )
                  )
                )
              )
            ),
            tabPanel(
              "4. Multi-Word Dictionary",
              value = 4,
              br(),
              tabsetPanel(
                id = "multiword_subtabs",
                tabPanel(
                  "Detected N-grams",
                  conditionalPanel(
                    condition = "output.has_ngram_detection_results",
                    br(),
                    uiOutput("ngram_detection_plot_uiOutput"),
                    br(),
                    DT::dataTableOutput("ngram_detection_table")
                  ),
                  conditionalPanel(
                    condition = "!output.has_ngram_detection_results",
                    div(
                      style = "padding: 60px 40px; text-align: center;",
                      div(
                        style = "max-width: 400px; margin: 0 auto;",
                        tags$i(class = "fa fa-search", style = "font-size: 48px; color: #CBD5E1; margin-bottom: 20px; display: block;", "aria-hidden" = "true"),
                        tags$p(
                          "Configure settings and click ",
                          tags$strong("'Apply'", style = "color: #4269BF;"),
                          " to detect multi-words",
                          style = "font-size: 18px; font-weight: 400; line-height: 1.7; color: #475569; margin: 0;"
                        )
                      )
                    )
                  )
                ),
                tabPanel(
                  "Construct Multi-Words",
                  conditionalPanel(
                    condition = "output.step_4_outdated == true",
                    div(
                      class = "status-outdated",
                      tags$i(class = "fa fa-exclamation-triangle status-icon-lg status-icon-warning"),
                      tags$span(
                        tags$strong("Outdated Results: "),
                        "Based on previous Step 3 settings. Click ",
                        tags$strong("Apply"),
                        " to update with latest Step 3 output."
                      )
                    )
                  ),
                  conditionalPanel(
                    condition = "output.has_dictionary_results",
                    br(),
                    uiOutput("selected_ngrams_plot_uiOutput"),
                    br(),
                    DT::dataTableOutput("dictionary_table")
                  ),
                  conditionalPanel(
                    condition = "!output.has_dictionary_results",
                    div(
                      style = "padding: 60px 40px; text-align: center;",
                      div(
                        style = "max-width: 400px; margin: 0 auto;",
                        tags$i(class = "fa fa-link", style = "font-size: 48px; color: #CBD5E1; margin-bottom: 20px; display: block;", "aria-hidden" = "true"),
                        tags$p(
                          "Select n-grams and click ",
                          tags$strong("'Apply'", style = "color: #4269BF;"),
                          " to compound multi-words",
                          style = "font-size: 18px; font-weight: 400; line-height: 1.7; color: #475569; margin: 0;"
                        )
                      )
                    )
                  )
                )
              )
            ),
            tabPanel(
              "5. Document-Feature Matrix",
              value = 5,
              conditionalPanel(
                condition = "output.step_5_outdated == true",
                div(
                  class = "status-outdated",
                  tags$i(class = "fa fa-exclamation-triangle status-icon-lg status-icon-warning"),
                  tags$span(
                    tags$strong("Outdated Results: "),
                    "Based on previous Step 4 settings. Click ",
                    tags$strong("Process"),
                    " to update with latest Step 4 output."
                  )
                )
              ),
              conditionalPanel(
                condition = "output.has_dfm_results",
                br(),
                div(
                  style = "margin-bottom: 20px; overflow: visible;",
                  plotly::plotlyOutput("dfm_plot", height = 500, width = "100%")
                ),
                br(),
                DT::dataTableOutput("dfm_table")
              ),
              conditionalPanel(
                condition = "!output.has_dfm_results",
                div(
                  style = "padding: 60px 40px; text-align: center;",
                  div(
                    style = "max-width: 400px; margin: 0 auto;",
                    tags$i(class = "fa fa-table", style = "font-size: 48px; color: #CBD5E1; margin-bottom: 20px; display: block;", "aria-hidden" = "true"),
                    tags$p(
                      "Click ",
                      tags$strong("'Process'", style = "color: #4269BF;"),
                      " to create document-feature matrix",
                      style = "font-size: 18px; font-weight: 400; line-height: 1.7; color: #475569; margin: 0;"
                    )
                  )
                )
              )
            )
          )
        )
      )
    ),
    tabPanel(
      "Lexical Analysis",
      uiOutput("lexical_analysis_ui")
    ),
    tabPanel(
      "Semantic Analysis",
      uiOutput("semantic_analysis_ui")
    ),
    tabPanel(
      "Topic Modeling",
      uiOutput("topic_modeling_ui")
    )
  )
  )
)
