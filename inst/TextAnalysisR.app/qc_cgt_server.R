# Qualitative coding: project file, group ruling, annotation, blind holdout
# Sourced inside the server function, so it shares the session's reactives.

qc_code_palette <- c("#378ADD", "#1D9E75", "#EF9F27", "#7F77DD",
                     "#D85A30", "#D4537E", "#639922", "#888780")
qc_fallback_color <- "#888780"

session$allowReconnect(TRUE)

qc_project <- reactiveVal(TextAnalysisR::new_coding_project("coder1"))
qc_coder <- reactiveVal("coder1")
qc_save_status <- reactiveVal("Not saved yet")
qc_ann_index <- reactiveVal(1L)
qc_autosave_ok <- reactiveVal(FALSE)
qc_saved_this_session <- reactiveVal(FALSE)
qc_pending_file <- reactiveVal(NULL)

qc_rulings_rv <- reactiveVal(isolate(qc_project())$rulings)
qc_holdout_rv <- reactiveVal(isolate(qc_project())$holdout)
qc_detection_rv <- reactiveVal(NULL)
qc_colors_rv <- reactiveVal(character(0))

qc_set_project <- function(p) {
  qc_project(p)
  qc_rulings_rv(p$rulings)
  qc_holdout_rv(p$holdout)
  qc_detection_rv(p$detection)
  cb <- p$codebook[!is.na(p$codebook$code) & nzchar(p$codebook$code), , drop = FALSE]
  qc_colors_rv(stats::setNames(cb$color, cb$code))
}

qc_edit_project <- function(fn) qc_set_project(fn(isolate(qc_project())))

qc_now <- function() Sys.time()

qc_color <- function(colors, code) {
  col <- unname(colors[code])
  ifelse(is.na(col), qc_fallback_color, col)
}

qc_doc_id <- function(unit_id) sub("\\.[0-9]+$", "", unit_id)

# project file, autosave, and coder name

qc_project_dir <- getOption("TextAnalysisR.project_dir", tools::R_user_dir("TextAnalysisR", "data"))

qc_path_for <- function(coder) {
  file.path(qc_project_dir, paste0("coding-project-", gsub("[^A-Za-z0-9_-]", "_", coder), ".rds"))
}

qc_autosave_path <- reactive(qc_path_for(qc_coder()))

qc_has_work <- function(p) {
  nrow(p$rulings) + nrow(p$assignments) + nrow(p$memos) + nrow(p$holdout) > 0
}

qc_check_saved_file <- function(coder) {
  path <- qc_path_for(coder)
  # projects autosaved before the R_user_dir default sit in the working directory; opening one moves later saves
  legacy <- file.path(getwd(), basename(path))
  if (!file.exists(path) && file.exists(legacy)) path <- legacy
  if (is_remote || !file.exists(path)) {
    qc_pending_file(NULL)
    qc_autosave_ok(TRUE)
    return(invisible(FALSE))
  }
  qc_autosave_ok(FALSE)
  qc_pending_file(path)
  invisible(TRUE)
}

isolate(qc_check_saved_file("coder1"))

qc_coder_typed <- debounce(reactive(trimws(input$qc_coder_name %||% "")), 800)

observeEvent(qc_coder_typed(), {
  new <- qc_coder_typed()
  old <- isolate(qc_coder())
  if (!nzchar(new) || identical(new, old)) return()
  qc_coder(new)
  qc_edit_project(function(p) {
    p$coder <- new
    p$assignments$coder[p$assignments$coder == old] <- new
    p$rulings$coder[p$rulings$coder == old] <- new
    p$units_read$coder[p$units_read$coder == old] <- new
    p
  })
  qc_saved_this_session(FALSE)
  qc_check_saved_file(new)
})

output$qc_file_banner <- renderUI({
  path <- qc_pending_file()
  if (is.null(path)) return(NULL)
  saved <- tryCatch(format(file.info(path)$mtime, "%Y-%m-%d %H:%M"), error = function(e) "earlier")
  div(class = "qc-restore-banner", role = "status",
      tags$i(class = "fa fa-folder-open", `aria-hidden` = "true"),
      tags$span(class = "qc-msg-text", sprintf("Saved project for %s found (%s). Autosave paused.", qc_coder(), saved)),
      actionButton("qc_open_saved", "Open", class = "btn-default btn-sm"),
      actionButton("qc_start_new", "Start new", class = "btn-link btn-sm"))
})

observeEvent(input$qc_open_saved, {
  path <- qc_pending_file()
  req(path)
  p <- tryCatch(TextAnalysisR::read_coding_project(path), error = function(e) {
    showNotification(paste("Could not open the saved project:", conditionMessage(e)), type = "error", duration = 10)
    NULL
  })
  if (is.null(p)) return()
  qc_load_project(p)
  qc_pending_file(NULL)
  qc_autosave_ok(TRUE)
  showNotification(sprintf("Opened the saved project of %s.", p$coder), type = "message", duration = 6)
})

observeEvent(input$qc_start_new, {
  path <- qc_pending_file()
  req(path)
  kept <- sub("\\.rds$", paste0("-", format(qc_now(), "%Y%m%d-%H%M%S"), ".rds"), path)
  moved <- file.rename(path, kept)
  qc_pending_file(NULL)
  qc_autosave_ok(TRUE)
  showNotification(if (moved) paste("The earlier project was kept as", basename(kept))
                   else "The earlier project could not be renamed; it stays as is.",
                   type = if (moved) "message" else "warning", duration = 8)
})

qc_project_debounced <- debounce(reactive(qc_project()), 1500)

qc_device_payload <- function(p) {
  keep <- function(d) as.list(as.data.frame(d, stringsAsFactors = FALSE))
  list(coder = p$coder, saved = format(qc_now(), "%Y-%m-%dT%H:%M:%S", tz = "UTC"),
       codebook = keep(p$codebook[, setdiff(names(p$codebook), "example")]),
       assignments = keep(p$assignments[, setdiff(names(p$assignments), "rationale")]),
       memos = keep(p$memos[, setdiff(names(p$memos), "text")]),
       rulings = keep(p$rulings),
       units_read = keep(p$units_read), holdout = keep(p$holdout),
       detection = p$detection)
}

observeEvent(qc_project_debounced(), {
  p <- qc_project_debounced()
  if (!qc_has_work(p) && nrow(p$codebook) == 0) return()
  session$sendCustomMessage("qcDeviceSave", qc_device_payload(p))
  if (is_remote) {
    qc_save_status(paste("Kept on this device", format(qc_now(), "%H:%M:%S"), "· download to keep a copy"))
    return()
  }
  if (!qc_autosave_ok()) {
    qc_save_status("Autosave paused · choose Open or Start new")
    return()
  }
  path <- qc_autosave_path()
  first <- !qc_saved_this_session()
  ok <- tryCatch({
    dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
    TextAnalysisR::save_coding_project(p, path, backup = first)
    TRUE
  }, error = function(e) FALSE)
  if (ok) qc_saved_this_session(TRUE)
  qc_save_status(if (ok) paste("Saved", format(qc_now(), "%H:%M:%S"), "to", basename(path))
                 else "Autosave failed · download the project")
}, ignoreInit = TRUE)

output$qc_save_status <- renderText(qc_save_status())

output$qc_autosave_note <- renderText({
  if (is_remote) "Work is kept on this device; download the project file to keep it."
  else paste0("Autosaves to ", basename(qc_autosave_path()), " in ", basename(dirname(qc_autosave_path())))
})

output$qc_project_download <- downloadHandler(
  filename = function() basename(qc_autosave_path()),
  content = function(file) TextAnalysisR::save_coding_project(qc_project(), file, backup = FALSE)
)

qc_load_project <- function(p) {
  qc_set_project(p)
  qc_coder(p$coder)
  updateTextInput(session, "qc_coder_name", value = p$coder)
  if (nrow(p$codebook) > 0) qc_codebook(p$codebook)
}

observeEvent(input$qc_project_file, {
  p <- tryCatch(TextAnalysisR::read_coding_project(input$qc_project_file$datapath),
                error = function(e) {
                  showNotification(paste("Could not open the project:", e$message), type = "error", duration = 10)
                  NULL
                })
  if (is.null(p)) return()
  qc_load_project(p)
  qc_pending_file(NULL)
  qc_autosave_ok(TRUE)
  showNotification(sprintf("Opened the project of %s.", p$coder), type = "message", duration = 6)
})

# device copy for a dropped connection; holds no response text or memo text

qc_device_copy <- reactiveVal(NULL)

observeEvent(input$qc_device_copy, {
  copy <- input$qc_device_copy
  if (is.null(copy) || qc_has_work(qc_project())) return()
  qc_device_copy(copy)
}, once = TRUE)

output$qc_restore_banner <- renderUI({
  copy <- qc_device_copy()
  if (is.null(copy)) return(NULL)
  n <- length(copy$rulings$group %||% list()) + length(copy$assignments$unit_id %||% list())
  if (n == 0) return(NULL)
  div(class = "qc-restore-banner", role = "status",
      tags$i(class = "fa fa-history", `aria-hidden` = "true"),
      tags$span(class = "qc-msg-text", sprintf("Unsaved work from %s UTC kept on this device (%d entries, no memo text).",
              sub("T", " ", copy$saved %||% "an earlier session"), n)),
      actionButton("qc_restore_device", "Restore", class = "btn-default btn-sm"),
      actionButton("qc_discard_device", "Discard", class = "btn-link btn-sm"))
})

qc_from_device <- function(copy) {
  p <- TextAnalysisR::new_coding_project(copy$coder %||% "coder1")
  to_tbl <- function(x, template) {
    if (is.null(x) || length(x) == 0) return(template)
    n <- max(lengths(x))
    if (n == 0) return(template)
    d <- tibble::as_tibble(lapply(x, function(col) {
      if (is.null(col)) return(rep(NA, n))
      unlist(lapply(as.list(col), function(v) if (is.null(v)) NA else v))
    }))
    for (nm in intersect(names(template), names(d))) {
      if (inherits(template[[nm]], "POSIXct")) {
        d[[nm]] <- as.POSIXct(sub("Z$", "", as.character(d[[nm]])), tz = "UTC",
                              tryFormats = c("%Y-%m-%dT%H:%M:%OS", "%Y-%m-%d %H:%M:%OS"))
      } else if (is.integer(template[[nm]])) d[[nm]] <- as.integer(d[[nm]])
      else if (is.numeric(template[[nm]])) d[[nm]] <- as.numeric(d[[nm]])
      else d[[nm]] <- as.character(d[[nm]])
    }
    dplyr::bind_rows(template, d)
  }
  for (tbl in c("codebook", "assignments", "memos", "rulings", "units_read", "holdout")) {
    p[[tbl]] <- to_tbl(copy[[tbl]], p[[tbl]])
  }
  if (!is.null(copy$detection$unit_ids)) {
    p$detection <- list(unit_ids = as.character(unlist(copy$detection$unit_ids)),
                        groups = as.character(unlist(copy$detection$groups)))
  }
  p
}

observeEvent(input$qc_restore_device, {
  qc_load_project(qc_from_device(qc_device_copy()))
  qc_device_copy(NULL)
  showNotification("Work restored from this device.", type = "message", duration = 6)
})

observeEvent(input$qc_discard_device, {
  qc_device_copy(NULL)
  session$sendCustomMessage("qcDeviceClear", list())
})

# codebook kept in the project, each code with a fixed color

qc_with_colors <- function(cb) {
  if (is.null(cb) || nrow(cb) == 0) return(cb)
  cb[] <- lapply(cb, as.character)
  if (!"example" %in% names(cb)) cb$example <- NA_character_
  if (!"color" %in% names(cb)) cb$color <- NA_character_
  if (!"theme" %in% names(cb)) cb$theme <- NA_character_
  cb$color[!grepl("^#[0-9A-Fa-f]{6}$", cb$color %||% "")] <- NA_character_
  free <- setdiff(qc_code_palette, cb$color)
  need <- which(is.na(cb$color))
  cb$color[need] <- rep_len(if (length(free)) free else qc_code_palette, length(need))
  cb[, c("code", "definition", "example", "color", "theme")]
}

observeEvent(qc_codebook(), {
  cb <- qc_with_colors(qc_codebook())
  if (is.null(cb)) return()
  qc_edit_project(function(p) {
    p$codebook <- tibble::as_tibble(cb)
    p
  })
}, ignoreNULL = TRUE)

qc_code_colors <- reactive(qc_colors_rv())

observeEvent(qc_suggestions(), {
  s <- qc_suggestions()
  if (is.null(s)) return()
  confirmed <- s[s$status %in% c("accepted", "edited"), , drop = FALSE]
  rows <- tibble::tibble(
    doc_id = as.character(confirmed$doc_id), unit_id = as.character(confirmed$unit_id),
    start = as.integer(confirmed$start), end = as.integer(confirmed$end),
    code = as.character(confirmed$code), coder = isolate(qc_coder()),
    confidence = as.numeric(confirmed$confidence), rationale = as.character(confirmed$rationale),
    status = "ai-confirmed", timestamp = qc_now(), stage = "annotate")
  covered <- unique(as.character(s$unit_id))
  qc_edit_project(function(p) {
    stale <- p$assignments$status == "ai-confirmed" & p$assignments$unit_id %in% covered
    p$assignments <- dplyr::bind_rows(p$assignments[!stale, ], rows)
    p
  })
})

# units, detected groups, and replayed rounds

qc_units <- reactive({
  model <- topic_model_result()
  emb <- model$embeddings %||% embeddings_cache$embeddings
  if (is.null(model) || is.null(model$topic_assignments) || is.null(emb)) return(NULL)
  texts <- residue_texts() %||% analysis_texts()
  if (length(texts) != nrow(emb) || length(texts) != length(model$topic_assignments)) return(NULL)
  ids <- names(texts) %||% paste0("u", seq_along(texts))
  clean <- gsub("\r\n?", "\n", unname(as.character(texts)))
  list(ids = as.character(ids), texts = stats::setNames(clean, ids),
       emb = emb, groups0 = as.character(model$topic_assignments))
})

qc_detection <- reactive({
  u <- qc_units()
  if (is.null(u)) return(list(groups = NULL, mismatch = FALSE))
  saved <- qc_detection_rv()
  if (!is.null(saved)) {
    if (identical(saved$unit_ids, u$ids)) return(list(groups = saved$groups, mismatch = FALSE))
    return(list(groups = u$groups0, mismatch = TRUE))
  }
  list(groups = u$groups0, mismatch = nrow(qc_rulings_rv()) > 0)
})

qc_replay <- reactive({
  u <- qc_units()
  d <- qc_detection()
  if (is.null(u) || is.null(d$groups)) return(list(groups = NULL, error = NULL))
  if (d$mismatch) return(list(groups = d$groups, error = "mismatch"))
  g <- d$groups
  r <- qc_rulings_rv()
  for (k in sort(unique(r$round))) {
    g <- tryCatch(suppressWarnings(TextAnalysisR::apply_ruling(g, r[r$round == k, ], u$emb)),
                  error = function(e) structure(conditionMessage(e), class = "qc_replay_error"))
    if (inherits(g, "qc_replay_error")) return(list(groups = d$groups, error = as.character(g)))
  }
  list(groups = g, error = NULL)
})

qc_groups <- reactive(qc_replay()$groups)

qc_round <- reactive({
  r <- qc_rulings_rv()
  if (nrow(r) == 0) 1L else max(r$round) + 1L
})

qc_settled <- reactive(TextAnalysisR::ruling_settled(qc_rulings_rv()))

# source lines in the sidebars

qc_tag <- function(label, kind) tags$span(class = paste("qc-tag", kind), label)
qc_src_row <- function(key, ...) div(class = "qc-route-item", tags$span(class = "qc-route-key", key), ...)

qc_codebook_line <- reactive({
  cb <- qc_codebook()
  src <- qc_codebook_source()
  if (is.null(cb) || nrow(cb) == 0) return(tagList(qc_tag("None", "qc-tag-plain")))
  tagList(lapply(if (length(src)) src else "Project file",
                 function(x) qc_tag(x, if (grepl("^Topic labels", x)) "qc-tag-topic" else "qc-tag-plain")),
          sprintf("%d codes", nrow(cb)))
})

output$qc_src_codebook <- renderUI(div(class = "qc-route", qc_src_row("Source", qc_codebook_line())))

output$qc_src_groups <- renderUI({
  u <- qc_units()
  div(class = "qc-route", qc_src_row("Groups",
    if (is.null(u)) tagList(qc_tag("None", "qc-tag-plain"), "no embedding model") else
      tagList(qc_tag("Embedding model", "qc-tag-topic"),
              sprintf("%d groups, round %d", length(unique(setdiff(qc_detection()$groups, "0"))), qc_round()))))
})

output$qc_src_coding <- renderUI({
  cb <- qc_codebook()
  s <- qc_suggestions()
  model <- switch(input$qc_provider %||% "", openai = input$qc_openai_model, gemini = input$qc_gemini_model, NULL)
  with_examples <- !is.null(cb) && "example" %in% names(cb) && any(!is.na(cb$example) & nzchar(cb$example))
  who <- paste(c(input$qc_provider, model)[nzchar(c(input$qc_provider, model))], collapse = " ")
  div(class = "qc-route",
      qc_src_row("Codebook", qc_codebook_line()),
      qc_src_row("Mode", qc_tag(if (with_examples) "AI with examples" else "AI zero-shot", "qc-tag-ai"),
                 if (nzchar(who)) who),
      qc_src_row("Groups", if (is.null(qc_units())) tagList(qc_tag("None", "qc-tag-plain"), "not used") else
        tagList(qc_tag("Embedding model", "qc-tag-topic"), "not used for coding")))
})

# stepper

output$qc_holdout_notice <- renderUI({
  p <- qc_project()
  if (is.null(qc_units()) || nrow(p$holdout) > 0) return(NULL)
  tags$p(class = "qc-tip",
    "Draw the blind holdout before ruling or annotating; units opened first cannot be held out. ",
    actionLink("qc_go_agree_first", "Draw holdout"))
})

observeEvent(input$qc_go_agree_first, if (input$qc_go_agree_first > 0) updateTabsetPanel(session, "qual_coding_tabs", selected = "qc_agree"), ignoreInit = TRUE)


# group ruling

output$has_qc_groups <- reactive(!is.null(qc_units()))
outputOptions(output, "has_qc_groups", suspendWhenHidden = FALSE)

qc_group_key <- function(g) paste0("g", g)
qc_rid <- function(prefix, round, g) paste0(prefix, round, "_", gsub("[^A-Za-z0-9]", "_", g))

qc_draft_label <- function(g) {
  labels <- tryCatch(isolate(generated_labels()), error = function(e) NULL)
  base <- sub("[a-z]+$", "", g)
  if (!is.null(labels) && "topic" %in% names(labels)) {
    hit <- labels$topic_label[as.character(labels$topic) == base]
    if (length(hit) && !is.na(hit[1]) && nzchar(hit[1])) return(as.character(hit[1]))
  }
  qc_group_key(g)
}

qc_evidence <- reactive({
  u <- qc_units()
  g <- qc_groups()
  req(u, g)
  if (length(unique(setdiff(g, "0"))) < 2) return(NULL)
  TextAnalysisR::group_evidence(u$emb, g, unit_ids = u$ids, hide = qc_holdout_rv()$unit_id,
                                seed = 2025L + qc_round())
})

qc_log_read <- function(ids, step) {
  if (length(ids) == 0) return()
  round <- isolate(qc_round())
  coder <- isolate(qc_coder())
  qc_edit_project(function(p) {
    p$units_read <- TextAnalysisR::log_units_read(p$units_read, ids, round = round, step = step, coder = coder)
    p
  })
}

observeEvent(list(qc_evidence(), input$qual_coding_tabs), {
  ev <- qc_evidence()
  if (is.null(ev) || !identical(input$qual_coding_tabs, "qc_ruling") || qc_settled()) return()
  if (!is.null(qc_replay()$error)) return()
  qc_log_read(unique(unlist(c(ev$random_units, ev$boundary_unit, ev$neighbor_units))), "ruling")
}, ignoreNULL = FALSE)

output$qc_ruling_status <- renderUI({
  ev <- qc_evidence()
  if (is.null(ev)) return(NULL)
  round <- qc_round()
  done <- sum(vapply(ev$group, function(g) !is.null(input[[qc_rid("qc_v_", round, g)]]), logical(1)))
  read <- length(unique(qc_project()$units_read$unit_id))
  tagList(
    div(class = "qc-round-head",
        tags$span(class = "qc-round-n", if (qc_settled()) "Settled" else paste("Round", round)),
        tags$span(sprintf("%d of %d groups ruled · %d units read", done, nrow(ev), read))),
    div(class = "qc-progress", div(style = sprintf("width:%d%%", round(100 * done / max(1, nrow(ev)))))))
})

output$qc_ruling_cards <- renderUI({
  replay <- qc_replay()
  if (!is.null(replay$error)) {
    return(div(class = "qc-empty", role = "alert",
               if (identical(replay$error, "mismatch"))
                 "This project's rounds were ruled on a different topic model run. Re-run that model, or start a new project, before ruling further."
               else paste("The saved rounds could not be replayed:", replay$error)))
  }
  ev <- qc_evidence()
  u <- qc_units()
  if (is.null(ev)) return(div(class = "qc-empty", "At least two groups are needed to rule."))
  if (qc_settled()) {
    return(div(class = "qc-empty",
               tags$i(class = "fa fa-check-circle", `aria-hidden` = "true"),
               " Refinement settled: the last round kept every group. The codebook holds the final labels."))
  }
  round <- qc_round()
  quote_list <- function(ids) tags$ul(class = "qc-units", lapply(ids, function(i) tags$li(u$texts[[i]])))
  lapply(seq_len(nrow(ev)), function(j) {
    g <- ev$group[j]
    others <- setdiff(ev$group, g)
    hint <- if (ev$separation[j] > ev$coherence[j]) {
      tags$p(class = "qc-hint", sprintf("Separation (%.2f) exceeds coherence (%.2f): check the closest units for a merge.",
                                        ev$separation[j], ev$coherence[j]))
    }
    div(class = "qc-card",
        div(class = "qc-card-head",
            tags$strong(sprintf("%s · n = %d", qc_group_key(g), ev$n[j])),
            tags$span(sprintf("coherence %.2f · nearest %s · separation %.2f",
                              ev$coherence[j], qc_group_key(ev$nearest[j]), ev$separation[j]))),
        hint,
        div(class = "qc-card-body",
            div(class = "qc-evidence",
                tags$h6("Three random units: one idea?"), quote_list(ev$random_units[[j]]),
                tags$h6("Furthest member: does the definition cover it?"), quote_list(ev$boundary_unit[[j]]),
                tags$h6(sprintf("Closest units from %s: do they belong here?", qc_group_key(ev$nearest[j]))),
                quote_list(ev$neighbor_units[[j]]),
                tags$button(type = "button", class = "btn btn-link qc-more", `data-group` = g,
                            "Read further into this group")),
            div(class = "qc-decision",
                radioButtons(qc_rid("qc_v_", round, g), "Verdict",
                             choices = c("Keep" = "keep", "Split" = "split", "Merge" = "merge"),
                             selected = character(0), inline = TRUE),
                conditionalPanel(sprintf("input['%s'] == 'merge'", qc_rid("qc_v_", round, g)),
                                 selectInput(qc_rid("qc_m_", round, g), "Merge into",
                                             choices = stats::setNames(others, qc_group_key(others)))),
                textInput(qc_rid("qc_l_", round, g), "Final label (name the act)", value = qc_draft_label(g)),
                textAreaInput(qc_rid("qc_d_", round, g), "Final definition", rows = 3,
                              placeholder = "A unit must state..."))))
  })
})

observeEvent(input$qc_more, {
  g <- as.character(input$qc_more)
  u <- qc_units()
  req(u)
  hidden <- qc_holdout_rv()$unit_id
  members <- u$ids[qc_groups() == g & !u$ids %in% hidden]
  qc_log_read(members, "ruling")
  showModal(modalDialog(
    title = sprintf("%s: all %d units%s", qc_group_key(g), length(members),
                    if (length(hidden)) " (holdout units hidden)" else ""),
    size = "l", easyClose = TRUE,
    tags$ol(class = "qc-units", lapply(members, function(i) tags$li(u$texts[[i]]))),
    footer = modalButton("Close")))
})

observeEvent(input$qc_close_round, {
  if (qc_settled()) {
    showNotification("Refinement has settled; there is no open round.", type = "message", duration = 6)
    return()
  }
  if (!is.null(qc_replay()$error)) return()
  ev <- qc_evidence()
  if (is.null(ev)) return()
  round <- qc_round()
  value <- function(prefix, g, default) input[[qc_rid(prefix, round, g)]] %||% default
  verdicts <- vapply(ev$group, function(g) value("qc_v_", g, ""), character(1))
  if (any(!nzchar(verdicts))) {
    showNotification(sprintf("%d group(s) still need a verdict.", sum(!nzchar(verdicts))), type = "warning", duration = 7)
    return()
  }
  labels <- vapply(ev$group, function(g) trimws(value("qc_l_", g, "")), character(1))
  settling <- all(verdicts == "keep")
  if (settling && (any(!nzchar(labels)) || anyDuplicated(labels))) {
    showNotification("Every kept group needs its own, non-empty label before the rounds can settle.",
                     type = "warning", duration = 8)
    return()
  }
  merge_to <- vapply(ev$group, function(g) value("qc_m_", g, NA_character_), character(1))
  rows <- tibble::tibble(
    round = round, group = ev$group, n = as.integer(ev$n), verdict = verdicts,
    merge_with = ifelse(verdicts == "merge", merge_to, NA_character_),
    label = labels,
    definition = vapply(ev$group, function(g) trimws(value("qc_d_", g, "")), character(1)),
    coder = qc_coder(), timestamp = qc_now(), stage = "refine")
  u <- qc_units()
  notes <- character(0)
  withCallingHandlers(
    TextAnalysisR::apply_ruling(qc_groups(), rows, u$emb),
    warning = function(w) {
      notes <<- c(notes, conditionMessage(w))
      invokeRestart("muffleWarning")
    })
  qc_edit_project(function(p) {
    if (is.null(p$detection)) p$detection <- list(unit_ids = u$ids, groups = u$groups0)
    p$rulings <- dplyr::bind_rows(p$rulings, rows)
    p
  })
  if (length(notes)) showNotification(paste(notes, collapse = " "), type = "warning", duration = 10)
  if (settling) {
    found <- tibble::tibble(code = rows$label, definition = rows$definition,
                            example = vapply(ev$random_units, function(ids) if (length(ids)) u$texts[[ids[1]]] else NA_character_,
                                             character(1)),
                            color = NA_character_)
    prior <- qc_codebook()
    kept <- if (is.null(prior) || nrow(prior) == 0) NULL else prior[!prior$code %in% found$code, , drop = FALSE]
    qc_codebook(qc_with_colors(dplyr::bind_rows(found, kept)))
    showNotification("Refinement settled: every group kept. The codebook now holds the final labels.",
                     type = "message", duration = 10)
  } else {
    showNotification(sprintf("Round %d closed. The next round shows the new groups.", round),
                     type = "message", duration = 7)
  }
})

output$qc_download_rulings <- downloadHandler(
  filename = function() paste0("ruling-log-", Sys.Date(), ".csv"),
  content = function(file) utils::write.csv(qc_project()$rulings, file, row.names = FALSE)
)

# annotation

qc_stage_of <- function(id) if (id %in% qc_holdout_rv()$unit_id) "confirm" else "annotate"

qc_ann_units <- reactive({
  u <- qc_units()
  src <- input$qc_ann_source %||% "loose"
  hidden <- qc_holdout_rv()$unit_id
  texts <- if (!is.null(u)) u$texts else {
    raw <- tryCatch(analysis_texts(), error = function(e) character(0))
    stats::setNames(gsub("\r\n?", "\n", unname(as.character(raw))), names(raw))
  }
  ids <- if (src == "holdout") hidden
         else if (is.null(u)) names(texts)
         else if (src == "loose") u$ids[qc_groups() == "0"]
         else u$ids[qc_groups() == sub("^group:", "", src)]
  if (src != "holdout") ids <- setdiff(ids, hidden)
  ids <- ids[ids %in% names(texts)]
  list(ids = ids, texts = texts)
})

observe({
  u <- qc_units()
  groups <- if (is.null(u)) character(0) else sort(unique(setdiff(qc_groups(), "0")))
  n_loose <- if (is.null(u)) length(tryCatch(analysis_texts(), error = function(e) NULL)) else sum(qc_groups() == "0")
  choices <- c(stats::setNames("loose", sprintf("Loose units (%d)", n_loose)), "Blind holdout" = "holdout",
               if (length(groups)) stats::setNames(paste0("group:", groups), paste("Group", qc_group_key(groups))))
  current <- isolate(input$qc_ann_source)
  fallback <- if (n_loose > 0 || length(groups) == 0) "loose" else paste0("group:", groups[1])
  keep <- !is.null(current) && current %in% choices && !(current == "loose" && n_loose == 0)
  updateSelectInput(session, "qc_ann_source", choices = choices,
                    selected = if (keep) current else fallback)
})

observeEvent(input$qc_ann_source, qc_ann_index(1L))
observeEvent(input$qc_ann_prev, qc_ann_index(max(1L, qc_ann_index() - 1L)))
observeEvent(input$qc_ann_next, qc_ann_index(min(length(qc_ann_units()$ids), qc_ann_index() + 1L)))

qc_ann_current <- reactive({
  a <- qc_ann_units()
  if (length(a$ids) == 0) return(NULL)
  i <- min(qc_ann_index(), length(a$ids))
  id <- a$ids[i]
  list(i = i, n = length(a$ids), id = id, text = a$texts[[id]])
})

observeEvent(qc_ann_current()$id, {
  cur <- qc_ann_current()
  qc_log_read(cur$id, if (identical(isolate(input$qc_ann_source), "holdout")) "holdout" else "annotate")
  m <- isolate(qc_project())$memos
  memo <- m$text[m$target_type == "unit" & m$target_id == cur$id]
  updateTextAreaInput(session, "qc_ann_memo", value = if (length(memo)) memo[length(memo)] else "")
})

qc_unit_spans <- function(p, id) {
  a <- p$assignments
  a[a$unit_id == id & a$coder == p$coder & a$status %in% c("human", "human-none"), , drop = FALSE]
}

qc_highlight <- function(text, spans, colors) {
  spans <- spans[!is.na(spans$code), , drop = FALSE]
  if (nrow(spans) == 0) return(htmltools::htmlEscape(text))
  cuts <- sort(unique(c(1L, nchar(text) + 1L, spans$start, spans$end + 1L)))
  cuts <- cuts[cuts >= 1 & cuts <= nchar(text) + 1]
  pieces <- vapply(seq_len(length(cuts) - 1), function(k) {
    a <- cuts[k]
    b <- cuts[k + 1] - 1L
    seg <- htmltools::htmlEscape(substr(text, a, b))
    on <- spans[spans$start <= a & spans$end >= b, , drop = FALSE]
    on <- on[!duplicated(on$code), , drop = FALSE]
    if (nrow(on) == 0) return(seg)
    cols <- qc_color(colors, on$code)
    stops <- seq(0, 100, length.out = length(cols) + 1)
    bands <- paste(vapply(seq_along(cols), function(i) {
      sprintf("color-mix(in srgb, %s 32%%, transparent) %.0f%% %.0f%%", cols[i], stops[i], stops[i + 1])
    }, character(1)), collapse = ", ")
    sprintf("<mark class=\"qc-mark%s\" style=\"--qc-color:%s; --qc-bg:linear-gradient(to bottom, %s)\" title=\"%s\">%s</mark>",
            if (nrow(on) == 1) "" else " qc-multi", cols[1], bands,
            htmltools::htmlEscape(paste(on$code, collapse = " + "), TRUE), seg)
  }, character(1))
  paste(pieces, collapse = "")
}

output$qc_ann_panel <- renderUI({
  cur <- qc_ann_current()
  if (is.null(cur)) {
    return(div(class = "qc-empty",
               if (length(qc_ann_units()$texts) == 0) "Upload and unite texts first (Upload, then Preprocess)."
               else if ((input$qc_ann_source %||% "loose") == "holdout") "Draw the blind holdout in the Confirm tab first."
               else "No units to show for this source."))
  }
  p <- qc_project()
  spans <- qc_unit_spans(p, cur$id)
  colors <- qc_code_colors()
  none <- any(spans$status == "human-none")
  coded <- spans[!is.na(spans$code), , drop = FALSE]
  tagList(
    div(class = "qc-ann-nav",
        tags$span(sprintf("Unit %d of %d · %s", cur$i, cur$n, cur$id)),
        div(actionButton("qc_ann_prev", NULL, icon = icon("arrow-left"), class = "btn-default btn-sm",
                         `aria-label` = "Previous unit"),
            actionButton("qc_ann_next", NULL, icon = icon("arrow-right"), class = "btn-default btn-sm",
                         `aria-label` = "Next unit"))),
    div(class = "qc-ann-unit",
        div(id = "qc_ann_text", class = "qc-ann-text", `data-unit` = cur$id, HTML(qc_highlight(cur$text, spans, colors))),
        if (none) tags$p(class = "qc-hint", "Marked: no code fits this unit.")),
    div(class = "qc-chips",
        tags$span(class = "qc-chip-label", "Select a phrase, then a code:"),
        lapply(names(colors), function(k) {
          tags$button(type = "button", class = "qc-chip", style = sprintf("--qc-color:%s", qc_color(colors, k)),
                      `data-code` = k, k)
        }),
        actionButton("qc_ann_none", "No code fits", class = "btn-default btn-sm")),
    if (nrow(coded) > 0) {
      div(class = "qc-span-list",
          tags$span(class = "qc-chip-label", "Codes on this unit:"),
          lapply(seq_len(nrow(coded)), function(i) {
            phrase <- substr(cur$text, coded$start[i], coded$end[i])
            tags$button(type = "button", class = "qc-span-chip",
                        style = sprintf("--qc-color:%s", qc_color(colors, coded$code[i])),
                        `data-code` = coded$code[i], `data-start` = coded$start[i], `data-end` = coded$end[i],
                        `aria-label` = sprintf("Remove %s from \"%s\"", coded$code[i], phrase),
                        sprintf("%s: “%s”", coded$code[i], strtrim(phrase, 40)),
                        tags$i(class = "fa fa-times", `aria-hidden` = "true"))
          }))
    },
    tags$p(class = "qc-tip",
           "Remove a code with ×. Overlaps show as stacked bands. A chip with no phrase selected codes the whole unit."))
})

observeEvent(input$qc_ann_apply, {
  cur <- qc_ann_current()
  req(cur)
  code <- as.character(input$qc_ann_apply$code)
  sel <- input$qc_ann_apply$selection
  start <- 1L
  end <- nchar(cur$text)
  if (!is.null(sel) && identical(sel$unit, cur$id) && isTRUE(sel$end > sel$start)) {
    start <- max(1L, as.integer(sel$start) + 1L)
    end <- min(nchar(cur$text), as.integer(sel$end))
  }
  row <- tibble::tibble(doc_id = qc_doc_id(cur$id), unit_id = cur$id, start = start, end = end,
                        code = code, coder = qc_coder(), confidence = NA_real_, rationale = NA_character_,
                        status = "human", timestamp = qc_now(), stage = qc_stage_of(cur$id))
  qc_edit_project(function(p) {
    keep <- !(p$assignments$unit_id == cur$id & p$assignments$coder == p$coder & p$assignments$status == "human-none")
    p$assignments <- dplyr::bind_rows(p$assignments[keep, ], row)
    p
  })
})

observeEvent(input$qc_ann_remove, {
  cur <- qc_ann_current()
  req(cur)
  r <- input$qc_ann_remove
  qc_edit_project(function(p) {
    a <- p$assignments
    hit <- a$unit_id == cur$id & a$coder == p$coder & a$status == "human" &
      a$code == as.character(r$code) & a$start == as.integer(r$start) & a$end == as.integer(r$end)
    p$assignments <- a[!hit, ]
    p
  })
})

observeEvent(input$qc_ann_none, {
  cur <- qc_ann_current()
  req(cur)
  row <- tibble::tibble(doc_id = qc_doc_id(cur$id), unit_id = cur$id, start = 1L, end = nchar(cur$text),
                        code = NA_character_, coder = qc_coder(), confidence = NA_real_, rationale = NA_character_,
                        status = "human-none", timestamp = qc_now(), stage = qc_stage_of(cur$id))
  qc_edit_project(function(p) {
    mine <- p$assignments$unit_id == cur$id & p$assignments$coder == p$coder &
      p$assignments$status %in% c("human", "human-none")
    p$assignments <- dplyr::bind_rows(p$assignments[!mine, ], row)
    p
  })
})

# themes

qc_themes <- reactive({
  cb <- qc_codebook()
  req(cb, "theme" %in% names(cb), any(!is.na(cb$theme) & nzchar(trimws(cb$theme))))
  TextAnalysisR::code_themes(qc_project()$assignments, cb)
})

output$has_qc_themes <- reactive(!is.null(tryCatch(qc_themes(), error = function(e) NULL)))
outputOptions(output, "has_qc_themes", suspendWhenHidden = FALSE)

output$qc_themes_table <- DT::renderDataTable({
  dt_table(setNames(qc_themes(), c("Theme", "Codes", "Code count", "Units", "Documents")),
           rownames = FALSE, selection = "none", options = list(dom = "t"))
})

# raw vs curated groups

qc_alignment <- reactive({
  u <- qc_units()
  req(u, nrow(qc_rulings_rv()) > 0)
  tryCatch(TextAnalysisR::align_categories(qc_groups(), u$groups0, unassigned = "0"), error = function(e) NULL)
})

output$has_qc_alignment <- reactive(!is.null(qc_alignment()))
outputOptions(output, "has_qc_alignment", suspendWhenHidden = FALSE)

output$qc_align_note <- renderUI({
  a <- qc_alignment()
  req(a)
  ari <- if (is.na(a$adjusted_rand)) "not defined" else format(round(a$adjusted_rand, 3), nsmall = 3)
  tags$p(class = "qc-tip", sprintf(
    "Adjusted Rand index %s over %d units: 1 means the rulings left the clusters unchanged, near 0 means unrelated.",
    ari, a$n))
})

output$qc_align_table <- DT::renderDataTable({
  b <- qc_alignment()$best_match
  req(b)
  dt_table(setNames(b, c("Curated group", "Raw cluster", "Units", "Share")), rownames = FALSE,
           selection = "none", options = list(dom = "t")) %>% DT::formatPercentage("Share", 1)
})

# saturation

qc_saturation <- reactive({
  a <- qc_project()$assignments
  a <- a[is.na(a$stage) | a$stage != "confirm", , drop = FALSE]
  TextAnalysisR::code_saturation(a, window = 10)
})

output$has_qc_saturation <- reactive(nrow(qc_saturation()$curve) > 0)
outputOptions(output, "has_qc_saturation", suspendWhenHidden = FALSE)

output$qc_saturation_note <- renderUI({
  s <- qc_saturation()$summary
  msg <- if (s$saturated) {
    sprintf("No new code in the last %d units (%d codes after %d units).", s$window, s$n_codes, s$n_units)
  } else if (is.na(s$last_new_step)) {
    sprintf("%d units coded, no codes yet.", s$n_units)
  } else {
    sprintf("New codes are still appearing: last new code at unit %d of %d (%d codes).", s$last_new_step, s$n_units, s$n_codes)
  }
  tags$p(class = "qc-tip", msg)
})

output$qc_saturation_plot <- plotly::renderPlotly({
  d <- qc_saturation()$curve
  req(nrow(d) > 0)
  d$tip <- sprintf("Unit %d: %d codes%s", d$step, d$distinct_codes,
                   ifelse(d$new_codes > 0, sprintf(" (+%d new)", d$new_codes), ""))
  p <- ggplot2::ggplot(d, ggplot2::aes(x = step, y = distinct_codes, text = tip)) +
    ggplot2::geom_step(color = "#4269BF", group = 1) +
    ggplot2::geom_point(data = d[d$new_codes > 0, , drop = FALSE], color = "#4269BF", size = 1.5) +
    ggplot2::labs(x = "Units coded", y = "Distinct codes") +
    ggplot2::scale_y_continuous(breaks = function(l) unique(round(pretty(l)))) +
    ggplot2::theme_minimal(base_size = 14)
  gg_to_plotly(p)
})

# memos

qc_memo_list <- function(memos) {
  if (nrow(memos) == 0) return(tags$p(class = "qc-tip", "No memos yet."))
  memos <- memos[order(memos$created, decreasing = TRUE), , drop = FALSE]
  tags$ul(class = "qc-memo-list", lapply(seq_len(nrow(memos)), function(i) {
    stage <- if (is.na(memos$stage[i])) "memo" else memos$stage[i]
    tags$li(tags$span(class = "qc-memo-meta", sprintf("%s, round %s", stage, memos$round[i])), memos$text[i])
  }))
}

observe({
  updateSelectInput(session, "qc_cat_memo_code", choices = qc_codebook()$code)
})

output$qc_cat_memo_list <- renderUI({
  req(input$qc_cat_memo_code)
  qc_memo_list(TextAnalysisR::get_memos(qc_project()$memos, "category", input$qc_cat_memo_code))
})

output$qc_ann_memo_list <- renderUI({
  cur <- qc_ann_current()
  req(cur)
  qc_memo_list(TextAnalysisR::get_memos(qc_project()$memos, "unit", cur$id))
})

observeEvent(input$qc_cat_memo_save, {
  code <- input$qc_cat_memo_code
  txt <- trimws(input$qc_cat_memo %||% "")
  req(nzchar(code %||% ""), nzchar(txt))
  qc_edit_project(function(p) {
    p$memos <- TextAnalysisR::add_memo(p$memos, "category", code, txt, round = qc_round(), stage = "refine")
    p
  })
  updateTextAreaInput(session, "qc_cat_memo", value = "")
})

output$qc_download_memos <- downloadHandler(
  filename = function() paste0("memos-", Sys.Date(), ".csv"),
  content = function(file) utils::write.csv(qc_project()$memos, file, row.names = FALSE)
)

observeEvent(input$qc_ann_save_memo, {
  cur <- qc_ann_current()
  txt <- trimws(input$qc_ann_memo %||% "")
  req(cur, nzchar(txt))
  qc_edit_project(function(p) {
    p$memos <- TextAnalysisR::add_memo(p$memos, "unit", cur$id, txt, round = qc_round(), stage = qc_stage_of(cur$id))
    p
  })
  showNotification("Memo saved.", type = "message", duration = 3)
})

# blind holdout

output$qc_holdout_summary <- renderUI({
  p <- qc_project()
  u <- qc_units()
  total <- if (is.null(u)) length(analysis_texts()) else length(u$ids)
  read <- length(unique(p$units_read$unit_id[p$units_read$step != "holdout"]))
  tags$p(class = "qc-tip",
         if (nrow(p$holdout) > 0) sprintf("Holdout: %d units, none read during refinement; they are hidden from ruling and annotation.", nrow(p$holdout))
         else sprintf("%d of %d units are unread and eligible.", total - read, total))
})

observe({
  cats <- tryCatch(colnames_cat_doc(), error = function(e) character(0))
  updateSelectInput(session, "qc_holdout_strata", choices = c("None" = "", cats))
})

observeEvent(input$qc_draw_holdout, {
  n <- input$qc_holdout_n
  if (is.null(n) || !is.finite(n) || n < 1) {
    showNotification("Enter a holdout size.", type = "warning", duration = 6)
    return()
  }
  u <- qc_units()
  ids <- if (!is.null(u)) u$ids else names(analysis_texts())
  p <- qc_project()
  if (nrow(p$holdout) > 0) {
    showNotification("A holdout is already drawn; drawing again would expose its units.", type = "warning", duration = 8)
    return()
  }
  strata <- NULL
  if (nzchar(input$qc_holdout_strata %||% "")) {
    doc_vals <- united_tbl()[[input$qc_holdout_strata]]
    doc_index <- suppressWarnings(as.integer(sub("^doc([0-9]+).*$", "\\1", ids)))
    if (anyNA(doc_index)) {
      showNotification("These units cannot be traced to their source rows, so the holdout is drawn without spreading.",
                       type = "warning", duration = 8)
    } else {
      strata <- as.character(doc_vals[doc_index])
    }
  }
  read <- p$units_read[p$units_read$step != "holdout", , drop = FALSE]
  picked <- withCallingHandlers(
    TextAnalysisR::draw_holdout(ids, read, n = n, strata = strata),
    warning = function(w) {
      showNotification(conditionMessage(w), type = "warning", duration = 8)
      invokeRestart("muffleWarning")
    })
  qc_edit_project(function(p) {
    p$holdout <- tibble::tibble(unit_id = picked)
    p
  })
  showNotification(sprintf("Holdout of %d units drawn. Code them in Annotate (source: Blind holdout).", length(picked)),
                   type = "message", duration = 8)
})

# accepted AI suggestions are AI-anchored, so agreement uses the coder's own codes only
qc_project_codes <- function(p, holdout_only) {
  a <- p$assignments[p$assignments$status %in% c("human", "human-none"), , drop = FALSE]
  if (holdout_only && nrow(p$holdout) > 0) a <- a[a$unit_id %in% p$holdout$unit_id, , drop = FALSE]
  a[, c("doc_id", "unit_id", "start", "end", "code", "coder", "confidence", "status")]
}
