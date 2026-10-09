## Headless walkthrough of the Shiny app: starts the app, drives the changed screens, saves screenshots
## Input: inst/TextAnalysisR.app (working tree) and the installed TextAnalysisR package
## Output: tests/manual/screenshots/*.png and a checklist printed to the console
## Run from the package root: Rscript tests/manual/drive-app.R

stopifnot(requireNamespace("chromote", quietly = TRUE), requireNamespace("processx", quietly = TRUE))
options(chromote.timeout = 60)

port <- 8123
out_dir <- file.path("tests", "manual", "screenshots")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

app <- processx::process$new(
  file.path(R.home("bin"), "Rscript"),
  c("-e", sprintf("shiny::runApp('inst/TextAnalysisR.app', port = %d, launch.browser = FALSE)", port)),
  stdout = "|", stderr = "2>&1"
)
on.exit(app$kill(), add = TRUE)

for (i in 1:120) {
  up <- tryCatch(identical(httr::status_code(httr::GET(sprintf("http://127.0.0.1:%d/", port))), 200L), error = function(e) FALSE)
  if (up) break
  Sys.sleep(1)
}

b <- chromote::ChromoteSession$new(width = 1400, height = 1000)
on.exit(b$close(), add = TRUE)

js <- function(expr) b$Runtime$evaluate(expr, awaitPromise = TRUE, returnByValue = TRUE, timeout_ = 120)$result$value
wait_for <- function(cond, secs = 60) {
  for (i in seq_len(secs * 2)) {
    if (isTRUE(tryCatch(js(cond), error = function(e) FALSE))) return(TRUE)
    Sys.sleep(0.5)
  }
  FALSE
}
tab_visible <- function(text) sprintf("[...document.querySelectorAll('a')].some(a => a.offsetParent && a.textContent.trim() === '%s')", text)
click_text <- function(text) {
  wait_for(tab_visible(text), 60)
  js(sprintf("(() => { const e = [...document.querySelectorAll('a, button')].find(x => x.offsetParent && x.textContent.trim() === '%s'); if (e) e.click(); return !!e; })()", text))
}
click_id <- function(id) js(sprintf("(() => { const e = document.getElementById('%s'); if (e) e.click(); return !!e; })()", id))
set_selectize <- function(id, value) js(sprintf("(() => { const s = document.getElementById('%s'); if (s && s.selectize) s.selectize.setValue('%s'); return !!s; })()", id, value))
selectize_options <- function(id) js(sprintf("(() => { const s = document.getElementById('%s'); return s && s.selectize ? Object.keys(s.selectize.options).join(', ') : 'missing'; })()", id))
close_modals <- function() js("(() => { document.querySelectorAll('.modal button').forEach(x => { if (/Close|OK/.test(x.textContent)) x.click(); }); return true; })()")
idle <- "!document.documentElement.classList.contains('shiny-busy')"
shot <- function(name) {
  Sys.sleep(1.5)
  b$screenshot(file.path(out_dir, paste0(name, ".png")), show = FALSE)
}
results <- list()
check <- function(label, ok) results[[label]] <<- isTRUE(ok)

b$Page$navigate(sprintf("http://127.0.0.1:%d/", port))
wait_for("!!(window.Shiny && Shiny.shinyapp && Shiny.shinyapp.isConnected())", 90)

click_text("Upload")
wait_for("!!document.getElementById('dataset_choice') && !!document.getElementById('dataset_choice').selectize")
set_selectize("dataset_choice", "Upload an Example Dataset")
Sys.sleep(3)

click_text("Preprocess")
wait_for("!!document.querySelector('input[type=checkbox][value=abstract]')")
js("(() => { const c = document.querySelector('input[type=checkbox][value=abstract]'); if (!c.checked) c.click(); return c.checked; })()")
click_id("apply"); Sys.sleep(2); wait_for(idle)
click_text("2. Segment Texts"); click_id("preprocess"); Sys.sleep(3); wait_for(idle, 120); close_modals()

click_text("3. Remove Stopwords"); Sys.sleep(3)
check("stopwords: nothing pre-selected", identical(js("JSON.stringify(Shiny.shinyapp.$inputValues.common_words || [])"), "[]"))
shot("01-stopwords-none-selected")
click_id("suggest_common_words")
check("stopwords: suggest fills 10 terms", wait_for("(Shiny.shinyapp.$inputValues.common_words || []).length === 10", 30))
shot("02-stopwords-after-suggest")
click_id("skip_stopwords"); Sys.sleep(3); wait_for(idle); close_modals()

click_text("5. Document-Feature Matrix"); click_id("dfm_btn"); Sys.sleep(5); wait_for(idle, 180); close_modals()

click_text("Lexical Analysis"); click_text("Readability")
wait_for("!!document.getElementById('readability_text_source')")
check("readability: defaults to longest prose column", identical(js("document.getElementById('readability_text_source').value"), "abstract"))
shot("03-readability-text-source")

click_text("Keywords"); click_text("Statistical Keyness")
check("group dropdown filled on first visit", grepl("reference_type", selectize_options("tfidf_group_var")))
set_selectize("tfidf_group_var", "reference_type")
wait_for("!!document.getElementById('keyness_target')", 30)
set_selectize("keyness_target", "thesis"); Sys.sleep(1)
click_id("run_keyword_extraction"); Sys.sleep(5); wait_for(idle, 120)
header <- js("[...document.querySelectorAll('table.dataTable thead th')].filter(h => h.offsetParent).map(h => h.textContent.trim()).join('|')")
check("keyness: effect size and adjusted p columns", grepl("Log_Ratio", header) && grepl("P_Adjusted", header))
shot("04-keyness-table")

click_text("Semantic Analysis"); click_text("Sentiment")
set_selectize("sentiment_lexicon", "nrc"); Sys.sleep(1)
click_id("run_sentiment_analysis"); Sys.sleep(5); wait_for(idle, 180)
click_text("Emotion"); Sys.sleep(3)
check("emotion dropdown filled on first visit", grepl("reference_type", selectize_options("emotion_group_var")))
set_selectize("emotion_group_var", "reference_type")
check("radar: one trace per group", wait_for("(() => { const e = document.getElementById('emotion_radar_plot'); return !!(e && e.data && e.data.length >= 2); })()"))
check("radar: aria-label follows title", wait_for("(() => { const e = document.getElementById('emotion_radar_plot'); return !!e && /per 1,000 retained tokens/.test(e.getAttribute('aria-label') || ''); })()", 20))
shot("05-emotion-radar-grouped")

# empty text: a numbers-only column must explain itself, never close the session
generic_seen <- ""
watch <- function(secs = 6) {
  for (i in seq_len(secs * 2)) { generic_seen <<- paste(generic_seen, js("[...document.querySelectorAll('.shiny-notification')].map(n => n.textContent).join(' ')")); Sys.sleep(0.5) }
}
connected <- "!!(Shiny.shinyapp && Shiny.shinyapp.isConnected())"
click_text("Preprocess"); click_text("1. Unite Texts"); Sys.sleep(2)
js("(() => { document.querySelectorAll('#show_vars input[type=checkbox]').forEach(c => { if (c.checked !== (c.value === 'year')) c.click(); }); return true; })()")
click_id("apply"); watch(3); wait_for(idle)
click_text("2. Segment Texts"); click_id("preprocess"); watch(4); wait_for(idle, 120); close_modals()
click_text("4. Multi-Word Dictionary"); Sys.sleep(2); click_id("dictionary"); watch(4); wait_for(idle); close_modals()
click_text("3. Remove Stopwords"); Sys.sleep(2); click_id("skip_stopwords"); watch(3); wait_for(idle); close_modals()
click_text("5. Document-Feature Matrix"); click_id("dfm_btn"); watch(5); wait_for(idle, 120); close_modals()
click_text("Lexical Analysis"); Sys.sleep(2); click_text("Annotation"); Sys.sleep(2); click_text("Word Forms (Lemmas)"); Sys.sleep(2)
click_id("skip"); watch(4); wait_for(idle)
click_text("Dispersion"); watch(4)
check("empty text: session stays connected", isTRUE(js(connected)))
check("empty text: no generic error notice", !grepl("unexpected error", generic_seen, ignore.case = TRUE))
check("empty text: no R errors in the server log", !grepl("Warning: Error|Error in", app$read_output()))

for (label in names(results)) cat(sprintf("[%s] %s\n", if (results[[label]]) "PASS" else "FAIL", label))
cat(sprintf("%d of %d checks passed; screenshots in %s\n", sum(unlist(results)), length(results), out_dir))
