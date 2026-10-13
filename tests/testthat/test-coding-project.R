test_that("new_coding_project starts every table empty and records the coder", {
  p <- new_coding_project("Coder A", context = "Open-ended survey item")
  expect_s3_class(p, "coding_project")
  expect_equal(p$coder, "Coder A")
  tables <- c("codebook", "assignments", "memos", "positionality", "rounds",
              "rulings", "units_read", "holdout")
  expect_true(all(tables %in% names(p)))
  expect_true(all(vapply(p[tables], nrow, integer(1)) == 0))
})

test_that("new_coding_project requires a coder name", {
  expect_error(new_coding_project(""), "coder")
  expect_error(new_coding_project(c("a", "b")), "coder")
})

test_that("save and read round-trip a project and keep the previous file as .bak", {
  path <- withr::local_tempfile(fileext = ".rds")
  p <- new_coding_project("A")
  p$codebook <- tibble::tibble(code = "Barrier", definition = "Names a limit",
                               example = NA_character_, color = "#EF9F27")
  save_coding_project(p, path)
  p$rulings <- dplyr::bind_rows(p$rulings, tibble::tibble(round = 1L, group = "g1", n = 10L,
                                                          verdict = "keep", coder = "A"))
  save_coding_project(p, path)
  back <- read_coding_project(path)
  expect_s3_class(back, "coding_project")
  expect_equal(back$codebook$code, "Barrier")
  expect_equal(nrow(back$rulings), 1)
  expect_false(is.na(back$saved))
  expect_true(file.exists(paste0(path, ".bak")))
  expect_equal(nrow(read_coding_project(paste0(path, ".bak"))$rulings), 0)
})

test_that("read_coding_project fills tables missing from an older file", {
  path <- withr::local_tempfile(fileext = ".rds")
  old <- unclass(new_coding_project("A"))
  old$rulings <- NULL
  old$units_read <- NULL
  old$version <- 0L
  saveRDS(old, path)
  back <- read_coding_project(path)
  expect_equal(nrow(back$rulings), 0)
  expect_true("unit_id" %in% names(back$units_read))
})

test_that("read_coding_project rejects files that are not projects or come from a newer version", {
  path <- withr::local_tempfile(fileext = ".rds")
  saveRDS(list(a = 1), path)
  expect_error(read_coding_project(path), "not a coding project")
  newer <- unclass(new_coding_project("A"))
  newer$version <- 99L
  saveRDS(newer, path)
  expect_error(read_coding_project(path), "newer version")
})

test_that("save_coding_project refuses objects that are not projects", {
  expect_error(save_coding_project(list(coder = "A"), tempfile(fileext = ".rds")), "new_coding_project")
})

test_that("save_coding_project skips the backup when asked", {
  path <- withr::local_tempfile(fileext = ".rds")
  p <- new_coding_project("A")
  save_coding_project(p, path, backup = FALSE)
  save_coding_project(p, path, backup = FALSE)
  expect_false(file.exists(paste0(path, ".bak")))
})

test_that("read_coding_project fills columns an older file lacks", {
  path <- withr::local_tempfile(fileext = ".rds")
  p <- unclass(new_coding_project("A"))
  p$rulings <- tibble::tibble(round = 1L, group = "g1", verdict = "keep")
  saveRDS(p, path)
  back <- read_coding_project(path)
  expect_true(all(names(new_coding_project("A")$rulings) %in% names(back$rulings)))
  expect_true(is.na(back$rulings$label))
})

test_that("stage tags are recorded and validated", {
  p <- new_coding_project("a")
  expect_true(all(c("stage") %in% names(p$assignments), "stage" %in% names(p$memos),
                  "stage" %in% names(p$rulings), "stage" %in% names(p$rounds)))
  m <- add_memo(NULL, "unit", "u1", "note", round = 1, stage = "annotate")
  expect_equal(m$stage, "annotate")
  expect_true(is.na(add_memo(NULL, "unit", "u1", "note")$stage))
  expect_error(add_memo(NULL, "unit", "u1", "note", stage = "review"), "stage must be")
  expect_equal(log_round(NULL, 3, 10, 2)$stage, "refine")
})

test_that("older project files get stages from what the rows are", {
  p <- new_coding_project("a")
  p$holdout <- tibble::tibble(unit_id = "u2")
  p$assignments <- tibble::tibble(doc_id = c("d1", "d2"), unit_id = c("u1", "u2"), start = 1L, end = 5L,
                                  code = "x", coder = "a", confidence = NA_real_, rationale = NA_character_,
                                  status = "human", timestamp = Sys.time())
  p$memos <- dplyr::bind_rows(add_memo(NULL, "unit", "u1", "n"), add_memo(NULL, "category", "x", "n"))
  p$memos$stage <- NULL
  p$rulings <- tibble::tibble(round = 1L, group = "g1", n = 4L, verdict = "keep", merge_with = NA_character_,
                              label = "L", definition = "D", coder = "a", timestamp = Sys.time())
  path <- tempfile(fileext = ".rds")
  saveRDS(unclass(p), path)
  q <- read_coding_project(path)
  expect_equal(q$assignments$stage, c("annotate", "confirm"))
  expect_equal(q$memos$stage, c("annotate", "refine"))
  expect_equal(q$rulings$stage, "refine")
})
