test_that("new_coding_project starts every table empty and records the coder", {
  p <- new_coding_project("Coder A", context = "AT survey RQ3")
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
