test_that("process_pdf_unified validates file existence", {
  result <- process_pdf_unified("nonexistent.pdf")

  expect_false(result$success)
  expect_equal(result$type, "error")
  expect_equal(result$method, "none")
  expect_match(result$message, "File not found")
})

test_that("process_pdf_unified falls back to R when Python unavailable", {
  skip_if_not_installed("pdftools")
  skip_if(!file.exists("test-sample.pdf"))

  result <- process_pdf_unified(
    "test-sample.pdf",
    use_multimodal = FALSE
  )

  expect_type(result, "list")
  expect_true("success" %in% names(result))
  expect_true("method" %in% names(result))
  expect_true("data" %in% names(result))
})

test_that("import_files handles DOCX text extraction", {
  skip_if_not_installed("officer")
  skip_if(!file.exists("test-sample.docx"))

  file_info <- data.frame(
    filepath = "test-sample.docx",
    stringsAsFactors = FALSE
  )

  result <- import_files(
    dataset_choice = "Upload Your File",
    file_info = file_info
  )

  expect_s3_class(result, "tbl_df")
  expect_true("text" %in% names(result))
  expect_true(nrow(result) > 0)
})

test_that("import_files handles CSV files", {
  temp_csv <- tempfile(fileext = ".csv")
  write.csv(
    data.frame(text = c("Test1", "Test2"), stringsAsFactors = FALSE),
    temp_csv,
    row.names = FALSE
  )

  file_info <- data.frame(filepath = temp_csv, stringsAsFactors = FALSE)
  result <- import_files("Upload Your File", file_info = file_info)

  expect_s3_class(result, "tbl_df")
  expect_equal(nrow(result), 2)

  unlink(temp_csv)
})

test_that("import_files handles TXT files", {
  temp_txt <- tempfile(fileext = ".txt")
  writeLines(c("Line 1", "Line 2"), temp_txt)

  file_info <- data.frame(filepath = temp_txt, stringsAsFactors = FALSE)
  result <- import_files("Upload Your File", file_info = file_info)

  expect_s3_class(result, "tbl_df")
  expect_equal(nrow(result), 2)

  unlink(temp_txt)
})

test_that("import_files validates empty files", {
  temp_txt <- tempfile(fileext = ".txt")
  writeLines("", temp_txt)

  file_info <- data.frame(filepath = temp_txt, stringsAsFactors = FALSE)
  result <- import_files("Upload Your File", file_info = file_info)

  expect_s3_class(result, "tbl_df")
  expect_equal(nrow(result), 0)

  unlink(temp_txt)
})

test_that("import_files keeps the PDF page for each row at every unit", {
  skip_if_not_installed("pdftools")
  pdf_path <- tempfile(fileext = ".pdf")
  grDevices::pdf(pdf_path)
  for (p in 1:2) {
    graphics::plot.new()
    graphics::text(0.5, 0.8, sprintf("Teachers described routine number %d", p))
    graphics::text(0.5, 0.4, sprintf("Students valued feedback in session %d", p))
  }
  grDevices::dev.off()
  on.exit(unlink(pdf_path))
  info <- data.frame(filepath = pdf_path)

  lines <- import_files("Upload Your File", file_info = info)
  pages <- import_files("Upload Your File", file_info = info, pdf_unit = "page")
  whole <- import_files("Upload Your File", file_info = info, pdf_unit = "document")

  expect_equal(nrow(lines), 4)
  expect_equal(lines$page, c(1, 1, 2, 2))
  expect_equal(nrow(pages), 2)
  expect_equal(pages$page, pages$page_end)
  expect_match(pages$text[2], "session 2")
  expect_equal(nrow(whole), 1)
  expect_equal(c(whole$page, whole$page_end), c(1, 2))
})
