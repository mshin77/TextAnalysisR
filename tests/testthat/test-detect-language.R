.dl_texts <- list(
  en = "The students with learning disabilities raised their mathematics performance after the intervention was applied in the classroom.",
  de = "Die Schuler mit Lernschwierigkeiten haben ihre Leistungen in der Mathematik verbessert, nachdem die Intervention in dem Klassenzimmer angewendet wurde.",
  fr = "Les eleves avec des difficultes d apprentissage ont ameliore leurs resultats en mathematiques apres que l intervention a ete appliquee dans la classe.",
  es = "Los estudiantes con dificultades de aprendizaje mejoraron sus resultados en matematicas despues de que la intervencion se aplico en el aula."
)

test_that("detect_language ranks the true language first", {
  skip_if_not_installed("stopwords")
  for (lg in names(.dl_texts)) {
    res <- detect_language(.dl_texts[[lg]])
    expect_equal(res$language[1], lg, info = lg)
  }
})

test_that("detect_language returns scores ordered descending in 0-1", {
  skip_if_not_installed("stopwords")
  res <- detect_language(.dl_texts$en)
  expect_true(all(diff(res$score) <= 0))
  expect_true(all(res$score >= 0 & res$score <= 1))
  expect_named(res, c("language", "score"))
})

test_that("detect_language returns NULL when there is nothing to score", {
  skip_if_not_installed("stopwords")
  expect_null(detect_language(character(0)))
  expect_null(detect_language(NA_character_))
  expect_null(detect_language("   "))
  expect_null(detect_language("123 456 789"))
})

test_that("detect_language scores near zero on non-language input", {
  skip_if_not_installed("stopwords")
  res <- detect_language("xyz qqq zzz vvv")
  expect_true(res$score[1] < 0.05)
})

test_that("detect_language honors the languages argument", {
  skip_if_not_installed("stopwords")
  res <- detect_language(.dl_texts$en, languages = c("en", "de"))
  expect_setequal(res$language, c("en", "de"))
  expect_equal(res$language[1], "en")
})

test_that("detect_language samples deterministically for a given seed", {
  skip_if_not_installed("stopwords")
  txt <- rep(unlist(.dl_texts), 60)
  a <- detect_language(txt, sample_n = 20, seed = 42)
  b <- detect_language(txt, sample_n = 20, seed = 42)
  expect_equal(a, b)
})

test_that("detect_language accepts a multi-document corpus", {
  skip_if_not_installed("stopwords")
  res <- detect_language(rep(.dl_texts$fr, 5))
  expect_equal(res$language[1], "fr")
})
