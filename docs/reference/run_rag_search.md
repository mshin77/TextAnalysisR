# RAG-Enhanced Semantic Search

Uses LangGraph multi-agent workflow for Retrieval Augmented Generation.
Provides question-answering over document corpus with source
attribution.

## Usage

``` r
run_rag_search(
  query,
  documents,
  ollama_model = "llama3",
  ollama_base_url = "http://localhost:11434",
  embedding_model = "nomic-embed-text",
  top_k = 5,
  envname = "textanalysisr-env"
)
```

## Arguments

- query:

  Character string, user question

- documents:

  Character vector, corpus to search

- ollama_model:

  Character string, LLM model (default: "llama3")

- ollama_base_url:

  Character string, Ollama API endpoint

- embedding_model:

  Character string, embedding model (default: "nomic-embed-text")

- top_k:

  Integer, number of documents to retrieve (default: 5)

- envname:

  Character string, Python environment name

## Value

List with:

- success: Logical

- answer: Generated answer

- confidence: Confidence score (0-1)

- sources: Vector of source document IDs

- retrieved_docs: Retrieved document chunks

- scores: Similarity scores

## Details

Multi-agent workflow:

1.  Retrieval Agent: Find relevant documents via embeddings

2.  Generation Agent: Create answer from context

3.  Validation Agent: Assess answer quality

4.  Conditional retry if confidence \< 0.4

Requires Ollama with embedding model:

    ollama pull llama3
    ollama pull nomic-embed-text

## See also

Other ai:
[`analyze_contrastive_similarity()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_contrastive_similarity.md),
[`call_ollama()`](https://mshin77.github.io/TextAnalysisR/reference/call_ollama.md),
[`check_ollama()`](https://mshin77.github.io/TextAnalysisR/reference/check_ollama.md),
[`create_label_selection_data()`](https://mshin77.github.io/TextAnalysisR/reference/create_label_selection_data.md),
[`format_label_candidates()`](https://mshin77.github.io/TextAnalysisR/reference/format_label_candidates.md),
[`generate_survey_items()`](https://mshin77.github.io/TextAnalysisR/reference/generate_survey_items.md),
[`generate_topic_content()`](https://mshin77.github.io/TextAnalysisR/reference/generate_topic_content.md),
[`generate_topic_labels_langgraph()`](https://mshin77.github.io/TextAnalysisR/reference/generate_topic_labels_langgraph.md),
[`get_content_type_prompt()`](https://mshin77.github.io/TextAnalysisR/reference/get_content_type_prompt.md),
[`get_content_type_user_template()`](https://mshin77.github.io/TextAnalysisR/reference/get_content_type_user_template.md),
[`get_recommended_ollama_model()`](https://mshin77.github.io/TextAnalysisR/reference/get_recommended_ollama_model.md),
[`list_ollama_models()`](https://mshin77.github.io/TextAnalysisR/reference/list_ollama_models.md),
[`validate_topic_labels_langgraph()`](https://mshin77.github.io/TextAnalysisR/reference/validate_topic_labels_langgraph.md)

## Examples

``` r
if (FALSE) { # \dontrun{
documents <- c(
  "Assistive technology helps students with disabilities access curriculum.",
  "Universal Design for Learning provides multiple means of engagement.",
  "Response to Intervention uses tiered support systems."
)

result <- run_rag_search(
  query = "How does assistive technology support learning?",
  documents = documents
)

if (result$success) {
  cat("Answer:", result$answer, "\n")
  cat("Confidence:", result$confidence, "\n")
  cat("Sources:", paste(result$sources, collapse = ", "), "\n")
}
} # }
```
