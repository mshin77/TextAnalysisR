# RAG Semantic Search

Simple in-memory RAG (Retrieval Augmented Generation) for
question-answering over document corpus with source attribution. Uses
local (Ollama) or cloud (OpenAI/Gemini) embeddings for semantic search
and LLM for answer generation.

## Usage

``` r
run_rag_search(
  query,
  documents,
  provider = c("ollama", "openai", "gemini"),
  api_key = NULL,
  embedding_model = NULL,
  chat_model = NULL,
  top_k = 5
)
```

## Arguments

- query:

  Character string, user question

- documents:

  Character vector, corpus to search

- provider:

  Character string, provider: "ollama" (local), "openai", or "gemini"

- api_key:

  Character string, API key for cloud providers (or from
  OPENAI_API_KEY/GEMINI_API_KEY env). Not required for Ollama.

- embedding_model:

  Character string, embedding model. Defaults: "nomic-embed-text"
  (ollama), "text-embedding-3-small" (openai), "gemini-embedding-001"
  (gemini)

- chat_model:

  Character string, chat model. Defaults: "llama3.2" (ollama),
  "gpt-4.1-mini" (openai), "gemini-2.5-flash-lite" (gemini)

- top_k:

  Integer, number of documents to retrieve (default: 5)

## Value

List with:

- success: Logical

- answer: Generated answer

- confidence: Confidence score (0-1)

- sources: Vector of source document indices

- retrieved_docs: Retrieved document chunks

- scores: Similarity scores

## Details

Simple RAG workflow:

1.  Generate embeddings for documents and query

2.  Find top-k similar documents via cosine similarity

3.  Generate answer using LLM with retrieved context

## See also

[`get_best_embeddings()`](https://mshin77.github.io/TextAnalysisR/reference/get_best_embeddings.md)
for the retrieval step alone;
[`call_llm_api()`](https://mshin77.github.io/TextAnalysisR/reference/call_llm_api.md)
for the answer-generation step alone;
[`sanitize_llm_input()`](https://mshin77.github.io/TextAnalysisR/reference/sanitize_llm_input.md)
for an input safety check before calling

## Examples

``` r
if (interactive()) {
documents <- c(
  "Assistive technology helps students with disabilities access curriculum.",
  "Universal Design for Learning provides multiple means of engagement.",
  "Response to Intervention uses tiered support systems."
)

# Using local Ollama (free, private)
result <- run_rag_search(
  query = "How does assistive technology support learning?",
  documents = documents,
  provider = "ollama"
)

# Using OpenAI (requires API key)
result <- run_rag_search(
  query = "How does assistive technology support learning?",
  documents = documents,
  provider = "openai"
)

if (result$success) {
  cat("Answer:", result$answer, "\n")
  cat("Sources:", paste(result$sources, collapse = ", "), "\n")
}
}
```
