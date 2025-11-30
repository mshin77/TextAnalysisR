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
