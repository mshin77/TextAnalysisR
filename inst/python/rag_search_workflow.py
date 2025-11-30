"""
RAG-Enhanced Semantic Search Workflow
Multi-agent system for document retrieval and question answering
"""

from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import operator


class RAGState(TypedDict):
    """State for RAG workflow"""
    query: str
    documents: List[str]
    retrieved_docs: List[str]
    retrieved_scores: List[float]
    answer: str
    confidence: float
    sources: List[int]
    error: str


def create_rag_workflow(
    ollama_model: str = "llama3",
    ollama_base_url: str = "http://localhost:11434",
    embedding_model: str = "nomic-embed-text",
    top_k: int = 5
):
    """
    Create LangGraph workflow for RAG-enhanced semantic search

    Args:
        ollama_model: LLM model for generation
        ollama_base_url: Ollama API endpoint
        embedding_model: Model for embeddings
        top_k: Number of documents to retrieve

    Returns:
        Compiled StateGraph workflow
    """

    # Initialize LLM and embeddings
    llm = Ollama(model=ollama_model, base_url=ollama_base_url)
    embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_base_url)

    # Define workflow nodes
    def retrieve_documents(state: RAGState) -> RAGState:
        """Agent 1: Retrieval - Find relevant documents"""
        query = state["query"]
        documents = state["documents"]

        # Create vector store
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = []
        doc_ids = []
        for i, doc in enumerate(documents):
            doc_chunks = text_splitter.split_text(doc)
            chunks.extend(doc_chunks)
            doc_ids.extend([i] * len(doc_chunks))

        vectorstore = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            metadatas=[{"doc_id": doc_id} for doc_id in doc_ids]
        )

        # Retrieve with scores
        results = vectorstore.similarity_search_with_score(query, k=top_k)

        retrieved_docs = [doc.page_content for doc, score in results]
        retrieved_scores = [float(score) for doc, score in results]
        sources = [doc.metadata["doc_id"] for doc, score in results]

        return {
            **state,
            "retrieved_docs": retrieved_docs,
            "retrieved_scores": retrieved_scores,
            "sources": list(set(sources))  # unique document IDs
        }

    def generate_answer(state: RAGState) -> RAGState:
        """Agent 2: Generation - Create answer from context"""
        query = state["query"]
        retrieved_docs = state["retrieved_docs"]

        context = "\n\n".join([
            f"Context {i+1}:\n{doc}"
            for i, doc in enumerate(retrieved_docs)
        ])

        prompt = f"""Based on the following context, answer the question.
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {query}

Answer:"""

        answer = llm.invoke(prompt)

        return {
            **state,
            "answer": answer
        }

    def validate_answer(state: RAGState) -> RAGState:
        """Agent 3: Validation - Check answer quality"""
        query = state["query"]
        answer = state["answer"]
        retrieved_docs = state["retrieved_docs"]

        validation_prompt = f"""Rate how well this answer addresses the question based on the context.
Rate from 0.0 (poor) to 1.0 (excellent).
Only respond with a number.

Question: {query}

Answer: {answer}

Context snippets: {len(retrieved_docs)}

Confidence score:"""

        try:
            confidence_str = llm.invoke(validation_prompt).strip()
            confidence = float(confidence_str)
            confidence = max(0.0, min(1.0, confidence))
        except:
            confidence = 0.5

        return {
            **state,
            "confidence": confidence
        }

    def should_retry(state: RAGState) -> str:
        """Conditional edge: Retry if confidence low"""
        if state.get("confidence", 0) < 0.4:
            return "retry"
        return "accept"

    # Build workflow graph
    workflow = StateGraph(RAGState)

    # Add nodes
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("validate", validate_answer)

    # Add edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "validate")

    # Conditional routing
    workflow.add_conditional_edges(
        "validate",
        should_retry,
        {
            "retry": "retrieve",  # Try with different retrieval params
            "accept": END
        }
    )

    return workflow.compile()


def run_rag_search(
    query: str,
    documents: List[str],
    ollama_model: str = "llama3",
    ollama_base_url: str = "http://localhost:11434",
    embedding_model: str = "nomic-embed-text",
    top_k: int = 5
) -> dict:
    """
    Execute RAG search workflow

    Args:
        query: User question
        documents: Corpus to search
        ollama_model: LLM model
        ollama_base_url: Ollama endpoint
        embedding_model: Embedding model
        top_k: Number of results

    Returns:
        dict with answer, confidence, sources
    """
    try:
        workflow = create_rag_workflow(
            ollama_model=ollama_model,
            ollama_base_url=ollama_base_url,
            embedding_model=embedding_model,
            top_k=top_k
        )

        initial_state = {
            "query": query,
            "documents": documents,
            "retrieved_docs": [],
            "retrieved_scores": [],
            "answer": "",
            "confidence": 0.0,
            "sources": [],
            "error": ""
        }

        result = workflow.invoke(initial_state)

        return {
            "success": True,
            "answer": result["answer"],
            "confidence": result["confidence"],
            "sources": result["sources"],
            "retrieved_docs": result["retrieved_docs"],
            "scores": result["retrieved_scores"]
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "answer": "",
            "confidence": 0.0,
            "sources": []
        }
