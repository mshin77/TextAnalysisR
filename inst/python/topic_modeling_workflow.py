from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM
from typing import TypedDict, Annotated, Optional
import operator
import json


class TopicModelState(TypedDict):
    """State schema for topic modeling workflow"""
    corpus: Optional[str]
    num_topics: int
    stm_results: Optional[dict]
    topic_terms: Optional[list]
    label_candidates: Optional[list]
    user_selected_labels: Optional[list]
    validation_metrics: Optional[dict]
    user_feedback: Optional[str]
    needs_revision: bool
    iteration: Annotated[int, operator.add]


def create_topic_modeling_workflow(
    ollama_model: str = "llama3",
    ollama_base_url: str = "http://localhost:11434"
):
    """
    Creates a LangGraph StateGraph workflow for topic modeling with human-in-the-loop.

    Args:
        ollama_model: Name of the Ollama model to use (default: "llama3")
        ollama_base_url: Base URL for Ollama API (default: "http://localhost:11434")

    Returns:
        Compiled LangGraph workflow application
    """

    llm = OllamaLLM(
        model=ollama_model,
        base_url=ollama_base_url,
        temperature=0.7
    )

    def generate_label_candidates(state: TopicModelState) -> dict:
        """
        Node: Generate multiple label candidates for each topic using LLM
        """
        if not state.get("topic_terms"):
            return {"label_candidates": [], "needs_revision": False}

        candidates = []

        for topic_idx, terms in enumerate(state["topic_terms"]):
            terms_str = ", ".join(terms[:10])

            prompt = f"""You are assisting in statistical topic modeling analysis.
Based on the top terms from a structural topic model:
{terms_str}

Generate 3 interpretive labels that:
1. Accurately reflect the statistical distribution of terms
2. Are concise (2-4 words each)
3. Are distinct and meaningful
4. Capture the semantic theme of the terms

Return your response as a JSON list of objects with 'label' and 'reasoning' fields.
Example: [{{"label": "Climate Change Policy", "reasoning": "Terms focus on environmental legislation"}}, ...]
"""

            try:
                response = llm.invoke(prompt)

                import re
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    topic_candidates = json.loads(json_match.group())
                else:
                    topic_candidates = [
                        {"label": f"Topic {topic_idx + 1} - {terms[0].title()}",
                         "reasoning": "Auto-generated from top term"}
                    ]
            except Exception as e:
                topic_candidates = [
                    {"label": f"Topic {topic_idx + 1}",
                     "reasoning": f"Error generating labels: {str(e)}"}
                ]

            candidates.append({
                "topic_index": topic_idx,
                "top_terms": terms,
                "candidates": topic_candidates
            })

        return {
            "label_candidates": candidates,
            "iteration": 1
        }

    def validate_labels(state: TopicModelState) -> dict:
        """
        Node: Validate user-selected labels using LLM
        """
        if not state.get("user_selected_labels"):
            return {"validation_metrics": {}, "needs_revision": False}

        selected_labels = state["user_selected_labels"]
        topic_terms = state.get("topic_terms", [])

        metrics = {
            "coherence_scores": [],
            "distinctiveness_scores": [],
            "overall_quality": 0.0
        }

        for idx, label in enumerate(selected_labels):
            if idx < len(topic_terms):
                terms_str = ", ".join(topic_terms[idx][:10])

                prompt = f"""Evaluate the quality of this topic label for statistical topic modeling:

Label: "{label}"
Top Terms: {terms_str}

Rate the following on a scale of 0-10:
1. How well does the label capture the semantic theme of the terms?
2. Is the label distinct and not generic?

Return your response as a JSON object:
{{"coherence": <score>, "distinctiveness": <score>, "explanation": "<brief explanation>"}}
"""

                try:
                    response = llm.invoke(prompt)

                    import re
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        eval_result = json.loads(json_match.group())
                        coherence = float(eval_result.get("coherence", 5.0))
                        distinctiveness = float(eval_result.get("distinctiveness", 5.0))
                    else:
                        coherence = 7.0
                        distinctiveness = 7.0
                except Exception:
                    coherence = 7.0
                    distinctiveness = 7.0

                metrics["coherence_scores"].append(coherence)
                metrics["distinctiveness_scores"].append(distinctiveness)

        if metrics["coherence_scores"]:
            avg_coherence = sum(metrics["coherence_scores"]) / len(metrics["coherence_scores"])
            avg_distinctiveness = sum(metrics["distinctiveness_scores"]) / len(metrics["distinctiveness_scores"])
            metrics["overall_quality"] = (avg_coherence + avg_distinctiveness) / 2

        needs_revision = metrics["overall_quality"] < 6.0 and state.get("iteration", 0) < 3

        return {
            "validation_metrics": metrics,
            "needs_revision": needs_revision
        }

    def should_revise(state: TopicModelState) -> str:
        """
        Conditional edge: Determine if labels need revision
        """
        if state.get("needs_revision", False):
            return "revise"
        else:
            return "finalize"

    workflow = StateGraph(TopicModelState)

    workflow.add_node("generate_labels", generate_label_candidates)
    workflow.add_node("validate", validate_labels)

    workflow.set_entry_point("generate_labels")
    workflow.add_edge("generate_labels", "validate")

    workflow.add_conditional_edges(
        "validate",
        should_revise,
        {
            "revise": "generate_labels",
            "finalize": END
        }
    )

    return workflow.compile()


def run_topic_label_generation(
    topic_terms: list,
    num_topics: int,
    ollama_model: str = "llama3",
    ollama_base_url: str = "http://localhost:11434"
) -> dict:
    """
    Run topic label generation workflow.

    Args:
        topic_terms: List of lists, where each inner list contains top terms for a topic
        num_topics: Number of topics
        ollama_model: Ollama model name
        ollama_base_url: Ollama API base URL

    Returns:
        Dictionary with label candidates and validation metrics
    """

    workflow = create_topic_modeling_workflow(
        ollama_model=ollama_model,
        ollama_base_url=ollama_base_url
    )

    initial_state = {
        "corpus": None,
        "num_topics": num_topics,
        "stm_results": None,
        "topic_terms": topic_terms,
        "label_candidates": None,
        "user_selected_labels": None,
        "validation_metrics": None,
        "user_feedback": None,
        "needs_revision": False,
        "iteration": 0
    }

    config = {"configurable": {"thread_id": "topic_modeling_session"}}

    try:
        final_state = workflow.invoke(initial_state, config)
        return {
            "success": True,
            "label_candidates": final_state.get("label_candidates"),
            "validation_metrics": final_state.get("validation_metrics"),
            "needs_revision": final_state.get("needs_revision", False)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "label_candidates": None
        }


def validate_user_labels(
    user_labels: list,
    topic_terms: list,
    ollama_model: str = "llama3",
    ollama_base_url: str = "http://localhost:11434"
) -> dict:
    """
    Validate user-selected topic labels.

    Args:
        user_labels: List of user-selected labels
        topic_terms: List of lists of top terms for each topic
        ollama_model: Ollama model name
        ollama_base_url: Ollama API base URL

    Returns:
        Dictionary with validation metrics
    """

    workflow = create_topic_modeling_workflow(
        ollama_model=ollama_model,
        ollama_base_url=ollama_base_url
    )

    state = {
        "corpus": None,
        "num_topics": len(user_labels),
        "stm_results": None,
        "topic_terms": topic_terms,
        "label_candidates": None,
        "user_selected_labels": user_labels,
        "validation_metrics": None,
        "user_feedback": None,
        "needs_revision": False,
        "iteration": 0
    }

    try:
        llm = OllamaLLM(model=ollama_model, base_url=ollama_base_url)

        from langgraph.graph import StateGraph

        def validate_only(s):
            metrics = {"coherence_scores": [], "distinctiveness_scores": [], "overall_quality": 0.0}

            for idx, label in enumerate(s.get("user_selected_labels", [])):
                if idx < len(s.get("topic_terms", [])):
                    terms_str = ", ".join(s["topic_terms"][idx][:10])
                    prompt = f"""Evaluate this topic label:
Label: "{label}"
Terms: {terms_str}
Rate coherence and distinctiveness (0-10 each) as JSON: {{"coherence": X, "distinctiveness": Y, "explanation": "..."}}"""

                    try:
                        response = llm.invoke(prompt)
                        import re, json
                        json_match = re.search(r'\{.*\}', response, re.DOTALL)
                        if json_match:
                            eval_result = json.loads(json_match.group())
                            metrics["coherence_scores"].append(float(eval_result.get("coherence", 7.0)))
                            metrics["distinctiveness_scores"].append(float(eval_result.get("distinctiveness", 7.0)))
                    except Exception:
                        metrics["coherence_scores"].append(7.0)
                        metrics["distinctiveness_scores"].append(7.0)

            if metrics["coherence_scores"]:
                metrics["overall_quality"] = (sum(metrics["coherence_scores"]) + sum(metrics["distinctiveness_scores"])) / (2 * len(metrics["coherence_scores"]))

            return {"validation_metrics": metrics}

        result = validate_only(state)

        return {
            "success": True,
            "validation_metrics": result.get("validation_metrics")
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "validation_metrics": None
        }
