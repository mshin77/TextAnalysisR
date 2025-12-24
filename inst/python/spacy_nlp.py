"""
Comprehensive spaCy NLP Module for TextAnalysisR

This module provides full access to spaCy's linguistic features including:
- Tokenization with all token attributes
- Part-of-speech tagging (universal and fine-grained)
- Lemmatization
- Named Entity Recognition
- Dependency parsing
- Morphological analysis
- Sentence segmentation
- Word vectors (when available)

Usage (Python):
    from spacy_nlp import SpacyNLP
    nlp = SpacyNLP()
    result = nlp.parse_texts(["The quick brown fox jumps."])

Usage (R via reticulate):
    spacy_nlp <- reticulate::import_from_path("spacy_nlp", path = system.file("python", package = "TextAnalysisR"))
    nlp <- spacy_nlp$SpacyNLP()
    result <- nlp$parse_texts(c("The quick brown fox jumps."))

Author: TextAnalysisR
License: MIT
"""

import spacy
from typing import List, Dict, Any, Optional, Union
import warnings


class SpacyNLP:
    """
    Comprehensive spaCy wrapper providing access to all linguistic features.

    Attributes:
        nlp: The loaded spaCy language model
        model_name: Name of the loaded model
    """

    def __init__(self, model: str = "en_core_web_sm"):
        """
        Initialize the spaCy NLP processor.

        Args:
            model: spaCy model name. Options include:
                - "en_core_web_sm" (small, fast, no word vectors)
                - "en_core_web_md" (medium, includes word vectors)
                - "en_core_web_lg" (large, better accuracy, word vectors)
                - "en_core_web_trf" (transformer-based, best accuracy)
        """
        try:
            self.nlp = spacy.load(model)
            self.model_name = model
        except OSError:
            raise RuntimeError(
                f"spaCy model '{model}' not found. Install with:\n"
                f"  python -m spacy download {model}"
            )

    def parse_texts(
        self,
        texts: List[str],
        include_pos: bool = True,
        include_lemma: bool = True,
        include_entity: bool = True,
        include_dependency: bool = True,
        include_morphology: bool = True,
        include_vectors: bool = False,
        batch_size: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Parse multiple texts and extract all linguistic features.

        Args:
            texts: List of text strings to parse
            include_pos: Include part-of-speech tags
            include_lemma: Include lemmatized forms
            include_entity: Include named entity recognition
            include_dependency: Include dependency parsing
            include_morphology: Include morphological features
            include_vectors: Include word vectors (requires md/lg model)
            batch_size: Batch size for processing

        Returns:
            List of dictionaries, one per document, containing:
                - doc_id: Document identifier
                - tokens: List of token dictionaries with all features
                - sentences: List of sentence boundaries
                - entities: List of named entities (span-level)
        """
        results = []

        for doc_id, doc in enumerate(self.nlp.pipe(texts, batch_size=batch_size)):
            doc_result = {
                "doc_id": f"text{doc_id + 1}",
                "text": doc.text,
                "tokens": [],
                "sentences": [],
                "entities": []
            }

            # Extract sentences
            for sent_id, sent in enumerate(doc.sents):
                doc_result["sentences"].append({
                    "sentence_id": sent_id + 1,
                    "start": sent.start,
                    "end": sent.end,
                    "text": sent.text
                })

            # Extract tokens with all features
            current_sent_id = 1
            for sent_id, sent in enumerate(doc.sents):
                for token in sent:
                    token_data = {
                        "token_id": token.i + 1,
                        "sentence_id": sent_id + 1,
                        "token": token.text,
                        "whitespace": token.whitespace_,
                        "is_punct": token.is_punct,
                        "is_space": token.is_space,
                        "is_stop": token.is_stop,
                        "is_alpha": token.is_alpha,
                        "is_digit": token.is_digit,
                    }

                    if include_pos:
                        token_data["pos"] = token.pos_  # Universal POS
                        token_data["tag"] = token.tag_  # Fine-grained tag

                    if include_lemma:
                        token_data["lemma"] = token.lemma_

                    if include_entity:
                        token_data["entity"] = token.ent_type_ if token.ent_type_ else ""
                        token_data["entity_iob"] = token.ent_iob_  # B, I, O

                    if include_dependency:
                        token_data["dep_rel"] = token.dep_  # Dependency relation
                        token_data["head_token_id"] = token.head.i + 1  # Head token
                        token_data["head_text"] = token.head.text

                    if include_morphology:
                        # Full morphological features as string
                        token_data["morph"] = str(token.morph)
                        # Parse individual morphological features
                        morph_dict = token.morph.to_dict()
                        for key, value in morph_dict.items():
                            token_data[f"morph_{key}"] = value

                    if include_vectors and token.has_vector:
                        token_data["has_vector"] = True
                        token_data["vector_norm"] = float(token.vector_norm)
                        # Don't include full vector by default (too large)
                    else:
                        token_data["has_vector"] = False

                    doc_result["tokens"].append(token_data)

            # Extract span-level entities
            for ent in doc.ents:
                doc_result["entities"].append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                    "start_token": ent.start,
                    "end_token": ent.end
                })

            results.append(doc_result)

        return results

    def parse_to_dataframe(
        self,
        texts: List[str],
        include_pos: bool = True,
        include_lemma: bool = True,
        include_entity: bool = True,
        include_dependency: bool = True,
        include_morphology: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Parse texts and return token-level data in a flat format suitable for DataFrames.

        This format is compatible with R data.frames and matches spacyr output format.

        Args:
            texts: List of text strings to parse
            include_pos: Include part-of-speech tags
            include_lemma: Include lemmatized forms
            include_entity: Include named entity recognition
            include_dependency: Include dependency parsing
            include_morphology: Include morphological features

        Returns:
            List of dictionaries (one per token) with columns:
                - doc_id, sentence_id, token_id, token
                - pos, tag (if include_pos)
                - lemma (if include_lemma)
                - entity, entity_iob (if include_entity)
                - dep_rel, head_token_id (if include_dependency)
                - morph, morph_* (if include_morphology)
        """
        rows = []

        for doc_id, doc in enumerate(self.nlp.pipe(texts)):
            doc_name = f"text{doc_id + 1}"

            for sent_id, sent in enumerate(doc.sents):
                for token in sent:
                    row = {
                        "doc_id": doc_name,
                        "sentence_id": sent_id + 1,
                        "token_id": token.i - sent.start + 1,  # Position within sentence
                        "token": token.text,
                    }

                    if include_pos:
                        row["pos"] = token.pos_
                        row["tag"] = token.tag_

                    if include_lemma:
                        row["lemma"] = token.lemma_

                    if include_entity:
                        row["entity"] = token.ent_type_ if token.ent_type_ else ""
                        row["entity_iob"] = token.ent_iob_

                    if include_dependency:
                        row["dep_rel"] = token.dep_
                        row["head_token_id"] = token.head.i - sent.start + 1

                    if include_morphology:
                        row["morph"] = str(token.morph)
                        # Add individual morph features as separate columns
                        morph_dict = token.morph.to_dict()
                        for key, value in morph_dict.items():
                            row[f"morph_{key}"] = value

                    rows.append(row)

        return rows

    def extract_entities(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Extract named entities at the span level.

        Args:
            texts: List of text strings

        Returns:
            List of entity dictionaries with:
                - doc_id, text, label, start_char, end_char
        """
        entities = []

        for doc_id, doc in enumerate(self.nlp.pipe(texts)):
            doc_name = f"text{doc_id + 1}"
            for ent in doc.ents:
                entities.append({
                    "doc_id": doc_name,
                    "text": ent.text,
                    "label": ent.label_,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char
                })

        return entities

    def extract_noun_chunks(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Extract noun chunks (base noun phrases).

        Args:
            texts: List of text strings

        Returns:
            List of noun chunk dictionaries
        """
        chunks = []

        for doc_id, doc in enumerate(self.nlp.pipe(texts)):
            doc_name = f"text{doc_id + 1}"
            for chunk in doc.noun_chunks:
                chunks.append({
                    "doc_id": doc_name,
                    "text": chunk.text,
                    "root_text": chunk.root.text,
                    "root_pos": chunk.root.pos_,
                    "root_dep": chunk.root.dep_
                })

        return chunks

    def get_sentence_vectors(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Get document/sentence vectors (requires md/lg model).

        Args:
            texts: List of text strings

        Returns:
            List of dictionaries with doc_id and vector
        """
        if not self.nlp.vocab.vectors.shape[0]:
            warnings.warn(
                f"Model '{self.model_name}' has no word vectors. "
                "Use 'en_core_web_md' or 'en_core_web_lg' for vectors."
            )
            return []

        results = []
        for doc_id, doc in enumerate(self.nlp.pipe(texts)):
            results.append({
                "doc_id": f"text{doc_id + 1}",
                "vector": doc.vector.tolist(),
                "has_vector": doc.has_vector
            })

        return results

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.

        Requires a model with word vectors (md/lg).

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        doc1 = self.nlp(text1)
        doc2 = self.nlp(text2)
        return doc1.similarity(doc2)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model metadata
        """
        return {
            "model_name": self.model_name,
            "lang": self.nlp.lang,
            "pipeline": [pipe[0] for pipe in self.nlp.pipeline],
            "has_vectors": self.nlp.vocab.vectors.shape[0] > 0,
            "vector_dim": self.nlp.vocab.vectors.shape[1] if self.nlp.vocab.vectors.shape[0] > 0 else 0
        }


# Convenience function for quick parsing
def parse_text(
    text: str,
    model: str = "en_core_web_sm",
    include_morphology: bool = True
) -> List[Dict[str, Any]]:
    """
    Quick function to parse a single text with all features.

    Args:
        text: Text string to parse
        model: spaCy model to use
        include_morphology: Include morphological features

    Returns:
        List of token dictionaries
    """
    nlp = SpacyNLP(model)
    return nlp.parse_to_dataframe([text], include_morphology=include_morphology)


if __name__ == "__main__":
    # Example usage
    nlp = SpacyNLP("en_core_web_sm")

    texts = [
        "Apple Inc. was founded by Steve Jobs in California.",
        "The quick brown fox jumps over the lazy dog."
    ]

    # Get dataframe-style output
    result = nlp.parse_to_dataframe(texts)

    print("Token-level analysis:")
    for row in result[:5]:
        print(row)

    print("\nModel info:")
    print(nlp.get_model_info())
