"""
Unified spaCy NLP module for TextAnalysisR

Provides direct spaCy access via reticulate with all linguistic features
including noun chunks, subject/object extraction, and word similarity
with graceful fallback.

Author: TextAnalysisR package
"""

import spacy
from spacy import displacy
from typing import List, Dict, Optional, Any, Tuple
import warnings
import pandas as pd


class SpacyNLP:
    """
    Unified spaCy NLP interface for TextAnalysisR.

    Provides:
    - Token-level parsing (POS, lemma, morph, dependency)
    - Named entity recognition
    - Noun chunk extraction (keyphrase extraction)
    - Subject/object extraction (SVO analysis)
    - Word similarity (with graceful fallback for models without vectors)
    - displaCy visualization
    """

    # Class-level model cache
    _model_cache: Dict[str, Any] = {}

    def __init__(self, model: str = "en_core_web_sm"):
        """
        Initialize the SpacyNLP instance with the specified model.

        Args:
            model: spaCy model name (default: "en_core_web_sm")
        """
        self.model_name = model

        # Use cached model if available
        if model in SpacyNLP._model_cache:
            self.nlp = SpacyNLP._model_cache[model]
        else:
            try:
                self.nlp = spacy.load(model)
                SpacyNLP._model_cache[model] = self.nlp
            except OSError as e:
                raise RuntimeError(
                    f"spaCy model '{model}' not found. "
                    f"Install with: python -m spacy download {model}"
                ) from e

        # Check if model has word vectors
        self._has_vectors = self.nlp.vocab.vectors.shape[0] > 0

    def has_vectors(self) -> bool:
        """Check if the loaded model has word vectors."""
        return self._has_vectors

    def parse_to_dataframe(
        self,
        texts: List[str],
        include_pos: bool = True,
        include_tag: bool = True,
        include_lemma: bool = True,
        include_entity: bool = False,
        include_dependency: bool = False,
        include_morph: bool = False,
        batch_size: int = 50,
        progress_callback: Optional[callable] = None
    ) -> pd.DataFrame:
        """
        Parse texts and return token-level annotations as list of dicts.

        This is the main parsing function for TextAnalysisR.
        Returns data in the same format expected by R.

        Uses nlp.pipe() for efficient batch processing (2-10x faster than
        processing texts individually).

        Args:
            texts: List of text strings to parse
            include_pos: Include coarse POS tags (NOUN, VERB, etc.)
            include_tag: Include fine-grained tags (NN, VBD, etc.)
            include_lemma: Include lemmatized forms
            include_entity: Include named entity tags (IOB format)
            include_dependency: Include dependency relations
            include_morph: Include morphological features
            batch_size: Batch size for nlp.pipe() (default: 50)
            progress_callback: Optional callback(current, total) for progress tracking

        Returns:
            pandas DataFrame with token-level annotations
        """
        results = []
        total = len(texts)

        # Use nlp.pipe() for efficient batch processing
        for doc_idx, doc in enumerate(self.nlp.pipe(texts, batch_size=batch_size)):
            # Report progress if callback provided
            if progress_callback and (doc_idx % 10 == 0 or doc_idx == total - 1):
                progress_callback(doc_idx + 1, total)
            doc_id = f"text{doc_idx + 1}"

            # Build sentence start index map for efficient head lookup
            if include_dependency:
                sent_starts = {sent.start: sent_idx for sent_idx, sent in enumerate(doc.sents)}

            for sent_idx, sent in enumerate(doc.sents):
                sent_start = sent.start  # Token index offset for this sentence

                for token_idx, token in enumerate(sent):
                    row = {
                        "doc_id": doc_id,
                        "sentence_id": sent_idx + 1,
                        "token_id": token_idx + 1,
                        "token": token.text
                    }

                    if include_pos:
                        row["pos"] = token.pos_

                    if include_tag:
                        row["tag"] = token.tag_

                    if include_lemma:
                        row["lemma"] = token.lemma_

                    if include_entity:
                        # Format: "ENTITY_IOB" or "" for non-entities
                        if token.ent_type_:
                            row["entity"] = f"{token.ent_type_}_{token.ent_iob_}"
                        else:
                            row["entity"] = ""

                    if include_dependency:
                        # Efficient head lookup using token indices
                        head_idx = token.head.i - sent_start + 1
                        row["head_token_id"] = head_idx if 0 < head_idx <= len(sent) else 0
                        row["dep_rel"] = token.dep_

                    if include_morph:
                        # Return as string in Universal Dependencies format
                        row["morph"] = str(token.morph)

                    results.append(row)

        return pd.DataFrame(results)

    def get_entities(self, texts: List[str], batch_size: int = 50,
                      progress_callback: Optional[callable] = None) -> pd.DataFrame:
        """
        Extract named entities from texts.

        Args:
            texts: List of text strings
            batch_size: Batch size for nlp.pipe()
            progress_callback: Optional callback(current, total) for progress tracking

        Returns:
            DataFrame with entity information
        """
        results = []
        total = len(texts)

        # Use nlp.pipe() for efficient batch processing
        for doc_idx, doc in enumerate(self.nlp.pipe(texts, batch_size=batch_size)):
            if progress_callback and (doc_idx % 10 == 0 or doc_idx == total - 1):
                progress_callback(doc_idx + 1, total)
            doc_id = f"text{doc_idx + 1}"

            for ent in doc.ents:
                results.append({
                    "doc_id": doc_id,
                    "text": ent.text,
                    "label": ent.label_,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                    "start_token": ent.start,
                    "end_token": ent.end
                })

        return pd.DataFrame(results)

    def get_noun_chunks(self, texts: List[str], batch_size: int = 50,
                         progress_callback: Optional[callable] = None) -> pd.DataFrame:
        """
        Extract noun chunks (keyphrases) from texts.

        Noun chunks are "base noun phrases" - flat phrases with a noun head.
        Examples: "the quick brown fox", "students with learning disabilities"

        Args:
            texts: List of text strings
            batch_size: Batch size for nlp.pipe()
            progress_callback: Optional callback(current, total) for progress tracking

        Returns:
            DataFrame with noun chunk information
        """
        results = []
        total = len(texts)

        # Use nlp.pipe() for efficient batch processing
        for doc_idx, doc in enumerate(self.nlp.pipe(texts, batch_size=batch_size)):
            if progress_callback and (doc_idx % 10 == 0 or doc_idx == total - 1):
                progress_callback(doc_idx + 1, total)
            doc_id = f"text{doc_idx + 1}"

            for chunk_idx, chunk in enumerate(doc.noun_chunks):
                results.append({
                    "doc_id": doc_id,
                    "chunk_id": chunk_idx + 1,
                    "text": chunk.text,
                    "root": chunk.root.text,
                    "root_pos": chunk.root.pos_,
                    "root_dep": chunk.root.dep_,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char
                })

        return pd.DataFrame(results)

    def get_subjects_objects(self, texts: List[str], batch_size: int = 50,
                               progress_callback: Optional[callable] = None) -> pd.DataFrame:
        """
        Extract subject-verb-object (SVO) triples from texts.

        Identifies syntactic subjects and objects using dependency parsing.
        Useful for analyzing "who does what to whom" in education research.

        Args:
            texts: List of text strings
            batch_size: Batch size for nlp.pipe()
            progress_callback: Optional callback(current, total) for progress tracking

        Returns:
            DataFrame with SVO information
        """
        results = []
        total = len(texts)

        # Dependency labels for subjects and objects
        SUBJ_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"}
        OBJ_DEPS = {"dobj", "pobj", "iobj", "attr", "oprd"}

        # Use nlp.pipe() for efficient batch processing
        for doc_idx, doc in enumerate(self.nlp.pipe(texts, batch_size=batch_size)):
            if progress_callback and (doc_idx % 10 == 0 or doc_idx == total - 1):
                progress_callback(doc_idx + 1, total)
            doc_id = f"text{doc_idx + 1}"

            for sent_idx, sent in enumerate(doc.sents):
                # Find verbs and their subjects/objects
                for token in sent:
                    if token.pos_ == "VERB":
                        verb = token.text
                        verb_lemma = token.lemma_

                        subjects = []
                        objects = []

                        for child in token.children:
                            if child.dep_ in SUBJ_DEPS:
                                # Get the full subtree for compound subjects
                                subjects.append(self._get_span_text(child))
                            elif child.dep_ in OBJ_DEPS:
                                # Get the full subtree for compound objects
                                objects.append(self._get_span_text(child))

                        # Only add if we found at least a subject or object
                        if subjects or objects:
                            results.append({
                                "doc_id": doc_id,
                                "sentence_id": sent_idx + 1,
                                "sentence": sent.text,
                                "subject": "; ".join(subjects) if subjects else "",
                                "verb": verb,
                                "verb_lemma": verb_lemma,
                                "object": "; ".join(objects) if objects else ""
                            })

        return pd.DataFrame(results)

    def _get_span_text(self, token) -> str:
        """Get the text of a token and its dependents (subtree)."""
        # Get all tokens in subtree, sorted by position
        subtree_tokens = sorted(list(token.subtree), key=lambda t: t.i)
        return " ".join(t.text for t in subtree_tokens)

    def get_word_similarity(
        self,
        word1: str,
        word2: str
    ) -> Dict[str, Any]:
        """
        Calculate similarity between two words using word vectors.

        Args:
            word1: First word
            word2: Second word

        Returns:
            Dict with similarity score and metadata
        """
        if not self._has_vectors:
            return {
                "similarity": None,
                "word1": word1,
                "word2": word2,
                "word1_has_vector": False,
                "word2_has_vector": False,
                "error": f"Model '{self.model_name}' has no word vectors. "
                         "Use en_core_web_md or en_core_web_lg for word similarity."
            }

        doc = self.nlp(f"{word1} {word2}")
        token1, token2 = doc[0], doc[1]

        return {
            "similarity": float(token1.similarity(token2)),
            "word1": word1,
            "word2": word2,
            "word1_has_vector": token1.has_vector,
            "word2_has_vector": token2.has_vector,
            "error": None
        }

    def find_similar_words(
        self,
        word: str,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Find words most similar to the given word using word vectors.

        Uses spaCy's efficient most_similar() when available, with fallback
        to optimized numpy search for smaller vocabularies.

        Args:
            word: Target word
            top_n: Number of similar words to return

        Returns:
            DataFrame with similar words and their scores
        """
        if not self._has_vectors:
            return pd.DataFrame([{
                "error": f"Model '{self.model_name}' has no word vectors. "
                         "Use en_core_web_md or en_core_web_lg for word similarity."
            }])

        # Get the word's vector
        doc = self.nlp(word)
        if not doc[0].has_vector:
            return pd.DataFrame([{
                "error": f"Word '{word}' not in vocabulary or has no vector."
            }])

        word_vector = doc[0].vector

        try:
            import numpy as np

            # Use spaCy's optimized most_similar if available
            if hasattr(self.nlp.vocab.vectors, 'most_similar'):
                # Reshape vector for most_similar API
                queries = np.array([word_vector])
                # Get more results to filter later
                keys, _, scores = self.nlp.vocab.vectors.most_similar(
                    queries, n=top_n + 10
                )

                results = []
                for key_id, score in zip(keys[0], scores[0]):
                    if key_id >= 0:
                        similar_word = self.nlp.vocab.strings[key_id]
                        # Skip the query word itself
                        if similar_word.lower() != word.lower() and similar_word.isalpha():
                            results.append({
                                "word": similar_word,
                                "similarity": float(score),
                                "rank": len(results) + 1
                            })
                            if len(results) >= top_n:
                                break

                return pd.DataFrame(results) if results else pd.DataFrame([{"word": None, "similarity": None, "rank": None}])

            # Fallback: vectorized numpy search (still faster than loop)
            vectors = self.nlp.vocab.vectors.data
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            normalized = vectors / norms
            word_norm = word_vector / np.linalg.norm(word_vector)

            similarities = np.dot(normalized, word_norm)
            top_indices = np.argsort(similarities)[::-1][:top_n + 10]

            results = []
            for idx in top_indices:
                key_id = self.nlp.vocab.vectors.keys()[idx]
                similar_word = self.nlp.vocab.strings[key_id]
                if similar_word.lower() != word.lower() and similar_word.isalpha():
                    results.append({
                        "word": similar_word,
                        "similarity": float(similarities[idx]),
                        "rank": len(results) + 1
                    })
                    if len(results) >= top_n:
                        break

            return pd.DataFrame(results)

        except Exception as e:
            return pd.DataFrame([{"error": f"Error finding similar words: {str(e)}"}])

    def get_sentences(self, texts: List[str], batch_size: int = 50,
                        progress_callback: Optional[callable] = None) -> pd.DataFrame:
        """
        Segment texts into sentences.

        Args:
            texts: List of text strings
            batch_size: Batch size for nlp.pipe()
            progress_callback: Optional callback(current, total) for progress tracking

        Returns:
            DataFrame with sentence information
        """
        results = []
        total = len(texts)

        # Use nlp.pipe() for efficient batch processing
        for doc_idx, doc in enumerate(self.nlp.pipe(texts, batch_size=batch_size)):
            if progress_callback and (doc_idx % 10 == 0 or doc_idx == total - 1):
                progress_callback(doc_idx + 1, total)
            doc_id = f"text{doc_idx + 1}"

            for sent_idx, sent in enumerate(doc.sents):
                results.append({
                    "doc_id": doc_id,
                    "sentence_id": sent_idx + 1,
                    "text": sent.text,
                    "start_char": sent.start_char,
                    "end_char": sent.end_char,
                    "n_tokens": len(sent)
                })

        return pd.DataFrame(results)

    def render_displacy_ent(
        self,
        text: str,
        colors: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Render entity visualization as HTML.

        Args:
            text: Text to visualize
            colors: Optional dict mapping entity types to colors

        Returns:
            HTML string, or empty string if no entities found
        """
        doc = self.nlp(text)

        # Check for empty entities to avoid W006 warning
        if len(doc.ents) == 0:
            return ""

        options = {}
        if colors:
            options["colors"] = colors

        html = displacy.render(doc, style="ent", page=False, options=options)
        return html

    def render_displacy_dep(
        self,
        text: str,
        compact: bool = True,
        distance: int = 100,
        word_spacing: int = 25,
        arrow_spacing: int = 12
    ) -> str:
        """
        Render dependency visualization as SVG.

        Args:
            text: Text to visualize
            compact: Use compact mode
            distance: Distance between tokens
            word_spacing: Spacing between words
            arrow_spacing: Spacing between arrows

        Returns:
            SVG string
        """
        doc = self.nlp(text)

        options = {
            "compact": compact,
            "distance": distance,
            "word_spacing": word_spacing,
            "arrow_spacing": arrow_spacing
        }

        svg = displacy.render(doc, style="dep", page=False, options=options)
        return svg

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "lang": self.nlp.lang,
            "pipeline": list(self.nlp.pipe_names),
            "has_vectors": self._has_vectors,
            "vocab_size": len(self.nlp.vocab),
            "vectors_size": self.nlp.vocab.vectors.shape[0] if self._has_vectors else 0,
            "vectors_dim": self.nlp.vocab.vectors.shape[1] if self._has_vectors else 0
        }


# Convenience functions for direct use without class instantiation

_default_nlp: Optional[SpacyNLP] = None


def get_nlp(model: str = "en_core_web_sm") -> SpacyNLP:
    """Get or create a SpacyNLP instance."""
    global _default_nlp
    if _default_nlp is None or _default_nlp.model_name != model:
        _default_nlp = SpacyNLP(model)
    return _default_nlp


def parse_texts(texts: List[str], **kwargs) -> List[Dict]:
    """Parse texts using default model."""
    return get_nlp().parse_to_dataframe(texts, **kwargs)


def extract_noun_chunks(texts: List[str], model: str = "en_core_web_sm") -> List[Dict]:
    """Extract noun chunks from texts."""
    return SpacyNLP(model).get_noun_chunks(texts)


def extract_subjects_objects(texts: List[str], model: str = "en_core_web_sm") -> List[Dict]:
    """Extract subject-verb-object triples from texts."""
    return SpacyNLP(model).get_subjects_objects(texts)
