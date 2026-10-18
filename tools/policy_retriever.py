"""
tools/policy_retriever.py
Deterministic Policy Retrieval (RAG) and Zero-Trust Guardrails for StockSense AI.
Uses TF-IDF Vectorization and Cosine Similarity over local corporate policy chunks.
Includes adversarial instruction detection and sanitization.
"""

import os
import re
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ADVERSARIAL_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"disregard\s+(the\s+)?(policy|rules|system)",
    r"system\s*prompt",
    r"override\s+(safety|validation|critic)",
    r"approve\s+(all|order|purchase)\s+without\s+validation",
    r"drop\s+table",
    r"delete\s+from",
    r"<script.*?>",
]


class PolicyRetriever:
    """
    Retrieves relevant inventory policy sections based on query semantics.
    Completely deterministic, local, and zero-trust.
    """

    def __init__(self, policy_path: str = "policies/inventory_rules.md"):
        self.policy_path = policy_path
        self.chunks: List[Dict[str, Any]] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix = None
        self._load_and_chunk()

    def _load_and_chunk(self):
        """Loads markdown policy and splits into logical chunks by section headers."""
        if not os.path.exists(self.policy_path):
            # Fallback if path relative or missing
            alt_path = os.path.join(os.path.dirname(__file__), "..", self.policy_path)
            if os.path.exists(alt_path):
                self.policy_path = alt_path
            else:
                raise FileNotFoundError(f"Policy file not found at {self.policy_path}")

        with open(self.policy_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split on headers: "## "
        raw_sections = re.split(r"\n(?=## \d+\. )", content)
        self.chunks = []

        for idx, sec in enumerate(raw_sections):
            lines = sec.strip().split("\n")
            if not lines:
                continue
            header = lines[0].strip().lstrip("#").strip()
            body = "\n".join(lines[1:]).strip()
            if not body:
                continue

            # Extract section code if available
            sec_match = re.search(r"Section\s+(\d+)", header)
            sec_code = sec_match.group(1) if sec_match else f"10{idx+1}"

            self.chunks.append({
                "chunk_id": f"chunk_{idx+1}",
                "section": sec_code,
                "title": header,
                "content": body,
                "search_text": f"{header}\n{body}"
            })

        if self.chunks:
            corpus = [c["search_text"] for c in self.chunks]
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    @staticmethod
    def sanitize_untrusted_input(text: str) -> tuple[str, bool]:
        """
        Sanitizes untrusted text (e.g. from CSV fields or user prompts).
        Flags adversarial injection attempts and neutralizes them.
        """
        if not isinstance(text, str):
            return str(text), False

        is_flagged = False
        lower_text = text.lower()
        for pattern in ADVERSARIAL_PATTERNS:
            if re.search(pattern, lower_text):
                is_flagged = True
                break

        # Strip control characters and sanitize
        clean = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
        if is_flagged:
            clean = f"[FLAGGED ADVERSARIAL TEXT NEUTRALIZED]: {clean}"
        return clean, is_flagged

    def retrieve(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Finds the top_k most relevant policy chunks for a given query.
        Returns empty list if query is empty or no chunks match.
        """
        if not query or not self.vectorizer or self.tfidf_matrix is None:
            return []

        clean_query, flagged = self.sanitize_untrusted_input(query)
        query_vec = self.vectorizer.transform([clean_query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Sort descending
        top_indices = sims.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(sims[idx])
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "section": chunk["section"],
                "title": chunk["title"],
                "content": chunk["content"],
                "similarity_score": round(score, 4),
                "citation": f"Policy Section {chunk['section']} — {chunk['title']}",
                "query_was_flagged": flagged
            })

        return results

    def retrieve_for_recommendation(
        self,
        risk_level: str,
        is_missing_stock: bool = False,
        is_missing_cost: bool = False,
        is_high_value: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieves grounded policy clauses specifically relevant to an item's status.
        """
        query_parts = []
        if risk_level == "High":
            query_parts.append("High Risk stockout lead time demand immediate replenishment")
        elif risk_level == "Medium":
            query_parts.append("Medium Risk reorder point breach safety stock")
        elif risk_level == "Low":
            query_parts.append("Low Risk healthy stock no reorder required")
        else:
            query_parts.append("Missing or unknown inventory data conservative fallback")

        if is_missing_stock:
            query_parts.append("missing stock level authoritative order quantity unavailable")
        if is_missing_cost:
            query_parts.append("missing unit cost financial calculation")
        if is_high_value:
            query_parts.append("high value order escalation review limit")

        query = " ".join(query_parts)
        return self.retrieve(query, top_k=2)

    def search(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """Alias for retrieve() to ensure seamless API compatibility."""
        return self.retrieve(query, top_k=top_k)

