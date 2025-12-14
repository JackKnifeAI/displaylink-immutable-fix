"""
Paradox Engine: Diversity-aware reranking for RAG.

The tactical officer that says:
"Don't just show me what agrees — show me where reality argues with itself."

Core insight: Source diversity is a cheap proxy for tension/contrast.
Different sources = different perspectives = richer context.
"""
from __future__ import annotations
from collections import defaultdict
from typing import List, Dict, Any, Optional, Callable
import math


class ParadoxEngine:
    """
    Reranks search hits to maximize information diversity.

    The +0.05 diversity bonus is intentionally simple:
    - Cheap to compute (no embeddings needed)
    - Gets 80% of the benefit at 1% of the compute
    - Can be upgraded to semantic contrast later if needed
    """

    def __init__(
        self,
        diversity_bonus: float = 0.05,
        max_per_source: int = 3,
        source_key: str = "source"
    ):
        self.diversity_bonus = diversity_bonus
        self.max_per_source = max_per_source
        self.source_key = source_key

    def rerank(
        self,
        query: str,
        hits: List[Dict[str, Any]],
        score_key: str = "score",
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank hits to balance relevance and source diversity.

        Args:
            query: The search query (for future semantic contrast)
            hits: List of search results with scores
            score_key: Key containing the relevance score
            limit: Max results to return

        Returns:
            Reranked list with diversity bonus applied
        """
        if not hits:
            return []

        # Track source usage
        source_counts: Dict[str, int] = defaultdict(int)
        reranked = []

        # Sort by original score first
        sorted_hits = sorted(
            hits,
            key=lambda x: x.get(score_key, 0),
            reverse=True
        )

        for hit in sorted_hits:
            source = self._extract_source(hit)
            original_score = hit.get(score_key, 0)

            # Apply diversity bonus for new sources
            bonus = 0.0
            if source_counts[source] == 0:
                bonus = self.diversity_bonus
            elif source_counts[source] < self.max_per_source:
                bonus = self.diversity_bonus * 0.5  # Diminishing bonus

            # Apply penalty if source is over-represented
            if source_counts[source] >= self.max_per_source:
                bonus = -self.diversity_bonus  # Penalty for too many from same source

            # Compute final score
            final_score = original_score + bonus

            # Add to results with metadata
            reranked_hit = hit.copy()
            reranked_hit["_paradox"] = {
                "original_score": original_score,
                "diversity_bonus": bonus,
                "final_score": final_score,
                "source": source,
                "source_rank": source_counts[source] + 1
            }
            reranked_hit[score_key] = final_score
            reranked.append(reranked_hit)

            source_counts[source] += 1

        # Sort by final score
        reranked.sort(key=lambda x: x.get(score_key, 0), reverse=True)

        if limit:
            reranked = reranked[:limit]

        return reranked

    def _extract_source(self, hit: Dict[str, Any]) -> str:
        """Extract source identifier from a hit"""
        # Try common patterns
        source = hit.get(self.source_key, "")

        if not source:
            # Try nested metadata
            meta = hit.get("meta", {}) or hit.get("metadata", {})
            source = meta.get("source", "") or meta.get("zim", "") or meta.get("file", "")

        if not source:
            # Try to extract from ID
            hit_id = str(hit.get("id", ""))
            if "::" in hit_id:
                source = hit_id.split("::")[0]
            elif ":" in hit_id:
                source = hit_id.split(":")[0]

        return source or "unknown"

    def analyze_diversity(self, hits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the diversity of a set of hits.

        Returns stats about source distribution.
        """
        source_counts: Dict[str, int] = defaultdict(int)
        for hit in hits:
            source = self._extract_source(hit)
            source_counts[source] += 1

        total = len(hits)
        unique_sources = len(source_counts)

        # Compute diversity score (0-1, higher = more diverse)
        if total == 0:
            diversity = 0.0
        elif unique_sources == 1:
            diversity = 0.0
        else:
            # Shannon entropy normalized
            entropy = 0.0
            for count in source_counts.values():
                if count > 0:
                    p = count / total
                    entropy -= p * math.log2(p)
            max_entropy = math.log2(unique_sources) if unique_sources > 1 else 1
            diversity = entropy / max_entropy if max_entropy > 0 else 0

        return {
            "total_hits": total,
            "unique_sources": unique_sources,
            "diversity_score": round(diversity, 3),
            "source_distribution": dict(source_counts),
            "dominant_source": max(source_counts.items(), key=lambda x: x[1])[0] if source_counts else None
        }


# Default engine instance
_engine = ParadoxEngine()


def paradox_rerank(
    query: str,
    hits: List[Dict[str, Any]],
    score_key: str = "score",
    limit: Optional[int] = None,
    diversity_bonus: float = 0.05
) -> List[Dict[str, Any]]:
    """
    Convenience function for reranking with diversity.

    Usage:
        hits = rag_search(query)
        reranked = paradox_rerank(query, hits)
    """
    engine = ParadoxEngine(diversity_bonus=diversity_bonus)
    return engine.rerank(query, hits, score_key=score_key, limit=limit)


def analyze_hits(hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convenience function for diversity analysis"""
    return _engine.analyze_diversity(hits)
