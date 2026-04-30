import re

class QueryRouter:
    def __init__(self, threshold: int = 50):
        self.threshold = threshold

    def get_complexity(self, query: str) -> int:
        # Simple complexity heuristic:
        # 1. Length of query
        # 2. Presence of complex keywords
        score = len(query.split())
        complex_keywords = [
            "explain", "jelaskan", "jelas", 
            "compare", "banding", "perbandingan", 
            "analyze", "analis", "analisa", "analisis",
            "why", "mengapa", "kenapa",
            "how", "bagaimana", "gimana",
            "evaluate", "evaluasi",
            "summarize", "rangkum", "ringkas", "simpul"
        ]
        for word in complex_keywords:
            if word in query.lower():
                score += 25  # Naikkan bobot sedikit
        return score

    def route(self, query: str) -> str:
        score = self.get_complexity(query)
        if score < self.threshold:
            return "local"
        else:
            return "cloud"
