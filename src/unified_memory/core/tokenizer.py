import tiktoken
from tiktoken.model import MODEL_TO_ENCODING
_encoder = None

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Accurately estimate tokens using tiktoken."""
    global _encoder
    try:
        if _encoder is None:
            _encoder = tiktoken.encoding_for_model(model)
        return len(_encoder.encode(text))
    except Exception:
        # Fallback if model not supported by tiktoken
        return max(1, len(text) // 4)

class ContextWindowManager:
    def __init__(self, max_tokens: int = 4000, model: str = "gpt-4o"):
        self.max_tokens = max_tokens
        self.tokenizer = tiktoken.encoding_for_model(model if model in MODEL_TO_ENCODING else "gpt-4o")

    def fit_results(self, results: list, reserved_tokens: int = 500) -> list:
        """Return as many full results as fit in the context window."""
        available = self.max_tokens - reserved_tokens
        fitted = []
        used = 0
        for r in results:
            tokens = len(self.tokenizer.encode(r.content or ""))
            if used + tokens > available:
                break
            fitted.append(r)
            used += tokens
        return fitted
