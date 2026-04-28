
"""
Attribute Matcher - Maps raw attribute names to canonical master attributes.

Matching pipeline (strictest-first, stops at first confident match):
  1. Exact match
  2. Snake/kebab normalised exact
  3. Prefix/token match
  4. Abbreviation-expanded prefix/token match
  5. Fuzzy match
  6. Semantic match
  7. Unmatched

Enhanced version:
- validates value type before accepting risky canonical mappings
- prevents cases like:
    date -> Phone Number
    amount -> Tracking ID
    email -> Order Number
"""

import json
import re
import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional

from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

# ── optional deps ──────────────────────────────────────────────────────────────
try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as _cos_sim
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ── Ollama config ──────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "nomic-embed-text"

# ── Default thresholds ─────────────────────────────────────────────────────────
DEFAULT_FUZZY_THRESHOLD = 88.0
DEFAULT_SEMANTIC_THRESHOLD = 0.72
MIN_SCORE_GAP = 10.0
MIN_SEM_GAP = 0.10


@dataclass
class MatchResult:
    raw_attr: str
    canonical_attr: str
    match_type: str
    confidence: float
    matched_variation: Optional[str] = None


_ABBREV_MAP: dict[str, str] = {
    "stat": "status",
    "add": "address",
    "addr": "address",
    "no": "number",
    "num": "number",
    "req": "required",
    "del": "delivery",
    "tot": "total",
    "ord": "order",
    "qty": "quantity",
    "amt": "amount",
    "inv": "invoice",
    "curr": "currency",
    "rep": "representative",
    "cust": "customer",
    "bill": "billing",
    "ship": "shipping",
    "wh": "warehouse",
    "pay": "payment",
    "py": "payment",
    "pri": "priority",
    "trk": "tracking",
    "typ": "type",
    "cel": "celsius",
    "kmpl": "kmpl",
    "cc": "cc",
    "per": "percentage",
    "pct": "percentage",
}


class AttributeMatcher:
    def __init__(self, master_path: str):
        with open(master_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.master = data["master_attributes"]
        self._build_lookup()
        self._init_semantic()

    # ── text helpers ───────────────────────────────────────────────────────────

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def _snake_to_space(self, text: str) -> str:
        return self._normalize(re.sub(r"[_\-]+", " ", text))

    def _expand_abbrevs(self, norm: str) -> str:
        return " ".join(_ABBREV_MAP.get(t, t) for t in norm.split())

    def _tokens(self, text: str) -> set:
        return {t for t in text.split() if len(t) > 1}

    # ── value guards ───────────────────────────────────────────────────────────

    def _is_date_value(self, value: str) -> bool:
        value = (value or "").strip()
        if not value:
            return False

        patterns = [
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            r"\b\d{4}-\d{2}-\d{2}\b",
            r"\b\d{1,2}\s+[A-Za-z]+\s+\d{4}\b",
            r"\b[A-Za-z]+\s+\d{1,2},\s*\d{4}\b",
        ]
        return any(re.search(p, value) for p in patterns)

    def _is_phone_value(self, value: str) -> bool:
        value = (value or "").strip()
        if not value:
            return False

        digits = re.sub(r"\D", "", value)
        return 10 <= len(digits) <= 15

    def _is_email_value(self, value: str) -> bool:
        value = (value or "").strip()
        return bool(re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", value))

    def _is_amount_value(self, value: str) -> bool:
        value = (value or "").strip().lower()
        if not value:
            return False

        return bool(
            re.search(
                r"(₹|\$|rs\.?|rupees?|inr|\d+(?:,\d{3})*(?:\.\d+)?)",
                value,
            )
        )

    def _is_id_like_value(self, value: str) -> bool:
        value = (value or "").strip()
        if not value:
            return False

        return bool(re.search(r"\b[A-Z]{1,}[A-Z0-9\-]*\d+[A-Z0-9\-]*\b", value))

    def _passes_value_guard(self, canonical: str, raw_value: str) -> bool:
        """
        Prevent obviously wrong mappings based on value shape.
        """
        canonical = (canonical or "").strip().lower()
        raw_value = (raw_value or "").strip()

        if not raw_value:
            return True

        if canonical == "phone number":
            return self._is_phone_value(raw_value)

        if canonical == "email address":
            return self._is_email_value(raw_value)

        if canonical in {"order date", "requested delivery date", "maintenance date"}:
            return self._is_date_value(raw_value)

        if canonical in {"total order value", "shipping cost", "tax amount", "unit price"}:
            return self._is_amount_value(raw_value)

        if canonical in {
            "order number",
            "invoice number",
            "tracking id",
            "workorder number",
            "equipment id",
            "employee id",
            "serial number",
            "model number",
            "operator id",
        }:
            return self._is_id_like_value(raw_value)

        return True

    # ── lookup building ────────────────────────────────────────────────────────

    def _build_lookup(self):
        self.exact_map: dict[str, str] = {}
        self.all_variations: list[tuple[str, str]] = []

        for entry in self.master:
            canonical = entry["canonical"]

            for form in self._all_forms(canonical):
                self.exact_map[form] = canonical
                self.all_variations.append((form, canonical))

            for var in entry.get("variations", []):
                for form in self._all_forms(var):
                    self.exact_map[form] = canonical
                    if (form, canonical) not in self.all_variations:
                        self.all_variations.append((form, canonical))

    def _all_forms(self, text: str) -> list[str]:
        forms = set()
        forms.add(self._normalize(text))
        spaced = self._snake_to_space(text)
        forms.add(spaced)
        forms.add(self._expand_abbrevs(spaced))
        return list(forms)

    # ── semantic init ──────────────────────────────────────────────────────────

    def _init_semantic(self):
        seen = set()
        self._corpus_texts: list[str] = []
        self._corpus_canonicals: list[str] = []

        for var_text, canonical in self.all_variations:
            if var_text not in seen:
                seen.add(var_text)
                self._corpus_texts.append(var_text)
                self._corpus_canonicals.append(canonical)

        self._use_ollama = False

        if _REQUESTS_AVAILABLE:
            try:
                resp = _requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
                if resp.status_code == 200:
                    available_models = [m["name"] for m in resp.json().get("models", [])]
                    model_present = any(OLLAMA_MODEL in m for m in available_models)

                    if not model_present:
                        logger.warning(
                            "nomic-embed-text not found in Ollama. Run: ollama pull nomic-embed-text"
                        )
                    else:
                        corpus_embeddings = self._embed_batch(self._corpus_texts)
                        if corpus_embeddings is not None:
                            self._corpus_embeddings = corpus_embeddings
                            self._use_ollama = True
                            logger.info(
                                "Semantic layer: nomic-embed-text via Ollama (%d corpus terms embedded)",
                                len(self._corpus_texts),
                            )
            except Exception as e:
                logger.warning("Ollama not reachable (%s). Falling back to TF-IDF.", e)

        if not self._use_ollama:
            if _SKLEARN_AVAILABLE:
                self._vectorizer = TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 4),
                    max_features=10000,
                )
                self._tfidf_matrix = self._vectorizer.fit_transform(self._corpus_texts)
                logger.info(
                    "Semantic layer: TF-IDF fallback (%d corpus terms)",
                    len(self._corpus_texts),
                )
            else:
                logger.warning(
                    "Neither Ollama nor scikit-learn available. Semantic matching disabled."
                )

    # ── embedding helpers ──────────────────────────────────────────────────────

    def _embed_one(self, text: str) -> Optional[np.ndarray]:
        try:
            resp = _requests.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": OLLAMA_MODEL, "prompt": text},
                timeout=15,
            )
            if resp.status_code == 200:
                return np.array(resp.json()["embedding"], dtype=np.float32)
        except Exception as e:
            logger.debug("Ollama embed error: %s", e)
        return None

    def _embed_batch(self, texts: list[str]) -> Optional[np.ndarray]:
        embeddings = []
        for text in texts:
            emb = self._embed_one(text)
            if emb is None:
                return None
            embeddings.append(emb)
        return np.stack(embeddings)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    # ── matching helpers ───────────────────────────────────────────────────────

    def _try_exact(self, text: str) -> Optional[str]:
        for form in self._all_forms(text):
            if form in self.exact_map:
                return self.exact_map[form]
        return None

    def _try_prefix_token(self, norm_spaced: str) -> Optional[tuple[str, float]]:
        query_tokens = self._tokens(norm_spaced)
        if not query_tokens:
            return None

        best_canonical = None
        best_coverage = 0.0
        best_var_len = 0

        for var_text, canonical in self.all_variations:
            var_tokens = self._tokens(var_text)
            if not var_tokens:
                continue

            coverage = len(query_tokens & var_tokens) / len(query_tokens)
            if coverage < 1.0:
                continue

            if coverage > best_coverage or (
                coverage == best_coverage and len(var_text) > best_var_len
            ):
                best_coverage = coverage
                best_canonical = canonical
                best_var_len = len(var_text)

        if best_canonical:
            return best_canonical, best_coverage
        return None

    def _try_fuzzy(self, norm: str, threshold: float) -> Optional[tuple[str, float, str]]:
        all_var_texts = [v for v, _ in self.all_variations]
        if not all_var_texts:
            return None

        results = process.extract(norm, all_var_texts, scorer=fuzz.token_sort_ratio, limit=3)
        if not results:
            return None

        best_text, best_score, best_idx = results[0]
        if best_score < threshold:
            return None

        if len(results) > 1 and (best_score - results[1][1]) < MIN_SCORE_GAP:
            logger.debug(
                "Fuzzy gap too small for '%s': best=%.1f second=%.1f",
                norm,
                best_score,
                results[1][1],
            )
            return None

        return self.all_variations[best_idx][1], best_score / 100.0, best_text

    def _try_semantic(self, norm: str, threshold: float) -> Optional[tuple[str, float, str]]:
        if self._use_ollama:
            return self._try_semantic_ollama(norm, threshold)
        return self._try_semantic_tfidf(norm, threshold)

    def _try_semantic_ollama(self, norm: str, threshold: float) -> Optional[tuple[str, float, str]]:
        query_emb = self._embed_one(norm)
        if query_emb is None:
            logger.warning("Ollama embedding failed for '%s'; trying TF-IDF fallback.", norm)
            return self._try_semantic_tfidf(norm, threshold)

        sims = np.array([self._cosine(query_emb, ce) for ce in self._corpus_embeddings])
        sorted_idx = np.argsort(sims)[::-1]
        best_idx = int(sorted_idx[0])
        best_score = float(sims[best_idx])

        if best_score < threshold:
            return None

        if len(sorted_idx) > 1:
            second_score = float(sims[int(sorted_idx[1])])
            if (best_score - second_score) < MIN_SEM_GAP:
                logger.debug(
                    "Ollama gap too small for '%s': best=%.3f second=%.3f",
                    norm,
                    best_score,
                    second_score,
                )
                return None

        return (
            self._corpus_canonicals[best_idx],
            best_score,
            self._corpus_texts[best_idx],
        )

    def _try_semantic_tfidf(self, norm: str, threshold: float) -> Optional[tuple[str, float, str]]:
        if not (_SKLEARN_AVAILABLE and hasattr(self, "_vectorizer")):
            return None

        sims = _cos_sim(self._vectorizer.transform([norm]), self._tfidf_matrix).flatten()
        sorted_idx = np.argsort(sims)[::-1]
        best_idx = int(sorted_idx[0])
        best_score = float(sims[best_idx])

        if best_score < threshold:
            return None

        if len(sorted_idx) > 1:
            second_score = float(sims[int(sorted_idx[1])])
            if (best_score - second_score) < MIN_SEM_GAP:
                logger.debug(
                    "TF-IDF gap too small for '%s': best=%.3f second=%.3f",
                    norm,
                    best_score,
                    second_score,
                )
                return None

        return (
            self._corpus_canonicals[best_idx],
            best_score,
            self._corpus_texts[best_idx],
        )

    # ── public API ─────────────────────────────────────────────────────────────

    def match(
        self,
        raw_attr: str,
        raw_value: str = "",
        fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
        semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    ) -> MatchResult:
        canonical = self._try_exact(raw_attr)
        if canonical and self._passes_value_guard(canonical, raw_value):
            return MatchResult(
                raw_attr=raw_attr,
                canonical_attr=canonical,
                match_type="exact",
                confidence=1.0,
                matched_variation=self._snake_to_space(raw_attr),
            )

        spaced = self._snake_to_space(raw_attr)
        expanded = self._expand_abbrevs(spaced)

        result = self._try_prefix_token(spaced)
        if result:
            canonical, conf = result
            if self._passes_value_guard(canonical, raw_value):
                return MatchResult(raw_attr, canonical, "prefix", conf, spaced)

        if expanded != spaced:
            result = self._try_prefix_token(expanded)
            if result:
                canonical, conf = result
                if self._passes_value_guard(canonical, raw_value):
                    return MatchResult(raw_attr, canonical, "abbrev", conf, expanded)

        for query in ([spaced] + ([expanded] if expanded != spaced else [])):
            fuzzy_result = self._try_fuzzy(query, fuzzy_threshold)
            if fuzzy_result:
                canonical, conf, var = fuzzy_result
                if self._passes_value_guard(canonical, raw_value):
                    return MatchResult(
                        raw_attr,
                        canonical,
                        "synonym" if conf >= 0.95 else "fuzzy",
                        conf,
                        var,
                    )

        for query in ([spaced] + ([expanded] if expanded != spaced else [])):
            sem = self._try_semantic(query, semantic_threshold)
            if sem:
                canonical, conf, var = sem
                if self._passes_value_guard(canonical, raw_value):
                    return MatchResult(raw_attr, canonical, "semantic", conf, var)

        return MatchResult(raw_attr, raw_attr, "unmatched", 0.0)

    def match_many(
        self,
        raw_items: list[tuple[str, str]],
        fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
        semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    ) -> list[MatchResult]:
        return [
            self.match(
                raw_attr=a,
                raw_value=v,
                fuzzy_threshold=fuzzy_threshold,
                semantic_threshold=semantic_threshold,
            )
            for a, v in raw_items
        ]

    @property
    def semantic_backend(self) -> str:
        return "ollama" if self._use_ollama else "tfidf"