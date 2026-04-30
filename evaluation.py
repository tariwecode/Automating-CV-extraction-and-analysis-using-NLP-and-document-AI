import math
import re
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_WORD_RE = re.compile(r"[A-Za-z0-9_]+", re.U)

def _tokens(s: str) -> List[str]:
    return _WORD_RE.findall((s or "").lower())

def _ngram_counts(tokens: List[str], n: int):
    from collections import Counter
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

def compute_bleu_score(hypothesis: str, reference: str) -> float:
    """
    Simple smoothed BLEU-2:
      - unigrams & bigrams with add-one smoothing
      - brevity penalty
    Returns a float in [0,1].
    """
    hyp = _tokens(hypothesis)
    ref = _tokens(reference)
    if not hyp or not ref:
        return 0.0

    precisions = []
    for n in (1, 2):
        hyp_counts = _ngram_counts(hyp, n)
        ref_counts = _ngram_counts(ref, n)
        overlap = sum(min(c, ref_counts.get(ng, 0)) for ng, c in hyp_counts.items())
        total = max(1, sum(hyp_counts.values()))
        precisions.append((overlap + 1) / (total + 1))  # add-one smoothing

    c = len(hyp); r = len(ref)
    bp = math.exp(1 - r / c) if c < r else 1.0
    score = bp * math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    return float(score)

def embedding_similarity(text_a: str, text_b: str) -> float:
    """
    TF-IDF cosine similarity (no torch).
    Returns a float in [-1,1], typically [0,1] for non-empty.
    """
    vect = TfidfVectorizer(max_features=4096, ngram_range=(1, 2))
    X = vect.fit_transform([(text_a or ""), (text_b or "")])
    return float(cosine_similarity(X[0], X[1])[0, 0])

def evaluate_skills(found_skills: List[str], jd_skills: List[str]) -> Dict[str, float]:
    """
    Precision/Recall/F1 for skills overlap.
    """
    set_found = {s.strip().lower() for s in (found_skills or []) if s}
    set_jd = {s.strip().lower() for s in (jd_skills or []) if s}
    tp = len(set_found & set_jd)
    fp = len(set_found - set_jd)
    fn = len(set_jd - set_found)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1_score": f1}
