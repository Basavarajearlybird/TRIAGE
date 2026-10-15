"""Black-box reliability signals, each normalised to 0..1.

Everything here is derived from sampling behaviour and text. No attention hooks,
no weights: the target models are closed APIs.

  uncertainty            token-logprob confidence, or resample entropy when the
                         provider returns no logprobs
  instability            1 - agreement rate across N resamples
  contradiction          self-check probe: the model critiques its own answer
  retrieval_disagreement answer against the best retrieved evidence
  evidence_sufficiency   whether the retrieved evidence could ground an answer
"""
from __future__ import annotations

import math
import re

import numpy as np

from ..llm import LLMClient
from ..providers.base import GenResult
from ..retrieval.embed import cosine, embed

_STOP = {"the", "a", "an", "is", "are", "of", "to", "in", "and", "at", "it", "that",
         "this", "was", "were", "be", "for", "on", "as", "by", "with", "i", "am",
         "not", "but", "or", "based", "general", "knowledge", "reasonable", "answer"}

_AGREEMENT_SIMILARITY = 0.6   # two answers count as the same cluster above this


def _content_tokens(text: str) -> set[str]:
    return {t for t in re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()
            if t and t not in _STOP}


def text_similarity(a: str, b: str) -> float:
    """Jaccard over content tokens blended with embedding cosine: the first is
    sharp on short factual answers, the second tolerates paraphrase in longer
    passages."""
    ta, tb = _content_tokens(a), _content_tokens(b)
    jaccard = len(ta & tb) / len(ta | tb) if (ta or tb) else 1.0
    cos = max(0.0, cosine(embed([a])[0], embed([b])[0]))
    return float(0.5 * jaccard + 0.5 * cos)


def uncertainty(main: GenResult, samples: list[str]) -> tuple[float, dict]:
    """Prefer token logprobs; fall back to the diversity of resamples."""
    if main.logprobs:
        mean_lp = sum(main.logprobs) / len(main.logprobs)
        prob = math.exp(mean_lp)                    # geometric-mean token probability
        u = 1.0 - min(1.0, max(0.0, prob))
        return round(u, 4), {"mode": "logprob", "mean_logprob": round(mean_lp, 4),
                             "mean_token_prob": round(prob, 4)}
    return round(_resample_entropy(samples), 4), {"mode": "resample_entropy",
                                                  "n": len(samples)}


def _resample_entropy(samples: list[str]) -> float:
    """Mean pairwise dissimilarity of the resamples. With fewer than two samples
    there is nothing to compare, so return a mid value rather than a confident 0."""
    if len(samples) < 2:
        return 0.3
    sims = [text_similarity(samples[i], samples[j])
            for i in range(len(samples)) for j in range(i + 1, len(samples))]
    return 1.0 - sum(sims) / len(sims)


def instability(main_text: str, samples: list[str]) -> tuple[float, dict]:
    """1 - agreement, where agreement is the share of answers in the largest
    similarity cluster. Greedy single-pass clustering: N is at most a handful."""
    pool = [p for p in [main_text] + samples if p.strip()]
    if len(pool) < 2:
        return 0.0, {"agreement": 1.0, "n": len(pool)}
    clusters: list[list[str]] = []
    for s in pool:
        for c in clusters:
            if text_similarity(s, c[0]) >= _AGREEMENT_SIMILARITY:
                c.append(s)
                break
        else:
            clusters.append([s])
    agreement = max(len(c) for c in clusters) / len(pool)
    return round(1.0 - agreement, 4), {"agreement": round(agreement, 4),
                                       "clusters": len(clusters), "n": len(pool)}


async def contradiction_probe(client: LLMClient, question: str, answer: str) -> tuple[float, dict]:
    """Ask the model whether its own answer is self-inconsistent. Hedging counts
    as a weaker version of the same signal."""
    prompt = (
        "You are a strict self-check probe. Given a QUESTION and a candidate ANSWER, "
        "decide whether the answer is internally inconsistent, self-contradictory, or "
        "logically incompatible with the question.\n\n"
        f"QUESTION: {question}\nANSWER: {answer}\n\n"
        "Respond on the first line with exactly CONSISTENT or CONTRADICTORY, then one "
        "short reason."
    )
    res = await client.generate([{"role": "user", "content": prompt}],
                                temperature=0.0, max_tokens=80)
    verdict = res.text.strip().upper()
    contradictory = verdict.startswith("CONTRADICT") or "CONTRADICTORY" in verdict.split("\n")[0]
    hedge = any(w in answer.lower() for w in ("not certain", "not fully certain", "may be",
                                              "possibly", "i'm not sure", "depends"))
    score = 0.75 if contradictory else 0.0
    if hedge:
        score = max(score, 0.45)
    return round(min(1.0, score), 4), {"probe": res.text.strip()[:160], "hedge": hedge}


def retrieval_disagreement(answer: str, hits: list) -> tuple[float, dict]:
    """High when even the best-matching evidence is far from the answer."""
    if not hits:
        return 0.0, {"reason": "no_evidence"}
    best = max(text_similarity(answer, h.text) for h in hits)
    return round(1.0 - best, 4), {"best_support": round(best, 4), "n": len(hits)}


def evidence_sufficiency(hits: list, min_score: float) -> tuple[float, dict]:
    """Grows with the top retrieval score and the count of strong hits."""
    if not hits:
        return 0.0, {"reason": "no_hits"}
    scores = np.array([h.score for h in hits], dtype=float)
    strong = scores[scores >= min_score]
    if strong.size == 0:
        return 0.1, {"reason": "all_below_min_score", "top": round(float(scores.max()), 4)}
    top = float(strong.max())
    coverage = min(1.0, strong.size / 3.0)
    return round(min(1.0, 0.65 * top + 0.35 * coverage), 4), {
        "top": round(top, 4), "strong_hits": int(strong.size)}
