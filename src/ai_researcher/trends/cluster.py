"""Story clustering.

The dashboard's core claim is "this is one story, covered six times". Getting
there takes three passes, cheapest first:

1. exact identity  — same canonical URL or same normalised text
2. semantic        — cosine over embeddings, or hashed TF-IDF when absent
3. evidence gate   — near-duplicate text merges on its own; merely-similar
                     text merges only when both items name the same
                     organisation or model
"""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from datetime import timedelta
from typing import Any

import numpy as np

from ..db import Database, jdump, jload
from ..enrich.embed import load_vectors
from ..util import iso, local_day, parse_datetime, tokens, utcnow
from .score import explain_cluster, score_cluster, score_item

log = logging.getLogger("ai_researcher.cluster")

# Two thresholds per method, because similarity alone cannot carry this
# decision. Measured on real ingested data, nomic-embed-text puts unrelated
# AI-topic headlines around 0.70-0.75 and genuinely-same stories around
# 0.75-0.90 — the bands overlap, so a single cutoff either merges unrelated
# stories or splits real ones.
#
#   NEAR  — text is close to duplicate; safe to merge on its own.
#   BASE  — plausibly the same story, but only merged with corroborating
#           evidence (a shared named entity).
#
# Items with no extracted entities therefore only merge at NEAR, which is the
# case that used to produce nonsense pairings like an OpenAI announcement
# merging with an unrelated forum post at 0.75.
EMBED_NEAR = 0.88
EMBED_BASE = 0.74
TFIDF_NEAR = 0.55
TFIDF_BASE = 0.30
HASH_DIM = 2048


class _UnionFind:
    def __init__(self, keys):
        self.parent = {k: k for k in keys}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def groups(self) -> dict[Any, list]:
        out: dict[Any, list] = defaultdict(list)
        for key in self.parent:
            out[self.find(key)].append(key)
        return out


def _hashed_tfidf(docs: list[list[str]]) -> np.ndarray:
    """L2-normalised hashed TF-IDF. Used when no embedding model is installed."""
    n = len(docs)
    matrix = np.zeros((n, HASH_DIM), dtype=np.float32)
    doc_freq = Counter()
    for doc in docs:
        doc_freq.update(set(doc))

    for i, doc in enumerate(docs):
        if not doc:
            continue
        counts = Counter(doc)
        for term, tf in counts.items():
            idf = math.log((n + 1) / (doc_freq[term] + 1)) + 1.0
            matrix[i, hash(term) % HASH_DIM] += (1.0 + math.log(tf)) * idf
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def build_clusters(db: Database, *, window_hours: int = 48, day: str | None = None) -> dict[str, Any]:
    """Recompute clusters for the trailing window and persist them for `day`."""
    now = utcnow()
    day = day or local_day(now)
    cutoff = iso(now - timedelta(hours=window_hours))

    rows = db.query(
        """
        SELECT i.id, i.title, i.body, i.url, i.url_hash, i.content_hash,
               i.published_at, i.fetched_at, i.engagement, i.comments,
               i.source_key, i.meta,
               COALESCE(e.summary, '')    AS summary,
               COALESCE(e.category, '')   AS category,
               COALESCE(e.entities, '[]') AS entities,
               COALESCE(e.tags, '[]')     AS tags,
               COALESCE(e.importance, 0.5) AS importance,
               COALESCE(e.why, '')        AS why,
               COALESCE(s.weight, 1.0)    AS source_weight,
               COALESCE(s.tier, 'news')   AS tier,
               COALESCE(s.name, i.source_key) AS source_name
        FROM items i
        LEFT JOIN enrichment e ON e.item_id = i.id
        LEFT JOIN sources s ON s.key = i.source_key
        WHERE COALESCE(i.published_at, i.fetched_at) >= ?
          AND COALESCE(i.relevant, 1) != 0
        ORDER BY COALESCE(i.published_at, i.fetched_at) DESC
        """,
        (cutoff,),
    )
    if not rows:
        return {"clusters": 0, "items": 0, "method": "none"}

    items = [dict(r) for r in rows]
    for item in items:
        item["entities"] = jload(item["entities"], [])
        item["tags"] = jload(item["tags"], [])
        item["meta"] = jload(item["meta"], {})
        item["score"] = score_item(item, now=now)

    ids = [it["id"] for it in items]
    index = {it["id"]: pos for pos, it in enumerate(items)}
    uf = _UnionFind(ids)

    # ── pass 1: exact identity ───────────────────────────────────────
    by_url: dict[str, int] = {}
    by_content: dict[str, int] = {}
    for item in items:
        uh = item["url_hash"]
        if uh:
            if uh in by_url:
                uf.union(by_url[uh], item["id"])
            else:
                by_url[uh] = item["id"]
        ch = item["content_hash"]
        if ch:
            if ch in by_content:
                uf.union(by_content[ch], item["id"])
            else:
                by_content[ch] = item["id"]

    # ── pass 2: semantic similarity ──────────────────────────────────
    vectors = load_vectors(db, ids)
    coverage = len(vectors) / len(items) if items else 0.0
    if coverage < 0.9 and len(vectors) >= 8:
        log.warning(
            "only %.0f%% of items have embeddings; the rest cannot merge and "
            "will appear as single-source stories", coverage * 100,
        )
    if len(vectors) >= max(8, len(items) * 0.5):
        method = "embeddings"
        near, base = EMBED_NEAR, EMBED_BASE
        dim = next(iter(vectors.values())).size
        matrix = np.zeros((len(items), dim), dtype=np.float32)
        present = np.zeros(len(items), dtype=bool)
        for item_id, vec in vectors.items():
            pos = index[item_id]
            matrix[pos] = vec
            present[pos] = True
    else:
        method = "tfidf"
        near, base = TFIDF_NEAR, TFIDF_BASE
        docs = [tokens(f"{it['title']} {it['summary'] or it['body'][:400]}") for it in items]
        matrix = _hashed_tfidf(docs)
        present = np.array([bool(d) for d in docs])

    entity_sets = [set(it["entities"]) for it in items]
    sims = matrix @ matrix.T
    np.fill_diagonal(sims, 0.0)
    candidates = np.argwhere(sims >= base)

    merged_near = merged_entity = 0
    for i, j in candidates:
        if i >= j or not (present[i] and present[j]):
            continue
        sim = sims[i, j]
        ei, ej = entity_sets[i], entity_sets[j]
        shared = ei & ej

        # Disjoint subjects are never one story, however close the wording.
        # "Anthropic releases a new frontier model" and "Mistral releases a new
        # frontier model" are nearly identical as text and are two events.
        if ei and ej and not shared:
            continue

        if shared:
            # Merely similar, but both name the same organisation or model —
            # that shared subject is what makes it one story.
            merged_entity += 1
        elif sim >= near:
            # No entity evidence either way, so demand near-duplicate text: a
            # syndicated copy or a re-post.
            merged_near += 1
        else:
            continue
        uf.union(items[i]["id"], items[j]["id"])

    # ── persist ──────────────────────────────────────────────────────
    groups = uf.groups()
    with db.tx() as conn:
        conn.execute(
            "DELETE FROM cluster_items WHERE cluster_id IN (SELECT id FROM clusters WHERE day=?)",
            (day,),
        )
        conn.execute("DELETE FROM clusters WHERE day=?", (day,))

    created = 0
    for members in groups.values():
        group = [items[index[i]] for i in members]
        record = _summarise_group(group, now=now)
        with db.tx() as conn:
            cur = conn.execute(
                """
                INSERT INTO clusters (day, label, summary, category, score, size,
                                      source_count, first_seen, last_seen, entities,
                                      ranking_why, confidence, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    day, record["label"], record["summary"], record["category"],
                    record["score"], len(group), record["source_count"],
                    record["first_seen"], record["last_seen"],
                    jdump(record["entities"]), record["ranking_why"],
                    record["confidence"], iso(now),
                ),
            )
            cluster_id = cur.lastrowid
            primary_id = record["primary_id"]
            conn.executemany(
                "INSERT OR IGNORE INTO cluster_items (cluster_id, item_id, is_primary) VALUES (?,?,?)",
                [(cluster_id, it["id"], 1 if it["id"] == primary_id else 0) for it in group],
            )
        created += 1

    log.info(
        "clustered %s items into %s stories via %s (%s near-duplicate, %s shared-entity merges)",
        len(items), created, method, merged_near, merged_entity,
    )
    return {
        "clusters": created, "items": len(items), "method": method, "day": day,
        "merged_near": merged_near, "merged_entity": merged_entity,
    }


def _summarise_group(group: list[dict], *, now) -> dict[str, Any]:
    """Pick the group's representative item and roll up its metadata."""
    # The primary is the highest-scoring item, with a nudge toward primary
    # sources: a lab's own post beats a news write-up of that post.
    def primacy(item: dict) -> float:
        bonus = {"lab": 0.30, "vendor": 0.18, "research": 0.12, "analyst": 0.06}.get(item["tier"], 0.0)
        return item["score"] + bonus

    primary = max(group, key=primacy)

    entity_counts = Counter()
    tag_counts = Counter()
    category_weight: dict[str, float] = defaultdict(float)
    for item in group:
        entity_counts.update(item["entities"])
        tag_counts.update(item["tags"])
        if item["category"]:
            category_weight[item["category"]] += item["score"] + 0.1

    # The primary item is the artifact itself; everything else is commentary on
    # it. A model release discussed in five forum threads is still a model
    # release, so the primary's category gets a decisive share of the vote —
    # otherwise chatter about a thing out-votes the thing.
    if primary["category"]:
        category_weight[primary["category"]] += sum(category_weight.values()) * 0.6 + 0.5

    category = (
        max(category_weight.items(), key=lambda kv: kv[1])[0]
        if category_weight else primary["category"] or "opinion-analysis"
    )

    times = [parse_datetime(it["published_at"] or it["fetched_at"]) for it in group]
    times = [t for t in times if t]
    first_seen = iso(min(times)) if times else iso(now)
    last_seen = iso(max(times)) if times else iso(now)

    source_count = len({it["source_key"] for it in group})
    confidence = min(
        1.0,
        0.35
        + 0.12 * min(source_count, 4)
        + 0.12 * (1 if primary.get("tier") in ("lab", "vendor", "research") else 0)
        + 0.18 * min(1.0, float(primary.get("importance") or 0.5))
        + 0.08 * min(1.0, len(group) / 5),
    )

    return {
        "label": primary["title"],
        "summary": primary["summary"] or primary["why"] or "",
        "category": category,
        "score": round(score_cluster(group, now=now), 4),
        "source_count": source_count,
        "first_seen": first_seen,
        "last_seen": last_seen,
        "entities": [e for e, _ in entity_counts.most_common(6)],
        "tags": [t for t, _ in tag_counts.most_common(6)],
        "primary_id": primary["id"],
        "ranking_why": explain_cluster(group, now=now),
        "confidence": round(confidence, 3),
    }
