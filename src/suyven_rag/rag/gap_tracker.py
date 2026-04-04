"""Knowledge Gap Tracker — detects recurring retrieval failures.

Analyzes query_log.jsonl to find patterns where the system consistently
fails to serve users well. No LLM calls — pure log analysis.

Usage:
    python -m rag.gap_tracker                  # full report
    python -m rag.gap_tracker --top 10         # top 10 gaps
    python -m rag.gap_tracker --since 7        # last 7 days only
    python -m rag.gap_tracker --json           # machine-readable output
"""

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import mean as _mean

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
QUERY_LOG = BASE_DIR / "data" / "eval" / "query_log.jsonl"

# Flags that indicate retrieval problems
_RETRIEVAL_ISSUE_FLAGS = {"empty_retrieval", "retrieval_failure", "weak_retrieval", "corpus_gap"}


@dataclass
class GapEntry:
    """A detected knowledge gap."""

    pattern: str  # normalized query pattern
    count: int  # how many times this gap was hit
    example_queries: list[str]  # up to 3 example queries
    flags: list[str]  # most common flags
    avg_reranker_score: float | None
    categories: list[str]  # categories these queries touched
    first_seen: str  # ISO timestamp
    last_seen: str  # ISO timestamp


@dataclass
class GapReport:
    """Full gap analysis report."""

    total_queries: int
    total_flagged: int
    flag_frequency: dict[str, int]
    gaps: list[GapEntry]
    top_missing_topics: list[tuple[str, int]]
    timestamp: str


def _normalize_query(query: str) -> str:
    """Normalize query for grouping similar queries."""
    q = query.lower().strip()
    q = re.sub(r"[?!.,;:'\"]", "", q)
    q = re.sub(r"\s+", " ", q)
    # Remove common prefixes
    q = re.sub(r"^(what is|how does|how do|how to|explain|describe|tell me about)\s+", "", q)
    q = re.sub(r"^(que es|como funciona|como se|explica|describe)\s+", "", q)
    return q.strip()


def _extract_topic(query: str) -> str:
    """Extract the main topic from a query for gap grouping."""
    norm = _normalize_query(query)
    # Take first 5 meaningful words
    words = [w for w in norm.split() if len(w) > 2]
    return " ".join(words[:5])


def load_query_log(since_days: int | None = None) -> list[dict]:
    """Load query log entries, optionally filtered by date."""
    if not QUERY_LOG.exists():
        return []

    entries = []
    cutoff = None
    if since_days is not None:
        cutoff = datetime.now(UTC) - timedelta(days=since_days)

    with open(QUERY_LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if cutoff:
                ts = entry.get("timestamp", "")
                try:
                    entry_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if entry_dt < cutoff:
                        continue
                except (ValueError, TypeError):
                    pass

            entries.append(entry)

    return entries


def analyze_gaps(entries: list[dict], top_n: int = 20) -> GapReport:
    """Analyze query log for knowledge gaps."""
    total = len(entries)
    flagged = [e for e in entries if set(e.get("flags", [])) & _RETRIEVAL_ISSUE_FLAGS]

    # Flag frequency
    flag_counter: Counter = Counter()
    for e in entries:
        for f in e.get("flags", []):
            flag_counter[f] += 1

    # Group flagged queries by topic
    topic_groups: dict[str, list[dict]] = defaultdict(list)
    for e in flagged:
        topic = _extract_topic(e.get("query", ""))
        if topic:
            topic_groups[topic].append(e)

    # Build gap entries, sorted by frequency
    gaps = []
    for topic, group in sorted(topic_groups.items(), key=lambda x: -len(x[1])):
        timestamps = [e.get("timestamp", "") for e in group]
        timestamps = [t for t in timestamps if t]

        all_flags: Counter = Counter()
        for e in group:
            for f in e.get("flags", []):
                if f in _RETRIEVAL_ISSUE_FLAGS:
                    all_flags[f] += 1

        scores = []
        for e in group:
            s = e.get("mean_reranker_score")
            if s is not None:
                scores.append(s)

        cats = set()
        for e in group:
            for c in e.get("source_categories", []):
                cats.add(c)

        gaps.append(
            GapEntry(
                pattern=topic,
                count=len(group),
                example_queries=list({e.get("query", "") for e in group})[:3],
                flags=[f for f, _ in all_flags.most_common(3)],
                avg_reranker_score=round(_mean(scores), 4) if scores else None,
                categories=sorted(cats),
                first_seen=min(timestamps) if timestamps else "",
                last_seen=max(timestamps) if timestamps else "",
            )
        )

    gaps.sort(key=lambda g: g.count, reverse=True)

    # Top missing topics (all flagged queries, broader grouping)
    topic_counter: Counter = Counter()
    for e in flagged:
        topic = _extract_topic(e.get("query", ""))
        if topic:
            topic_counter[topic] += 1

    return GapReport(
        total_queries=total,
        total_flagged=len(flagged),
        flag_frequency=dict(flag_counter.most_common()),
        gaps=gaps[:top_n],
        top_missing_topics=topic_counter.most_common(top_n),
        timestamp=datetime.now(UTC).isoformat(),
    )


def print_report(report: GapReport) -> None:
    """Print human-readable gap report."""
    print(f"\n{'=' * 62}")
    print("  Knowledge Gap Report")
    print(f"{'=' * 62}")
    print(f"\n  Total queries analyzed: {report.total_queries}")
    print(
        f"  Queries with issues:    {report.total_flagged} ({report.total_flagged / max(report.total_queries, 1):.0%})"
    )

    if report.flag_frequency:
        print("\n  Flag frequency:")
        for flag, count in report.flag_frequency.items():
            print(f"    {flag:<25s} {count:>4d}")

    if report.gaps:
        print(f"\n  {'=' * 58}")
        print("  Top Knowledge Gaps (recurring failures)")
        print(f"  {'=' * 58}")
        for i, gap in enumerate(report.gaps[:15], 1):
            score_str = (
                f"avg_score={gap.avg_reranker_score:.2f}"
                if gap.avg_reranker_score is not None
                else "no scores"
            )
            print(f"\n  {i}. [{gap.count}x] {gap.pattern}")
            print(f"     Flags: {', '.join(gap.flags)}")
            print(f"     {score_str} | categories: {', '.join(gap.categories[:3]) or 'none'}")
            if gap.example_queries:
                print(f'     Example: "{gap.example_queries[0][:80]}"')

    if not report.gaps:
        print("\n  No recurring gaps detected.")

    print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Knowledge Gap Tracker")
    parser.add_argument("--top", type=int, default=20, help="Show top N gaps")
    parser.add_argument("--since", type=int, default=None, help="Only analyze last N days")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    entries = load_query_log(since_days=args.since)
    if not entries:
        print("No query log entries found. Run some queries first.")
        return

    report = analyze_gaps(entries, top_n=args.top)

    if args.json:
        import dataclasses

        print(json.dumps(dataclasses.asdict(report), indent=2, ensure_ascii=False))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
