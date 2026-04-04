#!/usr/bin/env python3
"""
Query the engineer-knowledge RAG.

Usage:
    python query.py "what is star schema?"         # single query
    python query.py                                # interactive mode
    python query.py --cat data-engineering "..."  # filter by category
"""

import logging
import sys

from suyven_rag.rag.index_registry import get_index
from suyven_rag.rag.orchestrator import execute_search, format_context, plan
from suyven_rag.rag.store import ensure_ollama

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s  %(message)s")


def run_query(query: str, category: str | None = None) -> None:
    route = plan(query, category=category)
    results = execute_search(query, route, category=category)
    print(f"\nQuery : {query}")
    if category:
        print(f"Filter: category={category}")
    print(f"Found : {len(results)} chunks\n")
    print(format_context(results))
    print()


def main() -> None:
    ensure_ollama()
    col = get_index()
    count = col.count()

    if count == 0:
        print("[error] Knowledge base is empty. Run: python ingest.py")
        sys.exit(1)

    print(f"Knowledge base: {count} chunks indexed")

    # Parse args: python query.py [--cat CATEGORY] [QUERY...]
    args = sys.argv[1:]
    category = None

    if "--cat" in args:
        idx = args.index("--cat")
        category = args[idx + 1]
        args = args[:idx] + args[idx + 2 :]

    if args:
        run_query(" ".join(args), category=category)
        return

    # Interactive mode
    print("Interactive mode — type a query, Ctrl+C to exit\n")
    while True:
        try:
            query = input(">> ").strip()
            if not query:
                continue
            run_query(query, category=category)
        except KeyboardInterrupt:
            print("\nBye.")
            break


if __name__ == "__main__":
    main()
