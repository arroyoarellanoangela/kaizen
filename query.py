#!/usr/bin/env python3
"""
Query the engineer-knowledge RAG.

Usage:
    python query.py "what is star schema?"         # single query
    python query.py                                # interactive mode
    python query.py --cat data-engineering "..."  # filter by category
"""

import sys

from rag.retriever import format_context, search
from rag.store import ensure_ollama, get_collection


def run_query(query: str, category: str | None = None) -> None:
    results = search(query, category=category)
    print(f"\nQuery : {query}")
    if category:
        print(f"Filter: category={category}")
    print(f"Found : {len(results)} chunks\n")
    print(format_context(results))
    print()


def main() -> None:
    ensure_ollama()
    col = get_collection()
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
        args = args[:idx] + args[idx + 2:]

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
