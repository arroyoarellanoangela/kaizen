"""Split text into overlapping chunks."""


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 80) -> list[str]:
    """
    Split text into chunks of ~chunk_size characters, breaking at paragraph
    or sentence boundaries. Adds overlap from the previous chunk's tail.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    paragraphs = text.split("\n\n")
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current.strip())
            # Single paragraph larger than chunk_size → split by sentences
            if len(para) > chunk_size:
                sentences = para.replace(". ", ".\n").split("\n")
                sub = ""
                for sent in sentences:
                    if len(sub) + len(sent) + 1 <= chunk_size:
                        sub = f"{sub} {sent}" if sub else sent
                    else:
                        if sub:
                            chunks.append(sub.strip())
                        sub = sent
                current = sub
            else:
                current = para

    if current.strip():
        chunks.append(current.strip())

    # Add overlap: prepend tail of previous chunk
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap:]
            overlapped.append(f"{tail} {chunks[i]}")
        return overlapped

    return chunks
