"""Group near-duplicate chunks (boilerplate copied across connector pages).

~16% of SearchUnify chunks are copies of the same text (e.g. "After the First Crawl" appears
on 56 connector pages). Each chunk gets `dup_group` = chunk_id of its group's first member and
`dup_count` = group size. At query time results are collapsed by `dup_group`, so five copies
of one paragraph can't fill the top 5.
"""

import re

from llama_index.core.schema import TextNode


def shingles(text: str, n: int = 5) -> frozenset[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return frozenset(tuple(words[i : i + n]) for i in range(max(len(words) - n + 1, 1)))


def assign_dup_groups(chunks: list[TextNode], threshold: float = 0.8) -> int:
    """Union-find over pairs with word 5-shingle Jaccard >= threshold. Returns #chunks in a group of 2+."""
    sh = [shingles(c.text) for c in chunks]
    parent = list(range(len(chunks)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(sh)):
        for j in range(i + 1, len(sh)):
            a, b = sh[i], sh[j]
            # Jaccard can't reach the threshold if sizes differ too much
            if min(len(a), len(b)) < threshold * max(len(a), len(b)):
                continue
            if len(a & b) >= threshold * len(a | b):
                parent[find(j)] = find(i)

    roots = [find(i) for i in range(len(chunks))]
    sizes: dict[int, int] = {}
    for r in roots:
        sizes[r] = sizes.get(r, 0) + 1
    for c, r in zip(chunks, roots):
        c.metadata["dup_group"] = chunks[r].id_
        c.metadata["dup_count"] = sizes[r]
    return sum(n for n in sizes.values() if n > 1)
