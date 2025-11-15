import collections
import math
from typing import Any, DefaultDict, List, Set, Tuple

SparseVector = DefaultDict[Any, float]
Position = Tuple[int, int]


def find_alphabetically_first_word(text: str) -> str:
    """
    Given a string |text|, return the word in |text| that comes first
    lexicographically. Words split by whitespace. Case-sensitive.
    """
    # BEGIN_YOUR_CODE
    words = text.split()
    if not words:
        return ""
    return min(words)
    # END_YOUR_CODE


def euclidean_distance(loc1: Position, loc2: Position) -> float:
    """
    Return the Euclidean distance between two 2D positions.
    """
    # BEGIN_YOUR_CODE
    return math.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)
    # END_YOUR_CODE


def mutate_sentences(sentence: str) -> List[str]:
    """
    Generate all sentences of the same length where each adjacent pair
    appears in the original sentence in the same order.
    """
    # BEGIN_YOUR_CODE
    words = sentence.split()
    if not words:
        return []
    # adjacency: for each word, collect set of words that can follow it
    adj = collections.defaultdict(set)
    for a, b in zip(words[:-1], words[1:]):
        adj[a].add(b)

    n = len(words)
    results = set()

    # DFS from a given starting word
    def dfs(path):
        if len(path) == n:
            results.add(" ".join(path))
            return
        last = path[-1]
        # if last has no outgoing edges, cannot extend
        for nxt in adj.get(last, []):
            dfs(path + [nxt])

    # start from every word in the original sentence (duplicates OK; set handles dedupe)
    for start in words:
        dfs([start])

    return list(results)
    # END_YOUR_CODE


def sparse_vector_dot_product(v1: SparseVector, v2: SparseVector) -> float:
    """
    Dot product of two sparse vectors represented as defaultdicts.
    """
    # BEGIN_YOUR_CODE
    return sum(v1[k] * v2[k] for k in v1 if k in v2)
    # END_YOUR_CODE


def increment_sparse_vector(v1: SparseVector, scale: float, v2: SparseVector) -> None:
    """
    Perform v1 += scale * v2 (in-place). v2 must not be modified.
    """
    # BEGIN_YOUR_CODE
    for k, val in v2.items():
        v1[k] += scale * val
    # END_YOUR_CODE


def find_nonsingleton_words(text: str) -> Set[str]:
    """
    Return the set of words that occur more than once in text (split on whitespace).
    """
    # BEGIN_YOUR_CODE
    counts = collections.defaultdict(int)
    for w in text.split():
        counts[w] += 1
    return {w for w, c in counts.items() if c > 1}
    # END_YOUR_CODE
