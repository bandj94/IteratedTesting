"""Simple text statistics. See README.md for the exact rules."""


def words(text):
    return text.split(" ")


def word_count(text):
    return len(words(text))


def top_words(text, n):
    counts = {}
    for w in words(text):
        counts[w] = counts.get(w, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return [w for w, _ in ranked[:n]]
