# textstats

Small helpers for counting words.

- `words(text)`: the words of `text`, lowercased. A word is a run of ASCII letters, digits or
  apostrophes (`[A-Za-z0-9']+`); every other character separates words.
- `word_count(text)`: the number of words in `text`.
- `top_words(text, n)`: the `n` most common words, most common first. Words with the same count are
  ordered alphabetically. If there are fewer than `n` distinct words, return all of them.
