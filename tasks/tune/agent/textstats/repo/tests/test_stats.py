import unittest

from stats import top_words, word_count, words


class StatsTests(unittest.TestCase):
    def test_words_are_lowercased_and_split_on_punctuation(self):
        self.assertEqual(words("Hello, world! It's 9am."), ["hello", "world", "it's", "9am"])

    def test_word_count_ignores_extra_whitespace(self):
        self.assertEqual(word_count("  one   two\nthree  "), 3)

    def test_empty_text(self):
        self.assertEqual(word_count(""), 0)

    def test_top_words(self):
        self.assertEqual(top_words("b a b c a b", 2), ["b", "a"])


if __name__ == "__main__":
    unittest.main()
