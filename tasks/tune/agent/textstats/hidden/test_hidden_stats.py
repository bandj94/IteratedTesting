import unittest

from stats import top_words, words


class HiddenStatsTests(unittest.TestCase):
    def test_ties_are_alphabetical(self):
        self.assertEqual(top_words("pear apple fig apple pear fig", 3), ["apple", "fig", "pear"])

    def test_case_insensitive_counts(self):
        self.assertEqual(top_words("Go go GO stop", 1), ["go"])

    def test_n_larger_than_vocabulary(self):
        self.assertEqual(top_words("x y", 5), ["x", "y"])

    def test_only_punctuation(self):
        self.assertEqual(words("... --- !!!"), [])

    def test_underscore_and_hyphen_separate(self):
        self.assertEqual(words("snake_case and kebab-case"), ["snake", "case", "and", "kebab", "case"])


if __name__ == "__main__":
    unittest.main()
