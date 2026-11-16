"""Unit tests for the harness (no model or GPU needed). Run: python -m unittest discover -s tests"""
import json
import tempfile
import unittest
from pathlib import Path

from iterbench.config import ModelSpec, flags_to_args, describe_diff, thinking_enabled
from iterbench.graders import codegen, longctx, toolcall
from iterbench.sandbox import run_guarded
from iterbench.tuner import apply_variant, pick_incumbent


def run(run_id, score, wall, ok=True):
    return {"run_id": run_id, "score": score, "wall_s": wall, "ok": ok}


class FlagTests(unittest.TestCase):
    def test_flags_to_args(self):
        self.assertEqual(flags_to_args({"ctx-size": 4096, "temp": 0.6, "jinja": True, "x": False, "y": None}),
                         ["--ctx-size", "4096", "--temp", "0.6", "--jinja"])
        self.assertEqual(flags_to_args({"temp": 1.0}), ["--temp", "1"])

    def test_describe_diff(self):
        self.assertEqual(describe_diff({"a": 1, "b": 2}, {"a": 1, "b": 3, "c": "x"}), "b=3, c=x")
        self.assertEqual(describe_diff({"a": 1}, {"a": 1}), "(no change)")

    def test_thinking(self):
        self.assertTrue(thinking_enabled({}))
        self.assertFalse(thinking_enabled({"reasoning": "off"}))
        self.assertFalse(thinking_enabled({"reasoning-budget": 0}))
        self.assertTrue(thinking_enabled({"reasoning-budget": 1024}))

    def test_apply_variant_reasoning_is_exclusive(self):
        f = apply_variant({"reasoning-budget": 1024, "temp": 0.6}, {"set": {"reasoning": "off"}})
        self.assertEqual(f, {"temp": 0.6, "reasoning": "off"})
        f = apply_variant({"reasoning": "off"}, {"set": {"reasoning-budget": 2048}})
        self.assertEqual(f, {"reasoning-budget": 2048})


class LauncherNoteTests(unittest.TestCase):
    def test_margin_costs_compare_only_the_margin(self):
        from iterbench.launchers import margin_costs
        probes = {"results": [
            {"label": "current", "flags": {"t": 1}, "turn_s": 15.0},                     # before load-mode
            {"label": "load-mode", "flags": {"t": 1, "lm": "none"}, "turn_s": 12.0},     # default margin
            {"label": "fit 512", "flags": {"t": 1, "lm": "none", "fit-target": 512}, "turn_s": 11.0},
            {"label": "fit 256", "flags": {"t": 1, "lm": "none", "fit-target": 256}, "turn_s": 10.0},
            {"label": "broken", "flags": {"t": 1, "lm": "none", "fit-target": 128}, "error": "oom"}]}
        costs = margin_costs({"probe": probes}, 256)
        self.assertEqual(sorted(costs), [512, 1024])
        self.assertAlmostEqual(costs[512], 10.0)
        self.assertAlmostEqual(costs[1024], 20.0)
        self.assertEqual(margin_costs({"probe": probes}, 512), {1024: 100 * (12.0 / 11.0 - 1)})
        self.assertEqual(margin_costs({}, 256), {})


class SelectionRuleTests(unittest.TestCase):
    def test_fastest_within_tolerance(self):
        runs = [run("a", 70, 600), run("b", 68, 400), run("c", 60, 100)]
        self.assertEqual(pick_incumbent(runs, None, 4, 0.05)["run_id"], "b")

    def test_more_accurate_wins_even_if_slower(self):
        runs = [run("a", 60, 100), run("b", 75, 900)]
        self.assertEqual(pick_incumbent(runs, "a", 4, 0.05)["run_id"], "b")

    def test_incumbent_kept_unless_clearly_faster(self):
        runs = [run("a", 70, 600), run("b", 70, 590)]
        self.assertEqual(pick_incumbent(runs, "a", 4, 0.05)["run_id"], "a")
        runs.append(run("c", 70, 500))
        self.assertEqual(pick_incumbent(runs, "a", 4, 0.05)["run_id"], "c")

    def test_failed_runs_ignored(self):
        runs = [run("a", 70, 600), {"run_id": "x", "ok": False, "score": None, "wall_s": None}]
        self.assertEqual(pick_incumbent(runs, "a", 4, 0.05)["run_id"], "a")
        self.assertIsNone(pick_incumbent([runs[1]], None, 4, 0.05))

    def test_pick_from_iteration(self):
        runs = [dict(run("base", 70, 500), iteration=1), dict(run("speed", 70, 560), iteration=2),
                dict(run("off", 50, 100), iteration=3)]
        self.assertEqual(pick_incumbent(runs, "speed", 4, 0.05, pick_from_iteration=2)["run_id"], "speed")
        self.assertEqual(pick_incumbent(runs, "speed", 4, 0.05)["run_id"], "base")

    def test_speed_run_counts_as_average(self):
        # The speed run's own score (69) dips below the band (70+), but averaged with the run it was built
        # from (72) it is in, so it stays the pick; a newer run in the band that is slower does not replace it.
        runs = [dict(run("best", 74, 900), iteration=1), dict(run("parent", 72, 400), iteration=2),
                dict(run("speed", 69, 340), iteration=3, pooled_score=70.5)]
        self.assertEqual(pick_incumbent(runs, "speed", 4, 0.05, pick_from_iteration=3)["run_id"], "speed")
        runs.append(dict(run("slower", 71, 420), iteration=4))
        self.assertEqual(pick_incumbent(runs, "speed", 4, 0.05, pick_from_iteration=3)["run_id"], "speed")
        runs.append(dict(run("faster", 71, 200), iteration=5))
        self.assertEqual(pick_incumbent(runs, "speed", 4, 0.05, pick_from_iteration=3)["run_id"], "faster")

    def test_speed_run_average_sets_the_bar(self):
        # The speed run's own score (80.8) is a lucky draw above the best (78.8); its average (77.9) is what
        # counts, so the band still starts at 74.8 and a faster run at 75.5 can take over.
        runs = [dict(run("best", 78.8, 2800), iteration=2), dict(run("parent", 75.0, 2000), iteration=5),
                dict(run("speed", 80.8, 1370), iteration=6, pooled_score=77.9)]
        self.assertEqual(pick_incumbent(runs, "speed", 4, 0.05, pick_from_iteration=6)["run_id"], "speed")
        runs.append(dict(run("faster", 75.5, 1000), iteration=7))
        self.assertEqual(pick_incumbent(runs, "speed", 4, 0.05, pick_from_iteration=6)["run_id"], "faster")

    def test_keeps_pick_when_nothing_newer_qualifies(self):
        # Older runs lack settings adopted since: if no newer run is in the band, keep the current pick.
        runs = [dict(run("best", 74, 900), iteration=1), dict(run("parent", 72, 400), iteration=2),
                dict(run("speed", 69, 340), iteration=3), dict(run("worse", 50, 100), iteration=4)]
        self.assertEqual(pick_incumbent(runs, "speed", 4, 0.05, pick_from_iteration=3)["run_id"], "speed")

    def test_no_ratchet(self):
        # Each step loses 3 points: the third is more than 4 below the best and must lose.
        runs = [run("a", 70, 600), run("b", 67, 500), run("c", 64, 400)]
        self.assertEqual(pick_incumbent(runs, "b", 4, 0.05)["run_id"], "b")


class CodeExtractionTests(unittest.TestCase):
    def test_picks_block_with_definition(self):
        text = "Here:\n```python\nprint(f(1))\n```\nand\n```python\ndef f(x):\n    return x\n```\n"
        self.assertIn("def f", codegen.extract_code(text, ["f"]))

    def test_unterminated_block(self):
        self.assertIn("def g", codegen.extract_code("```python\ndef g():\n    pass\n", ["g"]))

    def test_two_entries_in_two_blocks(self):
        text = "```python\ndef a():\n    pass\n```\n```python\ndef b():\n    pass\n```"
        code = codegen.extract_code(text, ["a", "b"])
        self.assertIn("def a", code)
        self.assertIn("def b", code)

    def test_grading_end_to_end(self):
        data = {"entry": "add", "tests": "check(lambda: add(1, 2), 3)\ncheck_raises(lambda: add(None, 1), TypeError)"}
        with tempfile.TemporaryDirectory() as d:
            good = codegen.grade("codegen", data, "```python\ndef add(a, b):\n    return a + b\n```", Path(d) / "w")
            bad = codegen.grade("codegen", data, "```python\ndef add(a, b):\n    return a - b\n```", Path(d) / "w")
            crash = codegen.grade("codegen", data, "```python\nraise SystemExit(3)\n```", Path(d) / "w")
        self.assertTrue(good["passed"])
        self.assertFalse(bad["passed"])
        self.assertEqual(bad["partial"], 0.5)
        self.assertFalse(crash["passed"])

    def test_main_block_does_not_run(self):
        data = {"entry": "f", "tests": "check(lambda: f(), 1)"}
        answer = "```python\ndef f():\n    return 1\n\nif __name__ == '__main__':\n    input()\n```"
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(codegen.grade("codegen", data, answer, Path(d) / "w")["passed"])


class ToolCallGraderTests(unittest.TestCase):
    def test_path_normalization(self):
        self.assertTrue(toolcall.match_value("./src\\a.py", {"path": "src/a.py"}))
        self.assertFalse(toolcall.match_value("src/b.py", {"path": "src/a.py"}))

    def test_number_rule(self):
        self.assertTrue(toolcall.match_value(120, {"number": 120}))
        self.assertTrue(toolcall.match_value("120", {"number": 120}))
        self.assertFalse(toolcall.match_value(True, {"number": 1}))

    def test_answer_vs_call(self):
        data = {"expect": {"type": "answer", "contains_all": ["yes"]}}
        self.assertTrue(toolcall.grade(data, "Yes, it is.", [])["passed"])
        call = {"name": "read_file", "args": {"path": "x"}, "arguments": "{}"}
        self.assertFalse(toolcall.grade(data, "", [call])["passed"])

    def test_invalid_json_args(self):
        data = {"expect": {"type": "call", "name": "read_file", "args": {"path": {"path": "a"}}}}
        call = {"name": "read_file", "args": None, "arguments": "{oops"}
        self.assertFalse(toolcall.grade(data, "", [call])["passed"])


class LongContextTests(unittest.TestCase):
    def test_deterministic_and_correct(self):
        a = longctx.generate(7, 3000)
        b = longctx.generate(7, 3000)
        self.assertEqual(a["code"], b["code"])
        self.assertTrue(a["questions"])
        self.assertGreaterEqual(a["questions"][0]["depth"], 1)

    def test_answer_parsing(self):
        self.assertEqual(longctx.parse_answer("blah\nANSWER: 42"), 42)
        self.assertEqual(longctx.parse_answer("**Answer:** -7"), -7)
        self.assertEqual(longctx.parse_answer("ANSWER: 1,234"), 1234)
        self.assertEqual(longctx.parse_answer("so the result is\n58"), 58)
        self.assertIsNone(longctx.parse_answer("I am not sure."))


class SandboxTests(unittest.TestCase):
    def _run(self, code):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "s.py").write_text(code, encoding="utf-8")
            return run_guarded(Path(d), ["script", "s.py"], timeout=20)

    def test_blocks_process_creation(self):
        r = self._run("import os\ntry:\n    os.system('echo hi')\nexcept PermissionError:\n    print('@@RESULT@@{\"ok\": 1}')\n")
        self.assertEqual(r.result, {"ok": 1})

    def test_blocks_writes_outside(self):
        target = Path(tempfile.gettempdir()) / "iterbench_guard_test.txt"
        code = ("try:\n    open(%r, 'w').write('x')\nexcept PermissionError:\n    print('@@RESULT@@{\"ok\": 1}')\n" % str(target))
        r = self._run(code)
        self.assertEqual(r.result, {"ok": 1})
        self.assertFalse(target.exists())

    def test_allows_writes_inside(self):
        r = self._run("open('inside.txt', 'w').write('x')\nprint('@@RESULT@@' + '{\"ok\": 1}')\n")
        self.assertEqual(r.result, {"ok": 1})

    def test_blocks_network(self):
        r = self._run("import socket\ntry:\n    socket.create_connection(('127.0.0.1', 9), timeout=1)\n"
                      "except PermissionError:\n    print('@@RESULT@@{\"ok\": 1}')\n")
        self.assertEqual(r.result, {"ok": 1})

    def test_timeout(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "s.py").write_text("while True:\n    pass\n", encoding="utf-8")
            r = run_guarded(Path(d), ["script", "s.py"], timeout=2)
        self.assertTrue(r.timed_out)


if __name__ == "__main__":
    unittest.main()
