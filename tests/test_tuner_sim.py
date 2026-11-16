"""Simulate the whole tuning loop with a fake server, to test the state machine without a GPU."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from iterbench import tuner
from iterbench.config import CONFIG_DIR, ModelSpec, load_config


def fake_score(flags, threads_factor=0.9, threads_delta=0.0):
    """A made-up model whose baseline runs out of tokens on some tasks: budget 4096 scores higher than
    the baseline, budget 2048 holds that, budget 1024 loses 8 points, thinking off loses 14, KV q4 loses 6."""
    score, wall = 70.0, 600.0
    budget = flags.get("reasoning-budget")
    if flags.get("reasoning") == "off":
        score, wall = 60.0, 150.0
    elif budget == 1024:
        score, wall = 66.0, 300.0
    elif budget == 2048:
        score, wall = 72.0, 380.0
    elif budget == 4096:
        score, wall = 74.0, 480.0
    if flags.get("cache-type-k") == "q4_0":
        score -= 6
        wall *= 0.9
    if flags.get("cache-type-k") == "f16":
        wall *= 1.2
    if flags.get("presence-penalty"):
        score -= 1
    if flags.get("threads") == 6:
        wall *= threads_factor
        score += threads_delta
    return score, wall


class TunerSimulation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        self.patches = [mock.patch.object(tuner, "STATE_PATH", tmp / "state.json"),
                        mock.patch.object(tuner, "RUNS_DIR", tmp)]
        for p in self.patches:
            p.start()
        # The real search space, with made-up paths: the tests must not need config/local.toml.
        cfg_dir = tmp / "config"
        cfg_dir.mkdir()
        shutil.copy(CONFIG_DIR / "models.toml", cfg_dir / "models.toml")
        (cfg_dir / "local.toml").write_text('[paths]\nserver_exe = "llama-server"\nmodels_dir = "models"\n',
                                            encoding="utf-8")
        self.cfg = load_config(cfg_dir)
        self.cfg.models = {"fake": ModelSpec(name="fake", file=tmp / "x.gguf", suite="full",
                                             baseline={"ctx-size": 4096, "temp": 0.6},
                                             dims=["reasoning", "speed", "sampling", "kv"])}
        self.calls = []
        self.threads_factor = 0.9    # what 6 threads really do to the benchmark's wall time
        self.threads_delta = 0.0     # ... and to its score (sampling noise in a real run)
        self.score_adjust = lambda flags: 0.0    # extra per-setting noise for a test

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def fake_bench(self, cfg, model, flags, splits=("tune",), label="", **kw):
        self.calls.append(dict(flags))
        score, wall = fake_score(flags, self.threads_factor, self.threads_delta)
        score += self.score_adjust(flags)
        return {"run_id": f"r{len(self.calls)}", "run_dir": "", "error": None, "flags": flags,
                "splits": {"tune": {"score": score, "ci95": [0, 100], "wall_s": wall, "gen_tps": 40.0,
                                    "pp_tps": 900.0, "long_pp_tps": 900.0, "solved_per_hour": 1.0,
                                    "reasoning_tokens": 100, "truncated": 0, "by_category": {},
                                    "n_passed": 1, "n_items": 2}}}

    def fake_probe(self, cfg, model, flags, log_path, label=""):
        turn = 10.0 * (0.85 if flags.get("threads") == 6 else 1.0)
        return {"label": label, "flags": dict(flags), "turn_s": turn, "pp_tps": 1, "tg_deep_tps": 1,
                "tg_short_tps": 1, "error": None}

    def test_full_loop(self):
        with mock.patch.object(tuner, "run_benchmark", self.fake_bench), \
                mock.patch.object(tuner, "run_probe", self.fake_probe):
            st = tuner.tune(self.cfg, ["fake"])
        ms = st["models"]["fake"]
        self.assertEqual(ms["status"], "converged")
        labels = [r["label"] for r in ms["runs"]]
        self.assertEqual(labels[:5], ["baseline", "budget 4096", "thinking off", "budget 1024", "budget 2048"])
        runs = {r["label"]: r for r in ms["runs"]}
        self.assertEqual(runs["budget 4096"]["decision"], "accepted")    # the reference does not end the search
        self.assertTrue(runs["thinking off"]["decision"].startswith("rejected: accuracy"))
        self.assertEqual(runs["budget 2048"]["decision"], "accepted")
        self.assertEqual(labels.count("budget 4096"), 1)                   # not run twice
        self.assertLess(labels.index("budget 2048"), labels.index("speed-calibrated settings"))
        self.assertLess(labels.index("speed-calibrated settings"), labels.index("KV q4_0"))
        self.assertIn("KV q4_0", labels)
        self.assertIn("KV f16", labels)
        pick = tuner.run_by_id(ms, ms["incumbent"])
        self.assertEqual(pick["flags"].get("reasoning-budget"), 2048)
        self.assertEqual(pick["flags"].get("threads"), 6)
        self.assertNotEqual(pick["flags"].get("cache-type-k"), "q4_0")
        self.assertLessEqual(st["iteration"], self.cfg.max_iterations)
        # Resuming a converged state does nothing.
        n = len(self.calls)
        with mock.patch.object(tuner, "run_benchmark", self.fake_bench), \
                mock.patch.object(tuner, "run_probe", self.fake_probe):
            tuner.tune(self.cfg, ["fake"])
        self.assertEqual(len(self.calls), n)

    def test_report_and_launchers(self):
        from iterbench import launchers, report
        tmp = Path(self.tmp.name)
        with mock.patch.object(tuner, "run_benchmark", self.fake_bench), \
                mock.patch.object(tuner, "run_probe", self.fake_probe):
            tuner.tune(self.cfg, ["fake"])
            st = tuner.load_state(self.cfg)
            st["final"]["fake"] = {
                "baseline": {"ok": True, "flags": {}, "run_id": "fb",
                             "summary": {"score": 60.0, "ci95": [40, 75], "wall_s": 300, "gen_tps": 40.0,
                                         "truncated": 3, "solved_per_hour": 5.0}},
                "tuned": {"ok": True, "flags": {}, "run_id": "ft",
                          "summary": {"score": 61.0, "ci95": [41, 76], "wall_s": 200, "gen_tps": 41.0,
                                      "truncated": 0, "solved_per_hour": 8.0}}}
            tuner.save_state(st)
            # Held-out items: final/a1 fails in both runs, final/b2 passes once.
            for run_id, passed in (("fb", (False, True)), ("ft", (False, False))):
                items = tmp / "fake" / run_id / "final" / "items"
                items.mkdir(parents=True, exist_ok=True)
                for tid, ok in zip(("final/a1", "final/b2"), passed):
                    (items / f"{tid.replace('/', '_')}.json").write_text(
                        json.dumps({"id": tid, "category": "bugfix", "passed": ok}), encoding="utf-8")
        with mock.patch.object(report, "RESULTS_DIR", tmp / "results"), \
                mock.patch.object(report, "RUNS_DIR", tmp), \
                mock.patch.object(launchers, "LAUNCHER_DIR", tmp / "launchers"):
            path = report.build(self.cfg)
            text = path.read_text(encoding="utf-8")
            self.assertIn("## fake", text)
            self.assertIn("budget 2048", text)
            self.assertIn("**(pick)**", text)
            self.assertIn("| 1.50x |", text)                                   # held-out speed-up, 300 s / 200 s
            self.assertIn("No held-out run passed `final/a1` (bugfix, 2 runs).", text)
            self.assertNotIn("`final/b2`", text)
            configs = json.loads((tmp / "results" / "best_configs.json").read_text(encoding="utf-8"))
            self.assertEqual(configs["fake"]["held_out"]["tuned"], {"score": 61.0, "wall_s": 200})
            self.assertTrue((tmp / "results" / "tradeoff.svg").exists())
            self.assertTrue((tmp / "results" / "iterations.csv").exists())
            written = launchers.write_launchers(self.cfg)
            self.assertEqual(len(written), 1)
            bat = written[0].read_text(encoding="utf-8")
            self.assertIn("--reasoning-budget 2048", bat)
            self.assertIn("--threads 6", bat)

    def test_iteration_cap(self):
        self.cfg.max_iterations = 3
        with mock.patch.object(tuner, "run_benchmark", self.fake_bench), \
                mock.patch.object(tuner, "run_probe", self.fake_probe):
            st = tuner.tune(self.cfg, ["fake"])
        self.assertEqual(st["iteration"], 3)
        self.assertEqual(len(self.calls), 3)
        self.assertEqual([c.get("reasoning") or c.get("reasoning-budget") for c in self.calls], [None, 4096, "off"])

    def test_speed_run_rejected_when_slower(self):
        """The probes say 6 threads is faster, but the real benchmark is 20% slower: keep the old pick."""
        self.threads_factor = 1.2
        with mock.patch.object(tuner, "run_benchmark", self.fake_bench), \
                mock.patch.object(tuner, "run_probe", self.fake_probe):
            st = tuner.tune(self.cfg, ["fake"])
        ms = st["models"]["fake"]
        speed = next(r for r in ms["runs"] if r["label"] == "speed-calibrated settings")
        self.assertTrue(speed["decision"].startswith("rejected: 20% slower"), speed["decision"])
        pick = tuner.run_by_id(ms, ms["incumbent"])
        self.assertIsNone(pick["flags"].get("threads"))

    def test_speed_run_counts_as_average(self):
        """The speed run's own score (69) is just below the band (70+), as sampling noise can make it; averaged
        with the run it was built from (72) it is in, so the speed settings are kept."""
        self.threads_delta = -3.0
        with mock.patch.object(tuner, "run_benchmark", self.fake_bench), \
                mock.patch.object(tuner, "run_probe", self.fake_probe):
            st = tuner.tune(self.cfg, ["fake"])
        ms = st["models"]["fake"]
        speed = next(r for r in ms["runs"] if r["label"] == "speed-calibrated settings")
        self.assertEqual((speed["score"], speed["pooled_score"]), (69.0, 70.5))
        self.assertTrue(speed["decision"].startswith("accepted (accuracy counted as 70.5%"), speed["decision"])
        self.assertEqual(tuner.run_by_id(ms, ms["incumbent"])["flags"].get("threads"), 6)
        later = [r for r in ms["runs"] if r["iteration"] > speed["iteration"]]
        self.assertTrue(later and all(r["decision"].startswith("rejected") for r in later))

    def test_pick_move_is_reported(self):
        """A slow run with a lucky score raises the best accuracy and pushes the pick out of the band; another
        run in the band (rejected earlier as not faster) becomes the pick, and the slow run's decision says so."""
        self.threads_delta = -3.0                       # speed run 69, averaged 70.5: just in the band (70+)
        self.score_adjust = lambda f: 4.0 if f.get("presence-penalty") else (7.0 if f.get("cache-type-k") == "f16" else 0.0)
        with mock.patch.object(tuner, "run_benchmark", self.fake_bench), \
                mock.patch.object(tuner, "run_probe", self.fake_probe):
            st = tuner.tune(self.cfg, ["fake"])
        ms = st["models"]["fake"]
        runs = {r["label"]: r for r in ms["runs"]}
        sampling = next(r for r in ms["runs"] if r["dim"] == "sampling")
        self.assertEqual((sampling["score"], sampling["decision"]), (72.0, "rejected: not faster than the current pick"))
        f16 = runs["KV f16"]
        self.assertEqual(f16["score"], 76.0)
        self.assertEqual(f16["pick_moved_to"], sampling["run_id"])
        self.assertIn("raised the best to 76.0%", f16["decision"])
        self.assertIn(f"moved to iteration {sampling['iteration']}", f16["decision"])
        self.assertEqual(ms["incumbent"], sampling["run_id"])

    def test_reference_skipped_when_not_in_only(self):
        """A model limited to thinking off / budget 1024 has no reference run: thinking off goes first."""
        self.cfg.models["fake"].only = {"reasoning": ["thinking off", "budget 1024"]}
        self.cfg.max_iterations = 2
        with mock.patch.object(tuner, "run_benchmark", self.fake_bench), \
                mock.patch.object(tuner, "run_probe", self.fake_probe):
            tuner.tune(self.cfg, ["fake"])
        self.assertEqual([c.get("reasoning") or c.get("reasoning-budget") for c in self.calls], [None, "off"])


if __name__ == "__main__":
    unittest.main()
