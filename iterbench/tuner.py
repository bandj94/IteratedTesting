"""The tuning loop.

Per model, following the order of its `dims` list (default: reasoning, speed, sampling, kv):
  iteration 1   benchmark the original launcher settings (baseline)
  reasoning     a 4096-token thinking budget first, as a reference (the baselines often run out of tokens
                while still thinking, so they understate the model); then thinking off and smaller
                budgets, fastest first, stopping at the first one accepted. Done first because it
                changes run time the most, so every later run is shorter.
  speed         speed probes for answer-neutral settings (threads, load mode, batch size, VRAM margin;
                probes are not benchmarks and do not count as iterations), then one benchmark run. That run
                has the same answer-affecting settings as the run it was built from, so its accuracy counts
                as the average of the two (single runs are noisy: see docs/METHOD.md, "Noise")
  then          one answer-affecting change per iteration (sampling preset, KV cache type, ...)

Selection rule ("as smart as the smartest, then fastest"): among all runs of a model whose accuracy is
within `tolerance_pts` of the best accuracy seen for that model, the incumbent is the fastest one. A
challenger must be at least `speed_margin` faster to replace an incumbent that is still in that group.
Each new candidate is the incumbent plus one change. A model stops when every planned change has been
tried (converged), when it reaches its own cap, or when the global cap of iterations is reached.
State is saved after every run, so an interrupted `tune` resumes where it stopped.
"""
from __future__ import annotations

from .bench import run_benchmark
from .config import Config, ModelSpec, describe_diff, thinking_enabled
from .probe import run_probe
from .server import ServerError
from .util import RUNS_DIR, log, now_iso, read_json, stamp, write_json

STATE_PATH = RUNS_DIR / "tuner_state.json"
NEED_PROBE = "need_probe"


def load_state(cfg: Config) -> dict:
    st = read_json(STATE_PATH) or {"version": 1, "created": now_iso(), "iteration": 0, "models": {}, "final": {}}
    for name in cfg.models:
        st["models"].setdefault(name, {"status": "active", "runs": [], "probe": None, "dims": {},
                                       "incumbent": None, "speed_done": False})
        st["final"].setdefault(name, {})
    return st


def save_state(st: dict) -> None:
    st["updated"] = now_iso()
    write_json(STATE_PATH, st)


def run_by_id(ms: dict, run_id: str | None) -> dict | None:
    return next((r for r in ms["runs"] if r["run_id"] == run_id), None)


def selection_accuracy(r: dict) -> float:
    """The accuracy the selection rule uses. A speed-calibrated run has the same answer-affecting settings as
    the run it was built from, so the two are measurements of one configuration: it counts as their average."""
    return r.get("pooled_score", r["score"])


def best_accuracy(runs: list[dict]) -> float:
    """The best accuracy seen for a model, in the same terms: a speed-calibrated run counts as its average,
    because the higher of its two measurements would be a lucky draw, not a better configuration."""
    return max(selection_accuracy(r) for r in runs if r.get("ok"))


def pick_incumbent(runs: list[dict], prev_id: str | None, tol: float, margin: float,
                   pick_from_iteration: int = 0) -> dict | None:
    """Fastest run within `tol` of the best accuracy. Runs before `pick_from_iteration` still count
    for the best accuracy but can no longer be picked (they lack settings adopted since)."""
    ok = [r for r in runs if r.get("ok")]
    if not ok:
        return None
    best = best_accuracy(ok)
    pool = [r for r in ok if selection_accuracy(r) >= best - tol and r.get("iteration", 0) >= pick_from_iteration]
    if not pool:
        prev = next((r for r in ok if r["run_id"] == prev_id), None)
        if prev is not None and prev.get("iteration", 0) >= pick_from_iteration:
            return prev       # nothing newer qualifies: keep the pick rather than lose settings adopted since
        pool = [r for r in ok if selection_accuracy(r) >= best - tol]
    fastest = min(pool, key=lambda r: r["wall_s"])
    prev = next((r for r in pool if r["run_id"] == prev_id), None)
    if prev is not None and prev["wall_s"] <= fastest["wall_s"] * (1 + margin):
        return prev
    return fastest


def apply_variant(flags: dict, variant: dict) -> dict:
    new = dict(flags)
    changes = variant.get("set", {})
    if "reasoning" in changes:
        new.pop("reasoning-budget", None)
    if "reasoning-budget" in changes:
        new.pop("reasoning", None)
    for k, v in changes.items():
        if v is None:
            new.pop(k, None)
        else:
            new[k] = v
    return new


def applicable(variant: dict, flags: dict) -> bool:
    when = variant.get("when")
    if when == "thinking":
        return thinking_enabled(flags)
    if when == "no_thinking":
        return not thinking_enabled(flags)
    return True


def variants_for(cfg: Config, model: ModelSpec, dim: str) -> list[dict]:
    """The dimension's variants for this model, in search order (its `reference` first, if any)."""
    vs = list(cfg.dims[dim]["variants"])
    only = model.only.get(dim)
    if only:
        vs = [v for v in vs if v["label"] in only]
    ref = reference_label(cfg, model, dim, vs)
    return sorted(vs, key=lambda v: v["label"] != ref) if ref else vs


def reference_label(cfg: Config, model: ModelSpec, dim: str, variants: list[dict] | None = None) -> str | None:
    """The dimension's reference variant (run first; its acceptance does not end the dimension), if this
    model has it."""
    ref = cfg.dims.get(dim, {}).get("reference")
    if not ref:
        return None
    if variants is None:
        variants = variants_for(cfg, model, dim)
    return ref if any(v["label"] == ref for v in variants) else None


def next_candidate(cfg: Config, model: ModelSpec, ms: dict):
    """Return NEED_PROBE, None (converged) or (label, dim, variant_label, flags)."""
    runs = ms["runs"]
    tried_flags = [r["flags"] for r in runs]
    if not runs:
        return ("baseline", None, None, dict(model.baseline))
    inc = run_by_id(ms, ms["incumbent"])
    if inc is None:
        return None                       # every run failed; nothing sensible to build on
    for dim in model.dims:
        if dim == "speed":                # pseudo-dimension: speed calibration probes, then one run
            if ms["speed_done"]:
                continue
            if ms["probe"] is None:
                return NEED_PROBE
            ms["speed_done"] = True
            pf = ms["probe"]["best_flags"]
            if pf != inc["flags"] and pf not in tried_flags:
                return ("speed-calibrated settings", "speed", "probe winners", pf)
            continue
        ds = ms["dims"].setdefault(dim, {"tried": [], "skipped": [], "done": False})
        if ds["done"]:
            continue
        for v in variants_for(cfg, model, dim):
            if v["label"] in ds["tried"] or v["label"] in ds["skipped"]:
                continue
            if not applicable(v, inc["flags"]):
                ds["skipped"].append(v["label"])
                continue
            budget = v.get("set", {}).get("reasoning-budget")
            if budget and inc.get("max_reasoning_tokens") is not None and inc["max_reasoning_tokens"] <= budget:
                ds["skipped"].append(v["label"])   # the model never thinks that long: no effect possible
                continue
            flags = apply_variant(inc["flags"], v)
            if flags == inc["flags"] or flags in tried_flags:
                ds["skipped"].append(v["label"])
                continue
            return (v["label"], dim, v["label"], flags)
        ds["done"] = True
    return None


def run_speed_calibration(cfg: Config, model: ModelSpec, ms: dict) -> None:
    inc = run_by_id(ms, ms["incumbent"])
    base_flags = dict(inc["flags"] if inc else model.baseline)
    folder = RUNS_DIR / model.name / f"{stamp()}_{model.name}_speed-calibration"
    log(f"--- {model.name}: speed calibration (probes, not counted as iterations)")
    results = []

    def probe(flags: dict, label: str) -> dict:
        r = run_probe(cfg, model, flags, folder / f"probe_{len(results):02d}.log", label)
        results.append(r)
        write_json(folder / "probes.json", results)
        return r

    cur = probe(base_flags, "current settings")
    if cur.get("error"):
        ms["probe"] = {"results": results, "best_flags": base_flags, "error": cur["error"]}
        return
    cur_flags = base_flags
    for dim in cfg.speed_dims:
        best = cur
        for v in dim["variants"]:
            flags = apply_variant(cur_flags, v)
            if flags == cur_flags:
                continue
            r = probe(flags, v["label"])
            if not r.get("error") and r["turn_s"] < best["turn_s"] * (1 - cfg.probe_margin):
                best = r
        if best is not cur:
            log(f"speed: {model.name} keeps '{best['label']}' ({cur['turn_s']:.2f} -> {best['turn_s']:.2f} s/turn)")
        cur, cur_flags = best, best["flags"]
    ms["probe"] = {"results": results, "best_flags": cur_flags, "baseline": results[0], "best": cur,
                   "folder": str(folder)}


def pooled_note(entry: dict, short: bool = False) -> str:
    if "pooled_score" not in entry or abs(entry["pooled_score"] - entry["score"]) < 0.05:
        return ""
    if short:
        return " (average with the run it was built from)"
    return f" (accuracy counted as {entry['pooled_score']:.1f}%, the average with the run it was built from)"


def note_pick_moved(runs: list[dict], old: dict, new: dict) -> str:
    return (f"; its score raised the best to {best_accuracy(runs):.1f}%, so the old pick (iteration "
            f"{old['iteration']}, {selection_accuracy(old):.1f}%) fell out of the band and the pick moved to "
            f"iteration {new['iteration']} ({new['label']})")


def _entry(record: dict, iteration: int, label: str, dim, variant, flags: dict) -> dict:
    s = record.get("splits", {}).get("tune")
    ok = s is not None and not record.get("error")
    items = []
    if ok:
        run = read_json(record["run_dir"] + "/run.json") or {}
        items = run.get("items", {}).get("tune", [])
    max_reason = max((it.get("reasoning_tokens", 0) for it in items), default=None) if items else None
    return {
        "run_id": record["run_id"], "iteration": iteration, "label": label, "dim": dim, "variant": variant,
        "flags": flags, "ok": ok, "error": record.get("error"),
        "score": s["score"] if ok else None, "ci95": s["ci95"] if ok else None,
        "wall_s": s["wall_s"] if ok else None, "gen_tps": s["gen_tps"] if ok else None,
        "pp_tps": s["pp_tps"] if ok else None, "long_pp_tps": s["long_pp_tps"] if ok else None,
        "solved_per_hour": s["solved_per_hour"] if ok else None,
        "reasoning_tokens": s["reasoning_tokens"] if ok else None, "max_reasoning_tokens": max_reason,
        "truncated": s["truncated"] if ok else None, "by_category": s["by_category"] if ok else None,
        "n_passed": s["n_passed"] if ok else None, "n_items": s["n_items"] if ok else None,
        "gpu": record.get("gpu"), "server": {k: record.get("server", {}).get(k) for k in
                                              ("load_s", "vram_used_mib", "ram_available_mib")},
        "run_dir": record.get("run_dir"), "finished": record.get("finished"),
    }


def tasks_fingerprint() -> str:
    """SHA-256 over every file under tasks/ (path + content), to freeze the task set while tuning."""
    import hashlib

    from .tasks import TASKS_DIR
    h = hashlib.sha256()
    for p in sorted(TASKS_DIR.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            h.update(p.relative_to(TASKS_DIR).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def check_frozen(st: dict) -> None:
    fp = tasks_fingerprint()
    if st.get("tasks_sha256") is None:
        st["tasks_sha256"] = fp
    elif st["tasks_sha256"] != fp:
        raise SystemExit("The task files changed since tuning started, so new runs would not be comparable "
                         "with earlier ones. Restore them, or start over by deleting runs/tuner_state.json.")


def tune(cfg: Config, models: list[str], *, iterations: int | None = None) -> dict:
    st = load_state(cfg)
    check_frozen(st)
    done_now = 0
    while st["iteration"] < cfg.max_iterations:
        if iterations is not None and done_now >= iterations:
            break
        it = st["iteration"] + 1
        log(f"========== iteration {it} of at most {cfg.max_iterations} ==========")
        ran = 0
        for name in models:
            model, ms = cfg.models[name], st["models"][name]
            if ms["status"] != "active":
                continue
            if model.max_iterations and len(ms["runs"]) >= model.max_iterations:
                ms["status"] = "capped"
                log(f"{name}: reached its own cap of {model.max_iterations} iterations")
                save_state(st)
                continue
            cand = next_candidate(cfg, model, ms)
            if cand == NEED_PROBE:
                run_speed_calibration(cfg, model, ms)
                save_state(st)
                cand = next_candidate(cfg, model, ms)
            if cand is None:
                ms["status"] = "converged"
                log(f"{name}: converged (every planned change has been tried)")
                save_state(st)
                continue
            label, dim, variant, flags = cand
            inc_before = run_by_id(ms, ms["incumbent"])
            change = describe_diff(inc_before["flags"], flags) if inc_before else "original launcher settings"
            try:
                record = run_benchmark(cfg, model, flags, splits=("tune",), label=f"it{it:02d}-{label}")
            except ServerError as e:
                record = {"run_id": f"{stamp()}_{name}_failed", "error": f"server failed to start: {e}"[:1500],
                          "splits": {}, "run_dir": ""}
                log(f"{name}: {record['error'][:300]}")
            entry = _entry(record, it, label, dim, variant, flags)
            entry["change"] = change
            entry["parent"] = inc_before["run_id"] if inc_before else None
            if dim == "speed" and entry["ok"] and inc_before and inc_before.get("ok"):
                entry["pooled_score"] = (entry["score"] + inc_before["score"]) / 2
            ms["runs"].append(entry)
            new_inc = pick_incumbent(ms["runs"], ms["incumbent"], cfg.tolerance_pts, cfg.speed_margin,
                                     ms.get("pick_from_iteration", 0))
            slower = None
            if dim == "speed" and entry["ok"]:
                # Speed settings cannot make the model more or less accurate (they only change which tokens
                # get sampled), and the probes already measured their speed precisely. So keep them whenever
                # the pooled accuracy is in the tie band, without the (noisier) wall-time margin. Older runs
                # without them can no longer become the pick. The one exception: a run that is clearly
                # slower than the pick means the probes did not predict real use.
                best_score = best_accuracy(ms["runs"])
                if inc_before and inc_before.get("wall_s"):
                    slower = entry["wall_s"] / inc_before["wall_s"] - 1
                if selection_accuracy(entry) >= best_score - cfg.tolerance_pts and (slower is None or
                                                                                    slower <= cfg.speed_run_slack):
                    new_inc = entry
                    ms["pick_from_iteration"] = entry["iteration"]
            accepted = new_inc is not None and new_inc["run_id"] == entry["run_id"]
            if new_inc is not None:
                ms["incumbent"] = new_inc["run_id"]
            if entry["ok"]:
                if inc_before is None:
                    entry["decision"] = "baseline"
                elif accepted:
                    entry["decision"] = "accepted" + pooled_note(entry)
                else:
                    best = best_accuracy(ms["runs"])
                    acc = selection_accuracy(entry)
                    if acc < best - cfg.tolerance_pts:
                        entry["decision"] = f"rejected: accuracy {acc:.1f}%{pooled_note(entry, short=True)} is " \
                                            f"more than {cfg.tolerance_pts:g} points below the best ({best:.1f}%)"
                    elif slower is not None and slower > cfg.speed_run_slack:
                        entry["decision"] = f"rejected: {slower * 100:.0f}% slower than the current pick, " \
                                            "although the speed probes said it is faster"
                    else:
                        entry["decision"] = "rejected: not faster than the current pick"
            else:
                entry["decision"] = "failed: " + (entry.get("error") or "unknown error").splitlines()[0][:200].rstrip(":")
            if entry["ok"] and not accepted and inc_before is not None and new_inc is not None \
                    and new_inc["run_id"] != inc_before["run_id"]:
                # This run was not picked, yet the pick changed: its score raised the best accuracy, the old pick
                # fell out of the band, and another run in the band took over. Say so, or the change is silent.
                entry["pick_moved_to"] = new_inc["run_id"]
                entry["decision"] += note_pick_moved(ms["runs"], inc_before, new_inc)
            if dim and dim in ms["dims"]:
                ds = ms["dims"][dim]
                ds["tried"].append(variant)
                if accepted and cfg.dims[dim].get("mode") == "first_accept" and \
                        variant != reference_label(cfg, model, dim):
                    ds["done"] = True
            if entry["ok"]:
                log(f"{name}: {label}: {entry['score']:.1f}% in {entry['wall_s'] / 60:.1f} min -> {entry['decision']}")
            save_state(st)
            ran += 1
        if ran == 0:
            log("every model has converged or reached its cap")
            break
        st["iteration"] = it
        save_state(st)
        done_now += 1
    return st


def run_final(cfg: Config, models: list[str], seeds: int = 1) -> dict:
    """Held-out test: the original settings and the tuned settings, back to back, on the final split."""
    st = load_state(cfg)
    check_frozen(st)
    for name in models:
        model, ms = cfg.models[name], st["models"][name]
        inc = run_by_id(ms, ms["incumbent"])
        configs = [("baseline", dict(model.baseline))]
        if inc and inc["flags"] != model.baseline:
            configs.append(("tuned", inc["flags"]))
        for label, flags in configs:
            for s in range(seeds):
                key = f"{label}" if seeds == 1 else f"{label}-seed{s + 1}"
                if st["final"][name].get(key, {}).get("ok"):
                    log(f"{name}: final {key} already done, skipping")
                    continue
                try:
                    rec = run_benchmark(cfg, model, flags, splits=("final",), label=f"final-{key}",
                                        seed_offset=1000 * s)
                    summ = rec["splits"].get("final")
                    st["final"][name][key] = {"ok": summ is not None and not rec.get("error"), "flags": flags,
                                              "run_id": rec["run_id"], "summary": summ, "gpu": rec.get("gpu"),
                                              "error": rec.get("error"), "run_dir": rec.get("run_dir")}
                except ServerError as e:
                    st["final"][name][key] = {"ok": False, "flags": flags, "error": str(e)[:1500]}
                save_state(st)
    return st
