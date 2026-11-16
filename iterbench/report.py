"""Build results/REPORT.md, results/iterations.csv, results/best_configs.json and results/tradeoff.svg."""
from __future__ import annotations

import csv
import html
from pathlib import Path

from .config import Config, describe, describe_diff
from .tasks import load_tasks, select
from .tuner import load_state, run_by_id
from .util import RESULTS_DIR, RUNS_DIR, fmt_s, read_json, wilson, write_json

COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]


def _f(v, spec="{:.1f}", none="-"):
    return none if v is None else spec.format(v)


def _md_escape(text: str) -> str:
    return str(text).replace("|", "\\|")


def _averaged(r: dict | None) -> str:
    """' (90.4% averaged)' for a speed-calibrated run, whose accuracy counts as the average with its parent."""
    if not r or r.get("pooled_score") is None or abs(r["pooled_score"] - r["score"]) < 0.05:
        return ""
    return f" ({r['pooled_score']:.1f}% averaged)"


def unsolved_final_tasks(st: dict) -> list[tuple[str, str, int]]:
    """Held-out tasks that at least two final runs attempted and none passed, as (id, category, runs).
    A task that every setting fails cannot tell settings apart, and it may be broken rather than hard."""
    seen: dict[str, list] = {}
    for name, fin in st.get("final", {}).items():
        for f in fin.values():
            if not f.get("run_id"):
                continue
            for p in sorted((RUNS_DIR / name / f["run_id"] / "final" / "items").glob("*.json")):
                it = read_json(p, {})
                if "id" in it:
                    entry = seen.setdefault(it["id"], [it.get("category", "?"), 0, 0])
                    entry[1] += 1
                    entry[2] += bool(it.get("passed"))
    return [(tid, cat, n) for tid, (cat, n, passed) in sorted(seen.items()) if n >= 2 and not passed]


def svg_tradeoff(st: dict, cfg: Config) -> str:
    """Accuracy (tune split) against suite time, every run of every model."""
    pts = []
    for i, (name, ms) in enumerate(st["models"].items()):
        for r in ms["runs"]:
            if r.get("ok"):
                pts.append((name, i, r["wall_s"] / 60, r["score"], r["run_id"] == ms["incumbent"],
                            r["label"] == "baseline"))
    if not pts:
        return ""
    W, H, L, R, T, B = 760, 440, 70, 190, 30, 60
    xmax = max(p[2] for p in pts) * 1.08
    ymin = max(0, min(p[3] for p in pts) - 10)
    ymax = min(100, max(p[3] for p in pts) + 5)

    def X(v):
        return L + (W - L - R) * v / xmax

    def Y(v):
        return T + (H - T - B) * (1 - (v - ymin) / ((ymax - ymin) or 1))

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Segoe UI, Arial, sans-serif" font-size="12">',
           f'<rect width="{W}" height="{H}" fill="white"/>']
    for k in range(6):
        yv = ymin + (ymax - ymin) * k / 5
        out.append(f'<line x1="{L}" y1="{Y(yv):.1f}" x2="{W - R}" y2="{Y(yv):.1f}" stroke="#e5e7eb"/>')
        out.append(f'<text x="{L - 8}" y="{Y(yv) + 4:.1f}" text-anchor="end" fill="#374151">{yv:.0f}%</text>')
    for k in range(6):
        xv = xmax * k / 5
        out.append(f'<text x="{X(xv):.1f}" y="{H - B + 18}" text-anchor="middle" fill="#374151">{xv:.0f}</text>')
    out.append(f'<line x1="{L}" y1="{H - B}" x2="{W - R}" y2="{H - B}" stroke="#9ca3af"/>')
    out.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H - B}" stroke="#9ca3af"/>')
    out.append(f'<text x="{(L + W - R) / 2}" y="{H - 18}" text-anchor="middle" fill="#111827">'
               f'Benchmark time (minutes, lower is better)</text>')
    out.append(f'<text x="18" y="{(T + H - B) / 2}" text-anchor="middle" fill="#111827" '
               f'transform="rotate(-90 18 {(T + H - B) / 2})">Tune-set score (higher is better)</text>')
    for name, i, x, y, best, base in pts:
        c = COLORS[i % len(COLORS)]
        if best:
            out.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="8" fill="{c}" stroke="#111827" stroke-width="1.5"/>')
        elif base:
            out.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="6" fill="white" stroke="{c}" stroke-width="2.5"/>')
        else:
            out.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4" fill="{c}" fill-opacity="0.45"/>')
    ly = T + 10
    for i, name in enumerate(st["models"]):
        c = COLORS[i % len(COLORS)]
        out.append(f'<circle cx="{W - R + 20}" cy="{ly}" r="6" fill="{c}"/>')
        out.append(f'<text x="{W - R + 32}" y="{ly + 4}" fill="#111827">{html.escape(name)}</text>')
        ly += 22
    ly += 10
    out.append(f'<circle cx="{W - R + 20}" cy="{ly}" r="6" fill="white" stroke="#374151" stroke-width="2.5"/>'
               f'<text x="{W - R + 32}" y="{ly + 4}" fill="#374151">original settings</text>')
    ly += 22
    out.append(f'<circle cx="{W - R + 20}" cy="{ly}" r="8" fill="#9ca3af" stroke="#111827" stroke-width="1.5"/>'
               f'<text x="{W - R + 32}" y="{ly + 4}" fill="#374151">final pick</text>')
    ly += 22
    out.append(f'<circle cx="{W - R + 20}" cy="{ly}" r="4" fill="#9ca3af" fill-opacity="0.6"/>'
               f'<text x="{W - R + 32}" y="{ly + 4}" fill="#374151">other tries</text>')
    out.append("</svg>")
    return "\n".join(out)


def build(cfg: Config) -> Path:
    st = load_state(cfg)
    env = read_json(RUNS_DIR / "environment.json", {})
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    md: list[str] = ["# Results", ""]
    if env:
        md += [f"Machine: {env.get('cpu', '?')}, {env.get('gpu', '?')}, {env.get('ram_gb', '?')} GB RAM. "
               f"llama.cpp {env.get('llama_cpp', '?')}. Python {env.get('python', '?')}.", ""]
    md += [f"Tuning iterations completed: {st['iteration']} (cap {cfg.max_iterations}). "
           f"Tie tolerance: {cfg.tolerance_pts:g} points; a tie only replaces the current pick if it is "
           f"at least {cfg.speed_margin * 100:.0f}% faster.", "",
           "Scores are the weighted share of benchmark tasks passed (agent tasks count double). "
           "\"Time\" is the wall-clock time for the whole task set, excluding model loading. "
           "Write and read speeds are tokens per second, averaged over every request in the run.", ""]

    # Summary table
    md += ["## Summary", "",
           "| Model | Original: score / time | Tuned: score / time | Speed-up | Held-out original | Held-out tuned "
           "| Held-out speed-up | Status |",
           "|---|---|---|---|---|---|---|---|"]
    best_configs = {}
    for name, ms in st["models"].items():
        model = cfg.models.get(name)
        base = ms["runs"][0] if ms["runs"] else None
        inc = run_by_id(ms, ms["incumbent"])
        fin = st.get("final", {}).get(name, {})
        fb, ft = fin.get("baseline", {}), fin.get("tuned", {}) or fin.get("baseline", {})

        def fscore(f):
            s = (f or {}).get("summary")
            return f"{s['score']:.1f}% / {fmt_s(s['wall_s'])}" if s else "-"

        speedup = (f"{base['wall_s'] / inc['wall_s']:.2f}x" if base and inc and base.get("ok") and inc.get("ok")
                   else "-")
        sb, st_ = (fb or {}).get("summary"), (fin.get("tuned") or {}).get("summary")
        final_speedup = f"{sb['wall_s'] / st_['wall_s']:.2f}x" if sb and st_ else "-"
        md.append(f"| {name} | {_f(base and base.get('score'))}% / {fmt_s(base and base.get('wall_s'))} | "
                  f"{_f(inc and inc.get('score'))}%{_averaged(inc)} / {fmt_s(inc and inc.get('wall_s'))} | {speedup} | "
                  f"{fscore(fb)} | {fscore(ft) if fin.get('tuned') else ('same settings' if fb else '-')} | "
                  f"{final_speedup} | {ms['status']} ({len(ms['runs'])} runs) |")
        if inc and model:
            best_configs[name] = {"flags": inc["flags"], "changes_vs_original": describe_diff(model.baseline, inc["flags"]),
                                  "tune_score": inc["score"], "tune_wall_s": inc["wall_s"], "run_id": inc["run_id"]}
            if _averaged(inc):
                best_configs[name]["tune_score_averaged"] = inc["pooled_score"]
            held_out = {k: {"score": f["summary"]["score"], "wall_s": f["summary"]["wall_s"]}
                        for k, f in fin.items() if f.get("summary")}
            if held_out:
                best_configs[name]["held_out"] = held_out
    md.append("")
    svg = svg_tradeoff(st, cfg)
    if svg:
        (RESULTS_DIR / "tradeoff.svg").write_text(svg, encoding="utf-8")
        md += ["![Score against time for every run](tradeoff.svg)", ""]

    # Per-model detail
    for name, ms in st["models"].items():
        model = cfg.models.get(name)
        md += [f"## {name}", ""]
        if model and model.notes:
            md += [model.notes, ""]
        md += ["| It | Change tried | Score (95% CI) | Time | Write tok/s | Read tok/s | Thinking tokens | Cut off | Decision |",
               "|---|---|---|---|---|---|---|---|---|"]
        for r in ms["runs"]:
            mark = " **(pick)**" if r["run_id"] == ms["incumbent"] else ""
            ci = f" ({r['ci95'][0]:.0f}-{r['ci95'][1]:.0f})" if r.get("ci95") else ""
            md.append(f"| {r['iteration']} | {_md_escape(r.get('change', r['label']))} | {_f(r.get('score'))}%{ci}{mark} | "
                      f"{fmt_s(r.get('wall_s'))} | {_f(r.get('gen_tps'))} | {_f(r.get('pp_tps'), '{:.0f}')} | "
                      f"{_f(r.get('reasoning_tokens'), '{:,.0f}')} | {_f(r.get('truncated'), '{:.0f}')} | "
                      f"{_md_escape(r.get('decision', ''))} |")
        md.append("")
        inc = run_by_id(ms, ms["incumbent"])
        if inc and inc.get("by_category"):
            md += ["Per category for the pick: " + ", ".join(
                f"{c} {v['passed']}/{v['total']}" for c, v in inc["by_category"].items()), ""]
        pr = ms.get("probe")
        if pr and pr.get("results"):
            md += ["Speed calibration probes (16K-token prompt; s/turn = 3000/read + 500/write@16K):", "",
                   "| Setting | Read tok/s | Write@16K tok/s | Write@0 tok/s | s/turn |", "|---|---|---|---|---|"]
            for p in pr["results"]:
                if p.get("error"):
                    md.append(f"| {_md_escape(p['label'])} | failed: {_md_escape(p['error'][:80])} | | | |")
                else:
                    md.append(f"| {_md_escape(p['label'])} | {p['pp_tps']:.0f} | {p['tg_deep_tps']:.1f} | "
                              f"{p['tg_short_tps']:.1f} | {p['turn_s']:.2f} |")
            md.append("")
        if inc and model:
            md += [f"Pick vs original launcher: `{describe_diff(model.baseline, inc['flags'])}`", "",
                   "Full flags: `" + describe(inc["flags"]) + "`", ""]
        fin = st.get("final", {}).get(name, {})
        if fin:
            md += ["Held-out (final) test:", "",
                   "| Settings | Score (95% CI) | Time | Write tok/s | Cut off | Solved/hour |", "|---|---|---|---|---|---|"]
            for key, f in fin.items():
                s = f.get("summary")
                if s:
                    md.append(f"| {key} | {s['score']:.1f}% ({s['ci95'][0]:.0f}-{s['ci95'][1]:.0f}) | {fmt_s(s['wall_s'])} | "
                              f"{_f(s['gen_tps'])} | {_f(s.get('truncated'), '{:.0f}')} | {_f(s['solved_per_hour'])} |")
                else:
                    md.append(f"| {key} | failed: {_md_escape((f.get('error') or '').splitlines()[0][:160] if f.get('error') else '')} | | | | |")
            md.append("")
    md += ["## How to read this", "",
           "* With about 50 tasks, one task is about 2 points, so differences of a few points are within noise "
           "(see the 95% intervals). The tuner treats anything within the tie tolerance as equal and then prefers "
           "the faster setting.",
           "* The held-out set was never used while tuning, so it is the fairest comparison of original vs tuned. "
           "Its tasks are different, so compare original and tuned within one set, not scores across the two sets.",
           "* Speed settings (threads, batch size, load mode, VRAM margin) cannot make a model more or less accurate; "
           "they were chosen with short speed probes rather than full benchmarks. They do change the exact arithmetic, "
           "so the sampled answers differ from run to run: the speed-calibrated run counts as the average of itself "
           "and the run it was built from.",
           "* \"Cut off\" counts requests that reached the output cap before the model finished; that answer counts "
           "as a failure (see docs/METHOD.md, \"Output caps\")."]
    lite = [n for n, m in cfg.models.items() if m.suite == "lite"]
    if lite:
        n_tune, n_final = (len(select(load_tasks(sp), "lite")) for sp in ("tune", "final"))
        md.append(f"* {', '.join(lite)} ran the `lite` subset ({n_tune} tuning tasks, {n_final} held-out) because it "
                  f"writes only a few tokens per second. There, one task is {100 / n_tune:.0f} to {100 / n_final:.1f} "
                  "points, so any lost task puts a change outside the tie band, and its held-out comparison is only a "
                  "rough check.")
    unsolved = unsolved_final_tasks(st)
    if unsolved:
        md.append("* No held-out run passed " + ", ".join(f"`{t}` ({c}, {n} runs)" for t, c, n in unsolved) +
                  ". A task that every setting fails cannot tell settings apart, and it may be broken rather than "
                  "hard; see docs/METHOD.md, \"Held-out test\".")
    md.append("")
    (RESULTS_DIR / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    write_json(RESULTS_DIR / "best_configs.json", best_configs)

    with open(RESULTS_DIR / "iterations.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "iteration", "label", "change", "ok", "score", "ci_low", "ci_high", "wall_s", "gen_tps",
                    "pp_tps", "long_pp_tps", "reasoning_tokens", "truncated", "solved_per_hour", "decision", "pick"])
        for name, ms in st["models"].items():
            for r in ms["runs"]:
                ci = r.get("ci95") or [None, None]
                w.writerow([name, r["iteration"], r["label"], r.get("change"), r["ok"], r.get("score"), ci[0], ci[1],
                            r.get("wall_s"), r.get("gen_tps"), r.get("pp_tps"), r.get("long_pp_tps"),
                            r.get("reasoning_tokens"), r.get("truncated"), r.get("solved_per_hour"),
                            r.get("decision"), r["run_id"] == ms["incumbent"]])
    return RESULTS_DIR / "REPORT.md"
