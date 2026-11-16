# Method

## What is measured

Each benchmark run starts `llama-server` with one set of launch flags, runs a fixed task set against
its OpenAI-compatible API, and stops the server. Every request uses a fixed seed derived from the
task id, so two runs of the same settings see the same prompts in the same order with the same seeds.

**Accuracy** is the weighted share of tasks passed (strict: a coding task passes only if every hidden
check passes). Five task types, chosen to look like the work a coding agent does:

| Type | What the model does | How it is graded |
|---|---|---|
| `toolcall` | Answers one turn of a conversation with tool definitions (read/edit/search/run, tickets) | The right tool with the right arguments (paths normalized, numbers, enums, exact text, JSON), or a direct answer when no tool is needed |
| `codegen` | Writes a function from a precise spec | Hidden unit checks run in a guarded subprocess; many problems have deliberate traps (float rounding, tie-breaking, edge cases, linear-time limits) |
| `bugfix` | Fixes a buggy function, sometimes with a symptom | Hidden checks; validation proves the original bug fails them |
| `longctx` | Traces a call through a generated 8K-24K-token code base | Exact integer answer, computed by executing the generated code |
| `agent` | Fixes a small repo with list/read/write/run-tests tools over up to 14 turns | The original tests are restored, then visible + hidden tests must all pass |

All problems are original. Expected answers never come from a model: they come from reference
solutions (checked by `iterbench validate`) or from executing generated code.

**Speed** is wall-clock time for the whole task set (excluding model load), plus the server's own
timings for prompt reading (prefill) and writing (generation). The time is what you feel as a user:
a setting that makes the model think twice as long costs time even if tokens per second do not change.

**Output caps.** Every request has a fixed `max_tokens`, thinking included: 8192 for coding and bug-fix
tasks, 6144 for long-context and agent turns, 4096 for tool calls (the 27B gets half, because it writes
about 3 tokens per second). A model that is still thinking when it reaches the cap gets no answer
and fails the task, like a time limit in a programming contest. This keeps every run finite, and it
makes long thinking a real cost: on the baselines, most coding failures came from models that ran
out of tokens before writing any code, not from wrong code (10 of qwen3.6-35b's 12 coding failures).
So the baseline scores are accuracy within these limits. With unlimited tokens, the long-thinking
baselines would solve some of those tasks, but each would take several minutes.

## Splits

* `tune` (49 tasks, weight 52): used for every tuning iteration.
* `final` (30 tasks, weight 32): held out. Only the final test runs it, so the before/after comparison
  is not inflated by tuning on the same questions.
* `lite`: a subset of both for very slow models (marked `lite = true`).

## The tuning loop

1. **Iteration 1: baseline.** The model's original launcher settings.
2. **Thinking first.** A 4096-token thinking budget runs first, as a reference: when the budget runs
   out, llama.cpp closes the thinking block and the model has to answer (`scripts/smoke_budget.py`
   checks this), so the score shows what the
   model achieves when it must answer within the output cap (the baselines often hit the cap before
   answering, so they understate it). Then thinking off and budgets of 1024 / 2048 tokens, fastest
   first, stopping at the first one within the tie band of the best accuracy seen. This goes first
   because thinking length dominates run time (on its baseline, about three quarters of the 35B's
   time went to the coding tasks, almost all of it thinking), so every later iteration gets shorter. The 27B tries only thinking off
   and budget 1024: at about 3 tokens per second, every 1000 thinking tokens cost it more than five
   minutes per answer.
3. **Speed calibration (probes are not iterations).** Settings that cannot make the model more or
   less accurate (CPU threads, load mode, micro-batch size, the VRAM margin used by `--fit`) are
   chosen with a short probe: read a 16K-token prompt, then write tokens at that depth and at short
   depth. Score = estimated seconds per agent turn (`3000 / prefill + 500 / write@16K`). Each setting
   is kept only if it is at least 2% faster; a probe request that takes over 15 minutes counts as a
   failed probe. Then one benchmark run with the calibrated settings. That run has the same
   answer-affecting settings as the run it was built from, so the two are measurements of one
   configuration and its accuracy counts as their average (see Noise). It is kept if that average is
   within the tie band and the run is not more than 10% slower than the pick (a slower run would mean
   the probe did not predict real use). Once kept, older runs without these settings can no longer
   become the pick.
4. **Then one answer-affecting change per iteration**, built on the current pick: the vendor's
   recommended sampling preset and KV-cache precision (q4_0 / f16), plus model-specific extras.
5. A model stops when every planned change has been tried, at its own cap, or at the global cap of
   20 iterations.

**Selection rule:** among all runs of a model whose accuracy is within the tie tolerance (4 points)
of the best accuracy seen for that model, the pick is the fastest. A new run must be at least 5%
faster to replace a pick that is still in that group. Comparing against the best accuracy seen (not
the last pick) stops a chain of small losses from adding up.

## Noise

With ~50 tasks, one task is ~2 points and a 95% interval is roughly +/-13 points. Within one model,
runs use the same tasks and seeds, but that does not make them repeatable: any change in the order of
floating-point operations (another thread count, batch size or GPU/CPU split) shifts the token
probabilities slightly, and once one sampled token differs, the rest of the answer differs.

The speed-calibrated runs measured this directly (`scripts/task_matrix.py` lists every task that
changed between runs). Each differs from the run it was built from only in
settings that cannot change accuracy, and in each full-suite pair 7 or 8 of 49 tasks changed result:
ornith-35b lost 4 and gained 4 (same score), qwen3.6-35b lost 5 and gained 3 (92.3% -> 88.5%), ornith-9b
lost 2 and gained 5 (75.0% -> 80.8%). So a single run's
score has a standard deviation of roughly 2 tasks (about 4 points), about the size of the tie band.
Decisions far outside the band (thinking off: 8 to 25 points lower) are solid; decisions near its edge
are close calls. That is why ties are treated as ties and broken by speed, and why the held-out final
test is the fairest single comparison.

Run time is noisy too. Speed per token is measured precisely, but the number of tokens is not: an
answer cut off at the output cap costs the whole cap, which for the 9B is about 2.2 minutes instead of
the usual half minute, so one cut-off answer adds about 7% to a run. And the best accuracy seen tends to
rise as runs accumulate, because it is the luckiest of several noisy runs. Both showed in the 9B's last
iterations:

* Iteration 9 (full-precision KV cache) scored 82.7% but took 50% longer, so it was rejected. Its score
  raised the bar to 78.7%, which left the pick (77.9%, averaged) outside the band, so the pick moved to
  iteration 8 (4-bit KV cache, 78.8%), the fastest run still in it.
* Iteration 10 (repeat penalty off, built on iteration 8) was 9% faster than iteration 8 and was kept.
  Most of that came from writing 7% fewer tokens (iteration 8 had one answer cut off, iteration 10 none);
  generation itself was 2.6% faster.

Every step followed the rule, but the 9B's last four settings are tied within noise in both accuracy
and time. Its pick is the fastest of them on these runs, not a clearly better setting.

## Held-out test

After tuning, each model's original settings and its pick ran the 30 held-out tasks once (the 27B:
its 8 lite tasks). Every pick was faster (2.4x to 3.5x) and scored at least as high. The original
settings had 2 to 9 answers cut off at the output cap and the picks none, which accounts for most of the
accuracy difference. The ornith gains (3.1 points each) are within noise; qwen3.6-35b's (18.8 points,
six tasks) is not; the 27B's (two of eight tasks, both cut off under the original settings) is a rough
check.

Three of the four models score lower on the held-out tasks than on the tuning tasks (ornith-35b's
original settings: 88.5% on the tuning tasks, 65.6% held out). The two sets have different tasks, so
the held-out numbers compare settings with each other, not with the tuning numbers.

Every held-out task that most runs failed was checked by hand (`scripts/task_matrix.py --final` lists
them with each failure):

* **`fbf04` has a defective check.** The task is to fix a merge of two sorted lists whose docstring
  requires stability (on equal items, the one from `a` comes first). Hidden checks 6 and 7 test that
  with a key class that defines only `<` and `==`. The textbook fix, `if a[i] <= b[j]:`, raises
  TypeError on that class, because Python does not derive `<=` from `<` and `==`, and the prompt never
  says the items support only `<`. Five of the six runs that had the task wrote exactly that fix,
  correct otherwise; the sixth (ornith-9b's pick) also left out the end of `b`. No run passed. The
  scores stay as measured: the task file was frozen before the first run, and changing a check after
  seeing results is what a held-out set exists to prevent. With a check that accepts `<=`:

  | Model | Original | Tuned | As measured |
  |---|---|---|---|
  | ornith-35b | 68.8% | 71.9% | 65.6% / 68.8% |
  | ornith-9b | 75.0% | 75.0% | 71.9% / 75.0% |
  | qwen3.6-35b | 65.6% | 84.4% | 62.5% / 81.2% |

  So the 9B's held-out gain becomes a tie, and nothing else changes (the 27B's lite subset does not
  include the task). For a new version of the task set, give the key class a `__le__` (the original
  bug still fails the checks) or say in the docstring that items support only `<`.
* **`ftc08` is a judgment call, not a defect.** Asked to install the package in editable mode with
  its dev extras, both ornith models listed the directory first (`ls -la`) in all four runs. The task
  accepts the install command or reading the package config (change 1 below, settled before any
  held-out run). The result is the same in all four runs, so it does not change the comparison, but it
  costs each ornith held-out score 3.1 points. Both qwen3.6-35b runs passed. The tuning set's `tc02`
  was the same kind of call (change 1).
* **The others are fair.** `temperature` (agent task, passed only by ornith-9b's pick): the docstring
  says to raise ValueError for anything that is not a reading, and the hidden tests check readings
  such as `"12K"` and `""`. `fcg11` (ZigZag varint): three picks wrote the same wrong encoding of
  2**63, from the 64-bit idiom `(n << 1) ^ (n >> 63)`, which the prompt rules out (it gives the exact
  formula and says to support arbitrarily large integers). `fcg10` (matrix rotation): two picks gave
  the same wrong answer; three other runs passed.

## Options tested and left out

**MTP speculative decoding** (`--spec-type draft-mtp`). Three of the models ship the multi-token
prediction layer, so llama.cpp can draft several tokens and verify them in one pass. A smoke test
(`scripts/smoke_spec.py`, thinking off, two short prompts per model) found no gain worth a benchmark
run on this laptop:

| Model | Code answer | Prose answer |
|---|---|---|
| ornith-35b | 34.4 -> 34.9 tok/s (79% of drafts accepted) | 38.9 -> 29.7 tok/s (44% accepted) |
| qwen3.8-27b | 3.08 -> 3.83 tok/s (97% accepted) | 3.21 -> 2.73 tok/s (61% accepted) |

Verifying a batch of drafted tokens costs almost as much as writing them one by one when much of the
model runs on the CPU (MoE experts for the 35B, about half the layers for the 27B). Thinking is
mostly prose, so the net effect on these benchmarks would be negative. The 9B, the one model that
fits entirely on the GPU, has no MTP layer. On a GPU with room for the whole model, MTP is worth
re-testing.

## Changes made during the run

Judgment calls made after tuning started, in order:

1. **Tool-call grading relaxed before any result was kept.** In the first minute of iteration 1, the
   35B searched the code before editing in `tc04`, which is a sensible agent move. `tc04`, `tc08` and
   `ftc08` now accept the reasonable alternatives. The partial run was deleted and tuning restarted.
   `tc02` ("Run only the tests in tests/test_api.py, with verbose output") was left strict, then and
   after the baselines, where both ornith models had explored the project instead of running the
   tests: the instruction says exactly what to run. Over all tuning runs, ornith-9b failed it every
   time, ornith-35b in 4 of 8 runs, and qwen3.6-35b never.
   From then on, the task files are frozen by a SHA-256 fingerprint stored in the tuner state, and the
   tuner refuses to continue if they change.
2. **Thinking moved before speed calibration.** The first baseline took 41.5 minutes instead of the
   expected 15-25, almost all of it thinking on coding tasks. Testing thinking settings first makes
   every later run shorter. Only the order changed; the set of candidates and the selection rule did
   not.
3. **Budget 4096 as a reference.** The baselines showed that most coding failures were answers that
   ran out of tokens while thinking. With "thinking off first, stop at the first accepted", thinking
   off would be compared only against those understated baselines. Running budget 4096 first
   measures what the model achieves when it has to answer in time.
4. **Speed-run check.** The speed-calibrated run was kept whenever accuracy held. It is now also
   rejected if it is more than 10% slower than the pick.
5. **Probe timeout.** After iteration 3, a probe request that takes more than 15 minutes fails the
   probe (before, it could wait up to 2 hours), in case a setting such as load-mode none makes a 35B
   model page to disk. No probe had come close: the slowest took under 2 minutes in total.
6. **Speed-calibrated run counted as an average.** In iteration 5, qwen3.6-35b's speed-calibrated run
   scored 88.5% against 92.3% for the run it was built from (5 tasks lost, 3 gained). The old rule
   compared that single score with the best run seen (94.2%) and rejected it, throwing away settings
   that made a run 13% faster (22% per agent turn in the probes) over sampling noise; ornith-35b's
   pair showed the same noise at an equal score. Since speed settings cannot change accuracy, the speed
   run now counts as the average of the two runs, and a kept speed run stays the pick until a newer run
   qualifies (before, the pick could fall back to an older run without the speed settings). Re-applied
   to all three speed runs so far, only qwen3.6-35b's decision changed: 90.4% against a band that
   starts at 90.2%, a close call. The old state is kept in `runs/tuner_state.before-pooling.json`, and
   the held-out final test checks the result.
7. **The average also sets the bar.** In iteration 6, ornith-9b's speed run scored 80.8% (pooled with
   its parent: 77.9%), and the code still took the single 80.8% as the best accuracy seen, so every
   later 9B run had to reach 76.8% instead of 74.8%. That contradicted change 6 (the speed run's
   accuracy *is* the average), so the best accuracy is now computed from the same averaged numbers. No
   pick changed; the only decision it touched (ornith-9b iteration 7, 71.2%) is rejected either way.
8. **Pick moves are logged.** In iteration 9 the 9B's pick moved to another run (see Noise), and nothing
   said so: the run that caused it was correctly marked rejected, and the new pick's own entry still
   read "rejected" from iteration 8. The tuner now adds the move to the decision of the run that
   caused it and records it in the state. The note was added to that run afterwards (old state:
   `runs/tuner_state.before-pick-notes.json`). No decision changed.
9. **Failed starts name their cause.** A server that failed to start was reported with the end of its
   log, which began with harmless warnings. The decision now quotes the first error line (for
   qwen3.6-35b's iteration 8: `CUDA error: out of memory`), and that entry was rewritten from its server
   log after the final test (old state: `runs/tuner_state.before-failed-cause.json`). No decision
   changed.

## Safety of running model-written code

Model code runs in a separate Python process with an audit hook that blocks starting processes,
network sockets, ctypes and writes outside a scratch folder; on Windows a job object also caps memory
at 1 GB and forbids child processes, and every run has a timeout. This is a guard rail for benign
benchmark code, not a security sandbox.
