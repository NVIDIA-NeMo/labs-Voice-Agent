---
name: "eval-result-analyzer"
description: "Analyze a NeMo-Voice-Agent eval result directory and produce a thorough structured report. Reads every artifact under the run directory (judge_result.json, final_agent_response.json, final_scenario_db_hash.txt, metrics.json, conversation_log.{txt,seglst.json}, bridge_log.txt, bot_logs_{agent,user}/llm_context.json, scenario_config/). Always performs latency analysis for BOTH agent and simulated-user sides separately. Compares user-sim intended utterances against agent-received STT to surface ASR transcript differences. Traces tool calls against the scenario's gold reference_answer. Classifies every issue by root cause (user-sim / agent / ASR / framework). Writes analysis_report.md back to the same input directory. Use when the user references an eval-result path, asks 'analyze this run' / 'why did this scenario fail' / 'audit this result folder', or pastes any path that looks like an eval-results directory."
license: "Apache-2.0"
metadata:
  tags:
    - voice-agent
    - evaluation
    - tau2
    - eva_airline
    - debugging
    - latency
    - asr
---

# Eval-Result Analyzer

## Goal

Given a path to an eval result directory, produce a thorough audit covering scoring signals, latency breakdown for **both sides** (agent and simulated user), ASR transcript diff (distinguishing benign normalization from real mistranscription), tool-call execution tracing, and root-cause classification. The output is a single `analysis_report.md` written back to the input directory.

## Input modes

The input path can be either:

| Mode | Looks like | Behavior |
|---|---|---|
| **Single scenario** | A directory containing `judge_result.json` directly (e.g. `tau2_retail__16/`). | Analyze just that scenario; write `analysis_report.md` inside it. |
| **Full run** | A directory containing `all_metrics.json` + `all_summary.txt` + per-scenario subdirectories. | Analyze every scenario subdir individually (write each per-scenario `analysis_report.md`), then write a run-level `analysis_report.md` at the top of the run dir that aggregates findings. |
| **Partial run** | Scenario subdirectories exist but `all_summary.txt` / `all_metrics.json` are MISSING at the top level. | Same as Full run, with three additional steps: (1) disambiguate "still running" vs "killed" via `evaluation_log.txt` mtime, (2) aggregate every headline rate from per-scenario `metrics.json` files (not `all_summary.txt`), (3) stamp the run-level report's title with `(PARTIAL RUN — N scenarios analyzed)` plus either `did not complete normally` or `STILL RUNNING — snapshot at HH:MM` based on the mtime check. See [Partial-run handling](#partial-run-handling) below. |

Detect mode by checking whether the path contains scenario subdirectories with `judge_result.json` (run-level) or has `judge_result.json` directly (scenario-level). Distinguish **Full** vs **Partial** at the run level by whether `all_summary.txt` exists in the input directory — `all_summary.txt` and `all_metrics.json` are written only at the end of the run loop (`runner.py`'s final block), so their absence means the run either didn't complete OR is still actively running. The mtime of `evaluation_log.txt` disambiguates those two cases (see Partial-run handling step 1). Per-scenario `metrics.json` is written incrementally so each completed scenario has full per-scenario artifacts regardless.

### Partial-run handling

When `all_summary.txt` is absent:

1. **Disambiguate "killed mid-run" vs "still actively running"** before anything else. The filesystem alone can't tell them apart — both produce identical "subdirs without top-level aggregates" shapes. Check `evaluation_log.txt`'s modification time:

   ```bash
   # Seconds since evaluation_log.txt was last modified.
   echo $(( $(date +%s) - $(stat -c %Y evaluation_log.txt) ))
   ```

   - **< ~300 s** (5 minutes): the run is **STILL ACTIVE**. The runner writes a fresh log line at least every scenario start and during scenario execution; quiescence of <5 min effectively never happens between events while the run is alive. Surface this prominently in the report header (e.g. `(STILL RUNNING — snapshot at HH:MM)`), and tell the user explicitly that their counts will move; recommend re-invoking the skill after the run finishes (when `all_summary.txt` appears) for a final pass.
   - **> ~300 s**: the run is **likely killed / errored / hung**. Proceed with normal partial-run aggregation. Optionally note in the report when the last log line was written and the in-flight scenario name (if any) so the operator knows where to resume.

   The mtime heuristic is approximate — a slow LLM call (we've seen up to ~600 s user-side TTFB in real runs) can make an active run look quiescent. If the mtime is in the 5–15 min ambiguous band, hedge: "log idle for X min; the run is probably stalled but may be in a long LLM call". When in doubt, **prefer the still-running framing** — a false-positive "killed" claim is harder to walk back than a false-positive "still running" hedge.

2. **Derive the intended N from `evaluation_log.txt` at the run-dir top level.** The runner logs `Starting Scenario K/N: <name>` for every scenario as it begins. The largest `K/N` line gives both the last-started scenario and the intended total N — even if the run died mid-scenario K. Use:

   ```bash
   grep -oE 'Starting Scenario [0-9]+/[0-9]+' evaluation_log.txt | tail -1
   # → e.g. "Starting Scenario 67/114"   (intended N = 114, last started = 67)
   ```

   Count completed scenarios as the number of subdirs with a `metrics.json` (incrementally written *after* each scenario finishes), not the largest K from the log (which counts the in-flight scenario too). Report both: `N_completed / N_intended (X.X%) — Y unrun, 1 may be partially in-flight`. Fall back to `run_evaluation.sh` / CLI flags in the user's prompt only if `evaluation_log.txt` is missing or unparseable.

   **Snapshot drift caveat for still-running runs**: the `N_completed` count is a point-in-time read. By the time the report is finalized, more scenarios may have completed. Either (a) re-aggregate immediately before writing the report so the numbers are as fresh as possible, OR (b) add a "Snapshot freshness" footer at the end of the report giving the re-aggregated numbers if they've drifted significantly. Patterns (dominant failure mode, anomalies, etc.) are typically stable across a few-scenario drift; exact rates are not.
2. **Aggregate ALL rates from per-scenario `metrics.json` directly.** Don't try to read `all_summary.txt` — compute:
   - `Overall Success Rate` — fraction of scenarios with `metrics.json["is_successful"] is True`, denominator = scenarios with at least one applicable signal (skip those where `is_successful == "N/A"`).
   - `DB-State Match Rate` — fraction with `metrics.json["db_state_match"] is True`, denominator = scenarios with `db_state_match` present.
   - `DB-State Assertion Pass Rate` — mean of `metrics.json["db_state_assertion_pass_rate"]` over scenarios that opt in.
   - `NL-Assertion Pass Rate` — sum of passed assertions / sum of total assertions across scenarios with `nl_assertions`.
   - `Mean Judge Score` — mean of `judge_result.json["score"]` over scenarios with a judge result.
   - `Token usage` totals — sum of per-scenario `metrics.json["token_usage"]` blocks.
3. **Surface the partial-run framing prominently** in the run-level report's opening paragraph so a reader doesn't mistake the numbers for a full sample. Example header: `# Run analysis: eval_20260610_HHMMSS (PARTIAL RUN — 100/114 scenarios analyzed, 87.7% completion)`.
4. **Note: per-scenario analyses are unaffected.** Each completed scenario subdir still has its full artifact set (`metrics.json`, `judge_result.json`, `bot_logs_*`, `bridge_log.txt`, `scenario_config/`), so the per-scenario `analysis_report.md` is unchanged for Partial-run mode.

## File inventory — read ALL of these per scenario

Every file in a scenario directory is informative; skip none unless explicitly noted.

| File | What you get from it |
|---|---|
| `judge_result.json` | Judge `score`, `reason`, per-assertion `nl_assertion_verdicts`, `nl_assertion_pass_rate` (when scenario has nl_assertions). |
| `final_agent_response.json` | The agent's recorded action list (one entry per `_record_action` call). Compare against reference. |
| `final_scenario_db_hash.txt` | Actual post-run DB hash; compare against scenario's `expected_scenario_db` hash. |
| `metrics.json` | `total_turns`, `scenario_duration`, `db_state_match`, `nl_assertion_pass_rate` (when applicable), `latency_stats`, full per-turn `latencies` list (one entry per agent response). |
| `conversation_log.txt` | Human-readable turn-by-turn AUDIO transcript: `[start_s - end_s] (duration_s) ROLE: text` plus `→ Response latency: NNNms` for agent turns. Also has `Stop reason: [EXIT \| TIMEOUT \| ...]`. **Audio turns ONLY** — does NOT show tool calls, agent reasoning, system prompt, or anything that happened in the agent's processing time between audio turns. NEVER draw conclusions about tool-call behavior or agent reasoning from this file alone — those live in `bot_logs_agent/llm_context.json`. |
| `conversation_log.seglst.json` | Same conversation in structured JSON (one entry per VAD segment); use when the text log isn't enough. |
| `conversation_log.wav` | The full audio mix-down. **Do not read** — note its presence and offer to point the user to it if acoustic issues are suspected. |
| `bridge_log.txt` | Lowest-level event log from the bridge orchestrator: TTFB metrics for `NvidiaLLMService`, `NemotronTTSService`, STT events, RTVI actions, tool-registration errors, prompt-token/completion-token counts. Use grep — it can be tens of thousands of lines. **Logs from bots that include the `_log_rtvi_event` helper carry side-tagged DEBUG lines**: `[AGENT EVENT] type=<X>` / `[USER EVENT] type=<X>` for every RTVI message type, plus expanded payload lines `[AGENT METRICS] ttfb processor=... value=...s`, `[AGENT METRICS] tokens prompt=... completion=...`, `[AGENT STT final|partial] '<text>'`, `[AGENT ACTION] <name>`. Always check for these tags first before falling back to time-correlation against the unstructured `ProtobufFrameSerializer:deserialize` dumps. |
| `bot_logs_agent/llm_context.json` | Final state of the **agent** bot's LLM context: list of `{role, content, tool_calls?}` messages. Tool calls visible in `assistant` messages with `tool_calls` field; tool results in `tool` messages. |
| `bot_logs_user/llm_context.json` | Final state of the **user-sim** bot's LLM context. `assistant` messages here = what the user-sim's LLM **generated** to be spoken. Pair with the agent context's `user` messages (= what STT delivered) to do the ASR transcript diff. |
| `scenario_config/agent_prompt.txt` | Exact agent system prompt for this scenario. |
| `scenario_config/user_prompt.txt` | Exact user-sim system prompt for this scenario. |
| `scenario_config/agent_tools.json` | Tool names the runner sent to the agent bot for this scenario. |
| `scenario_config/user_tools.json` | Tool names sent to the user-sim bot (usually empty `"{}"`). |
| `scenario_config/reference_answer.json` | Gold-replay action list — what the agent SHOULD have executed. |

## Cross-reference against the scenario class (gold replay)

If you need the **expected_scenario_db hash** to verify DB match, or want the actual gold reference actions in case `scenario_config/reference_answer.json` is stale, import the scenario class:

```bash
.venv/bin/python << 'PY'
import json
import nemo_voice_agent.evaluation.scenarios.data  # triggers @register_eval_scenario discovery
from nemo_voice_agent.evaluation.scenarios import ALL_EVAL_SCENARIOS
from nemo_voice_agent.evaluation.db_hash import get_dict_hash

inst = ALL_EVAL_SCENARIOS['<scenario_name>']()
print('expected_db_hash:', get_dict_hash(inst.expected_scenario_db))
print('reference_actions:')
for a in inst.reference_answer['actions']:
    print(f'  - {a["name"]}({a.get("arguments")})')
print('nl_assertions:', getattr(inst, 'nl_assertions', None))
PY
```

Use this when an `actual` and `expected` DB hash disagree to understand which mutations are missing/extra.

## Analysis phases

Execute these in order. Don't skip phases — each surfaces a different class of issue.

### Phase 0 — Read the scenario's scoring contract FIRST

Before judging any "missing" file or "absent" signal as a framework bug, read the scenario's `success_signals` whitelist. Different domains gate on different signals; treating an informational signal as a gating one (or vice versa) produces deeply misleading reports.

```bash
# The whitelist is persisted on every scenario alongside metadata.json:
.venv/bin/python -c "
import json
m = json.load(open('<scenario_dir>/scenario_config/metadata.json'))
print('success_signals:', m.get('success_signals'))
print('domain:', m.get('domain'))
"
```

Or import the scenario class (see [Cross-reference against the scenario class](#cross-reference-against-the-scenario-class-gold-replay)).

**The five signals and what each requires on disk:**

| Signal (enum value in `success_signals`) | Requires for scoring | Output file when scored |
|---|---|---|
| `is_action_match` (`ACTION_MATCH`) | `scenario_config/reference_answer.json` + `final_agent_response.json` (both exist + parseable) | `metrics.json["is_action_match"]` |
| `db_state_match` (`DB_STATE_MATCH`) | `scenario.expected_scenario_db` (in-process) + `final_scenario_db_hash.txt` (from bridge pull) | `metrics.json["db_state_match"]` + `db_state_expected_hash` / `db_state_actual_hash` |
| `db_state_assertion` (`DB_STATE_ASSERTION`) | `scenario.db_state_assertions` (in-process) + bridge-pulled inline DB | `metrics.json["db_state_assertion_pass_rate"]` + `db_state_assertion_verdicts` |
| `nl_assertion` (`NL_ASSERTION`) | `scenario.nl_assertions` + judge configured + judge call succeeded | `judge_result.json["nl_assertion_verdicts"]` + `metrics.json["nl_assertion_pass_rate"]` |
| `judge_passed` (`JUDGE_PASSED`) | `--judge-url` + `--judge-threshold` + `reference_answer.json` + `final_agent_response.json` | `judge_result.json["score"]` + `metrics.json["judge_passed"]` |

**Critical rule — absent ≠ broken.** A signal that is *not* in the whitelist may legitimately produce no artifact on disk:

- `reference_answer.json` **missing** is only a framework finding if `ACTION_MATCH` or `JUDGE_PASSED` is in the whitelist AND the scenario class returns a non-None `reference_answer`. For domains that gate purely on `DB_STATE_MATCH` (eva_airline, tau2_airline) or `DB_STATE_ASSERTION` (tau2_telecom), missing reference is expected — those domains may not author one at all, especially for scaffolded scenarios.
- `judge_result.json` **missing** is only a framework finding if `JUDGE_PASSED` or `NL_ASSERTION` is in the whitelist AND `--judge-url` was passed AND the scenario has a `reference_answer`. Otherwise it's expected absence — the judge wasn't relevant to scoring.
- `is_action_match` = `"N/A"` is only a framework finding if `ACTION_MATCH` is in the whitelist. For domains where it isn't, "N/A" is the correct value.

**Practical implication for the report:** in Phase 1's headline table, label each row with `(gating)` or `(informational)` based on the whitelist. In Phase 7's root-cause attribution, never use the `framework` label for missing files that the scenario's contract doesn't require — those go in `agent` (the agent didn't do the work) or simply don't appear as findings.

When in doubt, lean toward "this is expected absence" rather than "this is a framework bug." False-positive bug claims send the operator on goose chases.

### Authoritative source per question type — DO NOT skip this table

Different files answer different questions about what happened. Using the wrong file produces wrong conclusions. **Cross-reference the relevant authoritative file before drawing any conclusion** — especially before claiming the agent / user-sim didn't do something.

| Question | Authoritative source | What NOT to use as primary source |
|---|---|---|
| **Did the agent call any tools? Which ones? With what arguments?** | `bot_logs_agent/llm_context.json` — walk `assistant` messages for `tool_calls` field. | NOT `conversation_log.txt`. Tool calls happen in the agent's processing time between audio turns and are INVISIBLE in the conversation log. The conversation log only shows audio TURNS. |
| **What did the agent's LLM actually generate (incl. reasoning)?** | `bot_logs_agent/llm_context.json` — `assistant` messages with `content` and `reasoning_content` (when reasoning mode is on). | NOT `conversation_log.txt`. That shows post-STT-on-the-other-side text — agent reasoning + Markdown-stripped content is not preserved there. |
| **What did the user-sim's LLM intend to say (pre-TTS)?** | `bot_logs_user/llm_context.json` — `assistant` messages. | NOT `conversation_log.txt`. The conversation log shows what STT delivered on the other side, not the user-sim's intent. |
| **What did the agent's STT actually deliver (post-ASR)?** | `bot_logs_agent/llm_context.json` — `user` messages. Equivalently, `[AGENT STT final]` lines in `bridge_log.txt`. | The conversation log is OK for this BUT has been cleaned for human readability — for the LLM's actual STT input, use the agent's context. |
| **How long did each audio turn last? Was there an interruption?** | `conversation_log.txt` — turn timestamps and `[INTERRUPTED]` markers. | NOT the LLM contexts (no timestamps on individual messages). |
| **Why did the conversation end?** | `conversation_log.txt` final block — `Stop reason: [EXIT \| TIMEOUT \| ...]`. | |
| **What was the per-LLM-call TTFB / token usage?** | `bridge_log.txt` — `[AGENT/USER METRICS] ttfb …` / `tokens …` events. | NOT `conversation_log.txt` (latency in the log is per-audio-turn, includes the full tool-call chain). |
| **Did the DB end up in the right state?** | `metrics.json["db_state_match"]` + `final_scenario_db_hash.txt`. **But interpret carefully** — see the "Common pitfall" below. |  |

**Common pitfall — DB-state match on read-only scenarios.** A scenario where `expected_scenario_db == initial_scenario_db` (policy-refusal / "deny the request" cases — common in eva_airline 5.x and 7.x) will report `db_state_match: True` whenever the agent makes no mutating tool calls. That is NOT the same as "the agent did the right thing" — an agent that crashed at greeting, looped in identity verification, or simply hung up would also pass. For these scenarios, the DB check alone is insufficient; verify by counting mutating tool calls in `bot_logs_agent/llm_context.json` AND looking at the agent's final assistant utterance to see whether it actually engaged with and refused the request, or just bailed.

**Common pitfall — "no tool calls" claims.** If you're about to write "the agent never called any tool" or "the agent didn't engage," **first** count `tool_calls` in `bot_logs_agent/llm_context.json` (one-liner: `python -c "import json; print(sum(1 for m in json.load(open('SCEN/bot_logs_agent/llm_context.json')) if m.get('tool_calls')))"`). The conversation log may show only INTERRUPTED audio turns and look like the agent did nothing, but tool calls and tool results live in the context file. Audio-side silence ≠ agent inactivity.

### Phase 1 — Scoring summary

Read `judge_result.json`, `final_agent_response.json`, `final_scenario_db_hash.txt`, `metrics.json`. Build the headline table, **labeling each signal as `(gating)` or `(informational)` based on the `success_signals` whitelist from Phase 0**:

| Signal | Value | Notes |
|---|---|---|
| `is_successful` (composite verdict) | from `metrics.json["is_successful"]` | `True` / `False` / `"N/A"`. The verdict; everything else explains it. |
| Stop reason | from `conversation_log.txt` last block | `[EXIT]` = good, `[TIMEOUT]` = bad, others = investigate |
| Judge score | from `judge_result.json["score"]` if file exists | If `judge_result.json` is missing AND `JUDGE_PASSED`/`NL_ASSERTION` not in whitelist, this is **expected absence** — don't flag it. |
| DB-state match | from `metrics.json["db_state_match"]` | `True/False/N/A`. Label `(gating)` for eva_airline / tau2_airline / tau2_retail; `(informational)` for tau2_telecom. |
| DB-state assertion pass rate | from `metrics.json["db_state_assertion_pass_rate"]` | only when scenario opts in. Label `(gating)` for tau2_telecom. |
| NL-assertion pass rate | from `metrics.json["nl_assertion_pass_rate"]` | only when scenario opts in. Label `(gating)` when in the whitelist. |
| Total turns | from `metrics.json["total_turns"]` | |
| Duration | from `metrics.json["scenario_duration"]` | seconds |
| Agent tool calls emitted | **REQUIRED** — count `tool_calls` across all `assistant` messages in `bot_logs_agent/llm_context.json`. Do NOT skip this even on TIMEOUTed / heavily-interrupted scenarios; the context file always has the ground truth. | **0 is a red flag IF the scenario expected tool use** — check the scenario's `agent_resources.tools` first; QA scenarios legitimately have zero tool calls. Conversely, on read-only / policy-refusal scenarios where the agent might "win" by inaction (see Phase 0's common pitfalls), the count of mutating tool calls (versus read-only ones like `Get*Tool`) is the key diagnostic. |
| Mutating vs read-only tool call breakdown | classify each tool call by whether it mutates `shared_state["db"]` (write tools subclass `WriteScenarioTool` for tau2; `*Place*`/`*Cancel*`/`*Modify*`/`*Return*`/`*Exchange*`/`*Refund*`/`*Rebook*`/`*Issue*` patterns for eva). | Critical for policy-refusal scenarios: if `db_state_match: True` AND zero mutating tool calls, the agent passed by inaction, NOT by correctly refusing the request. |
| Reference actions required | count from `scenario_config/reference_answer.json` (if present) | Skip this row entirely if no reference file exists AND `ACTION_MATCH`/`JUDGE_PASSED` aren't in the whitelist — the scenario doesn't ship a reference by design. |
| Actions matched | reference name match (allow arg-superset on agent side) | Same skip condition as above. |

### Phase 2 — Latency analysis (BOTH sides, always)

This is the most quantitatively dense phase. **Run it whether or not the user asked for latency** — it's a required part of every report.

#### Parse all turn boundaries from `conversation_log.txt`

Each non-interrupted turn line has the form `[start_s - end_s] (dur_s) ROLE: text`. Build a Python helper:

```bash
.venv/bin/python << 'PY'
import re, json
from pathlib import Path

LOG = Path('<scenario_dir>/conversation_log.txt')
text = LOG.read_text()

# Match: [  4.670s -  21.622s] (16.952s) AGENT: ...
turn_re = re.compile(r'^\[\s*([\d.]+)s\s*-\s*([\d.]+)s\]\s*\(([\d.]+)s\)\s*(AGENT|USER):\s*(.*)$', re.MULTILINE)
turns = []
for m in turn_re.finditer(text):
    start, end, dur, role, snippet = m.groups()
    turns.append({"role": role, "start": float(start), "end": float(end),
                  "dur": float(dur), "snippet": snippet[:80]})

# Compute response latencies: gap from previous-side-end to this-side-start
for i, t in enumerate(turns):
    prev = next((p for p in reversed(turns[:i]) if p["role"] != t["role"]), None)
    t["response_latency"] = (t["start"] - prev["end"]) if prev else t["start"]

# Aggregate
def total(role, key): return sum(t[key] for t in turns if t["role"] == role)
agent_playback = total("AGENT", "dur")
user_playback = total("USER", "dur")
agent_latency = total("AGENT", "response_latency")
user_latency = total("USER", "response_latency")
duration = turns[-1]["end"] if turns else 0
other = duration - (agent_playback + user_playback + agent_latency + user_latency)
# `other` captures VAD silence between turns, bridge handshake, interruption windows.
print(json.dumps({
    "duration_s": duration,
    "agent_playback_s": agent_playback,
    "user_playback_s": user_playback,
    "agent_response_latency_s": agent_latency,
    "user_response_latency_s": user_latency,
    "other_s": other,
    "n_turns": len(turns),
}, indent=2))
PY
```

#### Pull per-component TTFB metrics from `bridge_log.txt`

**Try the side-tagged lines first** (newer logs, written by the bridge's `_log_rtvi_event` helper). Each TTFB event is emitted as a one-line DEBUG entry with an explicit `[AGENT METRICS]` or `[USER METRICS]` prefix:

```
2026-06-03 12:00:18.008 | DEBUG | bridge:_log_rtvi_event:1577 - [AGENT METRICS] ttfb processor=NvidiaLLMService#0 value=3.132s
2026-06-03 12:00:19.244 | DEBUG | bridge:_log_rtvi_event:1577 - [USER METRICS]  ttfb processor=NemotronTTSService#0 value=0.276s
2026-06-03 12:00:20.195 | DEBUG | bridge:_log_rtvi_event:1577 - [AGENT METRICS] tokens prompt=5207 completion=110
```

Same shape for token-usage events (`tokens prompt=N completion=M` instead of `ttfb processor=X value=Ys`).

```bash
# Side-tagged extraction (preferred — no time-correlation needed):
.venv/bin/python << 'PY'
import re
from pathlib import Path
from statistics import mean, median

SCEN = Path('<scenario_dir>')
ttfb_re = re.compile(r'\[(AGENT|USER) METRICS\] ttfb processor=(\S+) value=([\d.]+)s')
tok_re = re.compile(r'\[(AGENT|USER) METRICS\] tokens prompt=(\d+) completion=(\d+)')
ttfb_events = []
token_events = []
for line in (SCEN / 'bridge_log.txt').read_text().splitlines():
    m = ttfb_re.search(line)
    if m:
        ttfb_events.append({"side": m.group(1).lower(), "processor": m.group(2), "value": float(m.group(3))})
    m = tok_re.search(line)
    if m:
        token_events.append({"side": m.group(1).lower(),
                             "prompt": int(m.group(2)), "completion": int(m.group(3))})

ttfb_buckets = {}
for ev in ttfb_events:
    ttfb_buckets.setdefault((ev["side"], ev["processor"]), []).append(ev["value"])

def stats(vs):
    vs = sorted(vs)
    def pct(p):  # nearest-rank percentile; works for tiny n without numpy
        return round(vs[min(len(vs)-1, int(len(vs)*p))], 2) if vs else 0.0
    return {"n": len(vs), "sum": round(sum(vs),2), "mean": round(mean(vs),2),
            "p50": round(median(vs),2), "p90": pct(0.90), "p95": pct(0.95),
            "max": round(max(vs),2)}

print(f"Total side-tagged TTFB events: {len(ttfb_events)}")
for (side, proc), vs in sorted(ttfb_buckets.items()):
    print(f'  {side:>6} | {proc:<26} | {stats(vs)}')

# Token usage per side (prompt + completion separately; ints not floats)
def int_stats(vs):
    vs = sorted(vs)
    def pct(p):
        return vs[min(len(vs)-1, int(len(vs)*p))] if vs else 0
    return {"n": len(vs), "sum": sum(vs), "mean": sum(vs)//len(vs) if vs else 0,
            "p50": vs[len(vs)//2] if vs else 0, "p90": pct(0.90), "p95": pct(0.95),
            "max": max(vs) if vs else 0}

token_buckets = {}
for ev in token_events:
    token_buckets.setdefault(ev["side"], []).append(ev)
print(f"\nTotal token events: {len(token_events)}")
for side in sorted(token_buckets):
    prompts = [e["prompt"] for e in token_buckets[side]]
    completions = [e["completion"] for e in token_buckets[side]]
    print(f'  {side:>6} | n={len(token_buckets[side])}  prompt={int_stats(prompts)}')
    print(f'         |          completion={int_stats(completions)}')
    print(f'         |          total tokens = {sum(prompts) + sum(completions):,}')
PY
```

If the grep returns **zero** events (older log without the side-tagged lines — predates the bridge `_log_rtvi_event` instrumentation), fall back to the time-correlation method below.

**Fallback: time-correlation against turn-start times.** Older logs only carry TTFBs inside `ProtobufFrameSerializer:deserialize` debug dumps without a side tag. Each looks like:

```
2026-06-03 11:44:18.008 | DEBUG | pipecat.serializers.protobuf:deserialize:143 - 
  ProtobufFrameSerializer: Transport message InputTransportMessageFrame#5(message:
  {'label': 'rtvi-ai', 'type': 'metrics', 'data':
    {'ttfb': [{'processor': 'NvidiaLLMService#0', 'value': 3.131599...}]}})
```

Grep pattern: `'ttfb':` (substring). Parse the timestamp, compute the offset from `Start Time: <ISO>` in `conversation_log.txt`, and attribute each event to the **next-following non-interrupted turn** in the conversation:

```bash
.venv/bin/python << 'PY'
import re
from datetime import datetime
from pathlib import Path
from statistics import mean, median

SCEN = Path('<scenario_dir>')
conv = (SCEN / 'conversation_log.txt').read_text()
scenario_start = datetime.fromisoformat(re.search(r'Start Time: (\S+)', conv).group(1))

turn_re = re.compile(r'^\[\s*([\d.]+)s\s*-\s*([\d.]+)s\]\s*\(([\d.]+)s\)\s*(AGENT|USER):\s*(.*)$', re.MULTILINE)
turns = []
for m in turn_re.finditer(conv):
    start, end, dur, role, snippet = m.groups()
    if "[INTERRUPTED]" in snippet:
        continue  # interrupted turns have no LLM/TTS pipeline to attribute to
    turns.append({"role": role.lower(), "start": float(start), "end": float(end)})
turns.sort(key=lambda t: t["start"])

ttfb_re = re.compile(
    r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+).*'ttfb':\s*\[\{'processor':\s*'([^']+)'.*?'value':\s*([\d.]+)\}"
)
events = []
for line in (SCEN / 'bridge_log.txt').read_text().splitlines():
    m = ttfb_re.match(line)
    if not m: continue
    ts = datetime.fromisoformat(m.group(1).replace(' ', 'T'))
    offset = (ts - scenario_start).total_seconds()
    events.append({"offset_s": offset, "processor": m.group(2), "value": float(m.group(3))})

# Attribute each TTFB to the next-following non-interrupted turn (no time window —
# agent tool-use cycles can stretch the gap to >60s).
for ev in events:
    nxt = next((t for t in turns if t["start"] >= ev["offset_s"]), turns[-1] if turns else None)
    ev["side"] = nxt["role"] if nxt else "unknown"

# ... (same bucketing + stats as above)
PY
```

**Caveats of the fallback heuristic** (don't apply when the side-tagged lines are present): Agent tool-use loops fire multiple LLM TTFBs that all map to the same audio turn — attribution to that turn's side is correct. The imprecise case is when an interrupted turn is filtered out — the heuristic may attribute its TTFB to a downstream turn. Note this in the report when only the fallback path was used.

Also extract bot-server logs if available:
- `evaluation/nemo_experiments/bot_agent_server.log` and `bot_user_server.log` carry per-side TTFB stats directly (`NvidiaLLMService#0 TTFB: <s>` format — that's the bot-side log pattern, distinct from the bridge's). Use only as a cross-check.

#### LLM calls per turn — bridging TTFB and user-perceived latency

**A single per-call TTFB does NOT equal "time until the user hears something."** TTFB on `NvidiaLLMService#0` measures the time from "we sent the prompt to the LLM endpoint" to "the LLM emitted its first response chunk" — but in a tool-calling cycle, that first chunk could be the opening of a `tool_call` delta, not user-facing text. For a turn where the agent does N tool calls before its final response, the user-perceived latency is approximately:

```
user_perceived_turn_latency ≈ Σ(TTFB_i + generation_i + tool_exec_i) for tool-call rounds
                              + TTFB_final + (TTS pre-buffer time)
```

The TTFB-per-LLM-call stats above are ONE of those `TTFB_i` terms. The **per-turn response latency** (computed from `conversation_log.txt` start/end timestamps in the conversation-budget extractor) is the actual user-perceived value and captures the entire tool-call chain. Always cite both, and call out the gap.

**Compute LLM calls per turn explicitly.** For each side, divide the count of `[AGENT/USER METRICS] ttfb processor=NvidiaLLMService#0` events by the count of non-interrupted turns on that side. Values close to 1.0 mean most turns are single-call (no tool chain); values > 1.5 mean a substantial fraction of turns involve tool calling.

```bash
.venv/bin/python << 'PY'
import re, glob
from pathlib import Path

RUN_OR_SCEN = Path('<run_or_scenario_dir>')
# Glob handles both single scenario (one bridge_log) and full run (many).
log_paths = list(RUN_OR_SCEN.glob('bridge_log.txt')) or list(RUN_OR_SCEN.glob('*/bridge_log.txt'))
conv_paths = list(RUN_OR_SCEN.glob('conversation_log.txt')) or list(RUN_OR_SCEN.glob('*/conversation_log.txt'))

llm_ttfb_re = re.compile(r'\[(AGENT|USER) METRICS\] ttfb processor=NvidiaLLMService')
llm_counts = {'agent': 0, 'user': 0}
for f in log_paths:
    for line in open(f):
        m = llm_ttfb_re.search(line)
        if m:
            llm_counts[m.group(1).lower()] += 1

turn_re = re.compile(r'^\[\s*[\d.]+s\s*-\s*[\d.]+s\]\s*\([\d.]+s\)\s*(AGENT|USER):\s*(.*)$', re.MULTILINE)
turn_counts = {'agent': 0, 'user': 0}
for f in conv_paths:
    for m in turn_re.finditer(open(f).read()):
        role, snip = m.groups()
        if '[INTERRUPTED]' in snip: continue
        turn_counts[role.lower()] += 1

for side in ('agent', 'user'):
    n_calls, n_turns = llm_counts[side], turn_counts[side]
    ratio = n_calls / n_turns if n_turns else 0
    print(f'  {side:>6}  LLM calls: {n_calls:>5}  turns: {n_turns:>5}  ratio: {ratio:.2f} calls/turn')
PY
```

**Interpretation guide** (numbers are heuristics, calibrate against your domain):

| `calls / turn` | What it means | Implication |
|---|---|---|
| ≈ 1.0 | Most turns single-call (greeting, simple Q&A, no tools). | Per-call TTFB ≈ user-perceived latency. Service health and UX track each other. |
| 1.1 – 1.5 | Some turns have 1 tool call (lookup, then respond). | Per-call TTFB underreports user-perceived latency on the tool-using subset. Per-turn distribution is the right operator metric. |
| > 1.5 | Many turns make 2+ tool calls before responding. | Per-call TTFB significantly underreports. Slow turns are likely tool-chain-driven, not single-LLM-call-driven. |

The **per-turn vs per-call gap** is the canonical signal for tool-call overhead. Report `(per-turn mean) - (per-call mean)` as "tool-chain overhead per turn." If this gap is small (~1 s), tool calls aren't the bottleneck; if it's large (>5 s), tool-chain length is the dominant slowness factor.

#### Latency report block

Output structure:

```
### Latency breakdown

**Conversation budget (total duration = NNNs):**
| Component                            | Time   | % of total |
|--------------------------------------|--------|-----------|
| Agent TTS playback                   | NNNs   | NN%       |
| User-sim TTS playback                | NNNs   | NN%       |
| Agent response latency (LLM + STT)   | NNNs   | NN%       |
| User-sim response latency            | NNNs   | NN%       |
| Other (VAD silence, interrupts, ...) | NNNs   | NN%       |

**Per-turn response latency** (from `conversation_log.txt` — captures the full tool-call chain; this is the USER-PERCEIVED metric):
| side  | n turns | sum(s) | mean(s) | p50(s) | p90(s) | p95(s) | max(s) |
|-------|---------|--------|---------|--------|--------|--------|--------|
| agent | …       | …      | …       | …      | …      | …      | …      |
| user  | …       | …      | …       | …      | …      | …      | …      |

**Per-LLM-call TTFB stats per side** (one event per LLM endpoint call — does NOT capture tool-call chains within a turn):
| side | processor | n | sum(s) | mean(s) | p50(s) | p90(s) | p95(s) | max(s) |
|---|---|---|---|---|---|---|---|---|
| agent | NvidiaLLMService#0 | … | … | … | … | … | … | … |
| agent | NemotronTTSService#0 | … | … | … | … | … | … | … |
| user | NvidiaLLMService#0 | … | … | … | … | … | … | … |
| user | NemotronTTSService#0 | … | … | … | … | … | … | … |

**LLM-call-per-turn ratio** (gap between user-perceived turn latency and per-call TTFB = tool-chain overhead):
| side  | LLM calls | non-interrupted turns | calls/turn | per-turn mean | per-call mean | gap (tool-chain overhead) |
|-------|-----------|-----------------------|------------|---------------|---------------|---------------------------|
| agent | …         | …                     | …          | …             | …             | …                         |
| user  | …         | …                     | …          | …             | …             | …                         |

**Token usage per side** (from `[AGENT METRICS] tokens prompt=N completion=N` events; one event per LLM call):
| side  | n calls | prompt sum | prompt mean | prompt p90 | prompt max | completion sum | completion mean | completion p90 | completion max | total tokens |
|-------|---------|------------|-------------|------------|------------|----------------|-----------------|----------------|----------------|--------------|
| agent | …       | …          | …           | …          | …          | …              | …               | …              | …              | …            |
| user  | …       | …          | …           | …          | …          | …              | …               | …              | …              | …            |

**Largest single LLM call**: side at t=NNNs with prompt=N completion=N total=N — cross-reference with the slowest-turns block; verbose completions typically correlate with the slowest turns.

**Slowest 3 agent turns:** list with timestamp, per-turn latency, and snippet. **Also note the number of LLM calls that occurred during each slow turn** (count `[AGENT METRICS] ttfb` events whose offset falls between the previous user-turn-end and this agent-turn-start) — a 60 s turn with 5 LLM calls is a tool-chain overhead problem; the same 60 s with 1 LLM call is an LLM-service slowness problem.
**Slowest 3 user turns:**  same shape.
```

**Interpreting slow turns**:
- **High calls-per-turn + slow turns → tool-chain overhead.** Fix in the agent prompt ("chain fewer tool calls" or "issue parallel tool calls when possible"), or by adding tools that batch operations.
- **Low calls-per-turn + slow turns → LLM service slowness.** The per-call TTFB max is informative here; if a single call took >30 s, the service had a slow window.
- **High prompt tokens + slow turns → context bloat.** TTFB on the Nemotron 120B model scales sharply with prompt size; large prompts produce large TTFBs even on a healthy service.

If any turn exceeded 60 s, call it out and identify the cause using the three categories above. Token-count correlation: pull `prompt_tokens`/`completion_tokens` from the bridge log for the slowest turns.

### Phase 3 — Tool-call execution trace

Walk `bot_logs_agent/llm_context.json` chronologically. For each `assistant` message with `tool_calls`, record `(idx, tool_name, arguments)`. For each immediately-following `tool` message, record the result (truncate to first 200 chars).

Compare against `reference_answer.json["actions"]`:
- **Matched** — same `name` and matching key arguments (use `compare_args` from reference when present, otherwise compare all args)
- **Missing** — reference has it, agent didn't call it
- **Extra** — agent called it, reference doesn't
- **Wrong-args** — name matches but arguments differ in a way that mutates state differently

For each non-matched action, cite the assistant message index and explain the deviation. If the agent emitted **zero** tool calls across the whole conversation, flag it as a top-level finding and look at the agent's text for fabricated tool results ("I've verified your identity", "I've located your user ID", etc.).

### Phase 4 — ASR transcript diff (cross-context)

This is the unique value of having both contexts. The user-sim's `bot_logs_user/llm_context.json` contains `assistant` messages = what the user-sim's LLM **wrote** to be spoken. The agent's `bot_logs_agent/llm_context.json` contains `user` messages = what the agent **heard** after STT. They should pair turn-by-turn.

**Differences are not necessarily errors.** STT routinely normalizes spoken forms back to canonical written form — `"seven, eight, seven, one, two"` → `"78712"`, `"f, a, t, i, m, a, underscore, j, o, h, n, s, o, n"` → `"fatima_johnson"`, `"hash w, four, two, eight"` → `"hash w4284"`. These reassemblies are **internal STT post-processing**, not bugs — the canonical form is exactly what the agent uses for tool calls, and downstream correctness is preserved. When walking the diffs, distinguish in prose:

- **Normalization** (benign): spelled→canonical reassembly, digit/letter formatting changes, currency/number formatting (`$8,276.23` vs `eight thousand two hundred seventy six dollars and twenty three cents`). Note these for completeness but don't treat as failures.
- **Real mistranscription** (actionable): phonetic confusion, word elision, concatenation of repeated values, severely garbled letter sequences, missing or fragmented utterances. These can affect agent behavior; lead with these in the report.

**Alignment note:** the agent's `user` messages list often has an extra entry at index 0 from a bridge-injected `"Hello"` kickoff — the user-sim's LLM never generated that. Drop the first agent entry if its content is exactly `"Hello"` before zipping with intents. (The bridge-injected greeting is visible only on the agent side; the user-sim's context is unaffected, so this is purely an alignment step, not a finding to report.)

```bash
.venv/bin/python << 'PY'
import json
from difflib import SequenceMatcher
import re

user_ctx = json.load(open('<scenario_dir>/bot_logs_user/llm_context.json'))
agent_ctx = json.load(open('<scenario_dir>/bot_logs_agent/llm_context.json'))

# What user-sim INTENDED to say (assistant messages on user side)
intents = [m['content'] for m in user_ctx if m.get('role') == 'assistant' and m.get('content')]
# What agent RECEIVED (user messages on agent side, excluding interruptions)
receiveds = [m['content'] for m in agent_ctx if m.get('role') == 'user' and m.get('content') and m['content'] != '[INTERRUPTED]']
# Drop the bridge-injected kickoff greeting if present so turns align 1:1
if receiveds and receiveds[0].strip().lower() == 'hello':
    receiveds = receiveds[1:]

n = min(len(intents), len(receiveds))
def norm(s): return re.sub(r'\s+', ' ', s.lower().strip())
for i in range(n):
    intent = intents[i]
    received = receiveds[i]
    sim_ratio = SequenceMatcher(None, norm(intent), norm(received)).ratio()
    if sim_ratio < 0.92:  # threshold tunable
        print(f'--- Turn {i+1} (similarity {sim_ratio:.2f}) ---')
        print(f'  intent:   {intent[:200]}')
        print(f'  received: {received[:200]}')
PY
```

**Faster alternative for logs with `_log_rtvi_event` side-tags**: final STT transcripts are emitted as `[AGENT STT final] '<text>'` and `[USER STT final] '<text>'` lines in `bridge_log.txt`, so you can ASR-diff without opening the two large `llm_context.json` files. The agent's `[AGENT STT final]` lines are what the agent's STT delivered (i.e., what the agent "heard"); pair them with the user-sim's `assistant` messages from `bot_logs_user/llm_context.json` (which is still the authoritative source for "what the user-sim intended to say" — there's no equivalent log line for the intended text itself). When both shapes are available, prefer the cross-context diff above — it's deterministic and side-tagged at source. Use the bridge-log STT lines for spot-checking or as a fallback when one of the `bot_logs_*` files is corrupted/missing. **Partial transcripts** (`[AGENT STT partial]`) reveal STT instability and mid-utterance corrections — useful when investigating concatenation or boundary-loss errors.

**Patterns to bucket each diff into:**

**Normalization patterns (benign — note for completeness, do NOT count as errors):**

| Pattern | Example | Why benign |
|---|---|---|
| Digit-word reassembly | `seven, eight, seven, one, two` → `78712` | Same value; agent uses canonical form for tool calls |
| Letter spell-out reassembly | `f, a, t, i, m, a` → `fatima` | Same string; STT recombines the spelled letters |
| Mixed alphanumeric reassembly | `hash w, four, two, eight` → `hash w428` | Same ID; canonical form is what the agent needs |
| Punctuation-word folding | `underscore` → `_`, `dash` → `-`, `at` → `@`, `dot` → `.` | STT renders punctuation-words as the actual character |
| Number/currency formatting | `eight thousand two hundred seventy six dollars` → `$8,276` | Same numeric value, different rendering |

**Real mistranscription patterns (actionable — lead with these):**

| Pattern | Example | Likely cause |
|---|---|---|
| Dropped letter | `Johnson` → `Ohnso` | Onset/coda elision; mild |
| Concatenation of repeated values | `4164 4164` → `41644160`, `5 to 7` → `457` | STT runs adjacent digits together; word elision |
| Spelled-letter confusion | `T, C, H` → `TIR` | Letter sequence misrecognized as a word |
| Digit-word swap (homophone) | `four` → `for`, `two` → `to` | Cross-class homophone failure |
| Boundary loss in numbers | `eight seven one two` → `eighty seven twelve` | Number-word aggregation across digit positions (rare; different from benign reassembly) |
| Length mismatch | Whole sentence missing | VAD fragmentation (look at `conversation_log.seglst.json` for fragment boundaries) |

If `len(intents) != len(receiveds)` after the kickoff alignment, that's a **turn-count mismatch** — note it and explain (interrupted turns, dropped utterances, role echo where agent audio was picked up by user-side STT).

### Phase 5 — Conversation flow audit

Walk the conversation chronologically. Flag:

- **Confirmation flow violations** — agent asks "yes/no" → user pivots without answering. Detect by looking for `?` at the end of agent turns followed by a user turn that doesn't include `yes`/`no`/`confirm`/`agree`/`proceed` early in the response. When this happens around a write action (cancel / refund / book / exchange / etc.), the agent may end up not firing the corresponding tool, leaving the DB unmutated where the reference expected a mutation.
- **Agent fabrications** — `assistant` text claims a tool was called when no matching `tool_calls` field exists. Common phrases: *"I've verified your identity"*, *"I've located your user ID"*, *"I've successfully [...]"*, *"I've cancelled [...]"*, *"Refund of $..."* — any of these without a corresponding tool call in the recent message history is a fabrication.
- **Looped responses** — three consecutive agent turns with >0.8 textual similarity. Stalemate symptom.
- **Role confusion** — `user` message that reads like an agent (e.g., starts with *"I'd be happy to help"*, *"I cannot look up..."*). Usually acoustic echo from the bridge picking up agent audio on the user-side STT.
- **Long agent responses (>1500 completion tokens)** — find via bridge log token-usage events; these often correlate with the slowest turns and verbose fabricated summaries.

### Phase 6 — Cross-reference with scenario class

Import the scenario class (see "Cross-reference against the scenario class" above) and compare:
- Expected DB hash vs actual DB hash. If mismatch, list the actions the reference DB-replay performed that the agent's recorded actions didn't.
- `scenario.nl_assertions` (the actual list) vs `judge_result.json["nl_assertion_verdicts"]` (the judge's findings). For each `passed: false`, look at the relevant point in the conversation and decide whether the failure was agent (didn't say it), ASR (said it but transcribed wrong), or judge (model misread the transcript).
- `scenario.user_persona.background`, `scenario.user_resources.info_sections`, and `scenario.tau2_task["user_scenario"]["instructions"]["unknown_info"]` (tau2 only) — verify the user-sim's prompt actually contained the expected `known_info` / `unknown_info` / anti-fabrication clause. If not, the bot is most likely running stale code that pre-dates a recent prompt-assembly change; recommend a bot restart and re-run before any further analysis.

### Phase 7 — Root-cause attribution

Every issue surfaced in Phases 3-6 gets one of four labels:

| Label | When to use |
|---|---|
| **user-sim** | The simulated user violated conversational flow (skipped confirmation, fabricated info not in known_info, dropped to silence prematurely). |
| **agent** | The agent didn't call a needed tool, called a wrong tool, fabricated tool results in text, or summarized actions it didn't take. |
| **ASR** | STT mistranscribed a user utterance (verified via Phase 4). When the agent then misacts on the mistranscription, primary fault is ASR, secondary is agent for not asking clarification on low-confidence input. |
| **framework** | Tool registration failed, deadlock in pipecat, prompt failed to propagate, hash function diverges across sides, etc. Surfaces as systematic across scenarios. **NEVER apply this label for missing `reference_answer.json` / `judge_result.json` / `is_action_match: "N/A"` without first checking the scenario's `success_signals` whitelist** — those files are only required when the corresponding signal is gating. See [Phase 0](#phase-0--read-the-scenarios-scoring-contract-first). False-positive framework claims send the operator on goose chases. |

For each finding, give one label as the **primary** root cause and (when relevant) a **secondary** label.

## Report output

Write the report to:
- **Single scenario mode**: `<scenario_dir>/analysis_report.md`
- **Full run mode**: per-scenario reports as above, PLUS a run-level summary at `<run_dir>/analysis_report.md`.

Run-level report uses the **structure below** (in this exact order):

```markdown
# Eval Run Analysis: <run_id>

**Result dir:** `<absolute path>`
**Run start:** <earliest `Start Time:` from any scenario's `conversation_log.txt` — do NOT parse from the result directory name (it's user-controlled and may not encode time)>
**Generated:** <current date — pass this in via context, do not call Date.now()>
**Scenarios in this run:** N

## Aggregate signals

| Signal | Value | Notes |
|---|---|---|
| Total scenarios | N | |
| Judge success rate | X% (P/N) | Prefer the value from `all_summary.txt`'s `Overall Success Rate` line; sanity-check against per-scenario `metrics.json["is_successful"]`. |
| Mean judge score | 0.NN | Mean of `judge_result.json["score"]` across all scenarios with a judge result. |
| Score distribution | 1.0: K · ≥0.8: K · ≥0.5: K · <0.5: K | One-line bucket counts. Useful for catching the difference between "all 1.0" and "mix of 1.0 and 0.4". |
| DB-state match rate | X% (P/N scenarios with `expected_scenario_db`) | From `all_summary.txt`'s `DB-State Match Rate`; verify via per-scenario `metrics.json["db_state_match"]`. Denominator is scenarios that opted into DB-state scoring, not all scenarios. |
| NL-assertion pass rate | X% (P/N assertions across scenarios) | From `all_summary.txt`'s `NL-Assertion Pass Rate`. Denominator is **assertions**, not scenarios. Absent when no scenario in the run carries `nl_assertions`. |
| Stop reason distribution | `[EXIT]`: K · `[TIMEOUT]`: K · other: K | Count `Stop reason:` lines from each `conversation_log.txt`. `[EXIT]` is the only healthy value; `[TIMEOUT]` and others always warrant investigation in their per-scenario report. |
| Run wall-clock duration | X s (Y min) | End-to-end clock time from first scenario start to last scenario finish. Parse the **earliest `Start Time:`** and the **latest `End time:`** across all per-scenario `conversation_log.txt`s, and compute the difference. **Do NOT parse the timestamp from the run directory name** — the directory name is user-controlled (e.g. via `run_evaluation.sh`) and not guaranteed to follow any timestamp pattern. **Includes any between-scenario warmup/transition overhead** (this is the wall-clock cost of the eval, the thing that determines "how long do I wait for results"). |
| Sum of scenario durations | Σ scenario_duration s (Y min) | Sum of `metrics.json["scenario_duration"]` across scenarios. Excludes between-scenario overhead. Compare against run wall-clock to see how much time is spent outside scenarios. |

**Headline:** <one or two sentences synthesizing the above — "X of N scenarios passed cleanly, with the failures concentrated around <pattern>" or "All N at 1.0; the run successfully validates <change-under-test>".>

## Per-scenario status

| Scenario | Judge | DB match | NL-assertion | Stop reason | Duration | Notes / report link |
|---|---|---|---|---|---|---|
| [`scenario_a`](./scenario_a/analysis_report.md) | 1.0 | ✓ | 100% (1/1) | `[EXIT]` | N.N min | brief one-liner |
| ... | | | | | | |

## Cross-scenario patterns

<Recurring real mistranscriptions, recurring agent fabrications, framework issues observed across multiple scenarios. Recurring STT normalization is benign and need not be called out at run level.>

## Run-level latency rollup

**TTFB distribution per side and processor** (concatenated across all scenarios; same `nearest-rank` percentile helper as Phase 2):

| side  | processor                | n   | sum(s) | mean(s) | p50(s) | p90(s) | p95(s) | max(s) |
|-------|--------------------------|-----|--------|---------|--------|--------|--------|--------|
| agent | NvidiaLLMService#0       | …   | …      | …       | …      | …      | …      | …      |
| agent | NemotronTTSService#0     | …   | …      | …       | …      | …      | …      | …      |
| user  | NvidiaLLMService#0       | …   | …      | …       | …      | …      | …      | …      |
| user  | NemotronTTSService#0     | …   | …      | …       | …      | …      | …      | …      |

**Per-turn response-latency distribution** (from `conversation_log.txt` of every scenario — this is the USER-PERCEIVED metric, includes full tool-call chains):

| side  | n   | sum(s) | mean(s) | p50(s) | p90(s) | p95(s) | max(s) |
|-------|-----|--------|---------|--------|--------|--------|--------|
| agent | …   | …      | …       | …      | …      | …      | …      |
| user  | …   | …      | …       | …      | …      | …      | …      |

**LLM-call-per-turn ratio (run-level)** — cross-scenario aggregate of the per-call vs per-turn gap. A wide gap means tool-chain overhead dominates the user-perceived slowness; a narrow gap means per-call TTFB IS the user-perceived latency.

| side  | total LLM calls | total turns (non-interrupted) | calls/turn | per-turn mean (s) | per-call mean (s) | gap (tool-chain overhead, s) |
|-------|-----------------|-------------------------------|------------|-------------------|-------------------|-------------------------------|
| agent | …               | …                             | …          | …                 | …                 | …                             |
| user  | …               | …                             | …          | …                 | …                 | …                             |

Interpretation: a `calls/turn > 1.5` with a large gap (>5 s) means the agent is making many tool calls per turn and that's the dominant slowness — fix in the agent prompt (chain fewer tools, batch where possible). A `calls/turn ≈ 1.0` with a large gap means TTS pre-buffering or post-LLM generation time is the slowness — different fix.

**Conversation budget rollup (sum across all scenarios):**

| Component                 | Sum (s) | % of total cross-scenario duration |
|---------------------------|---------|------------------------------------|
| Agent TTS playback        | …       | …                                  |
| Agent response latency    | …       | …                                  |
| User-sim TTS playback     | …       | …                                  |
| User-sim response latency | …       | …                                  |
| Other                     | …       | …                                  |

**Token usage rollup (concatenated across all scenarios):**

| side  | n calls | prompt sum | prompt mean | prompt p90 | prompt max | completion sum | completion mean | completion p90 | completion max | total tokens |
|-------|---------|------------|-------------|------------|------------|----------------|-----------------|----------------|----------------|--------------|
| agent | …       | …          | …           | …          | …          | …              | …               | …              | …              | …            |
| user  | …       | …          | …           | …          | …          | …              | …               | …              | …              | …            |

**Run-level token total:** N agent + N user-sim = **N total tokens** consumed across N scenarios. Useful for cost / quota tracking when running batches against metered endpoints.

## Next steps

- <Pending failure investigations, recommended re-runs, framework fixes worth prioritizing>
```

**How to populate the aggregate signals**:
- Always **try `all_summary.txt` first** for `Overall Success Rate`, `DB-State Match Rate`, and `NL-Assertion Pass Rate` — the runner already computes them, citing the canonical numbers avoids drift.
- Re-derive when the file is missing/malformed, OR for sanity-checking — sum per-scenario `metrics.json["is_successful"]`, `metrics.json["db_state_match"]`, and `metrics.json["nl_assertion_pass_rate"]` with the right denominators (scenarios with reference; scenarios with expected_scenario_db; total nl_assertions, NOT scenarios).
- **Stop-reason distribution** isn't in `all_summary.txt` — grep `Stop reason:` from every `conversation_log.txt` and count buckets.
- **Score distribution** isn't either — bucket the `judge_result.json["score"]` values by threshold yourself.

**Detailed latency rollup**: feed the concatenated event list across scenarios into the same `nearest-rank` percentile helper used in Phase 2. The p90 is genuinely useful at the run level even though it's noisy per-scenario, because run-level aggregation has 4-10× more samples.

## Per-scenario report structure (use this exact section ordering)

This template is for single-scenario `analysis_report.md`s. The run-level structure is defined above under "Report output".

```markdown
# Eval Result Analysis: <scenario_name>

**Result dir:** `<absolute path>`
**Run start:** <earliest `Start Time:` from any scenario's `conversation_log.txt` — do NOT parse from the result directory name (it's user-controlled and may not encode time)>
**Generated:** <current date — pass this in via context, do not call Date.now()>

## 1. Headline

<table from Phase 1>

**Bottom line:** <one-sentence verdict on whether the scenario succeeded, why, and the dominant failure mode if it didn't>

## 2. Latency breakdown

<budget table + per-side TTFB stats + slowest turns, from Phase 2>

## 3. Tool-call trace

<chronological list of agent's tool calls with match/missing/extra/wrong-args annotations, from Phase 3>

## 4. ASR transcript diff

<turn-by-turn intent-vs-received diffs where similarity < threshold, from Phase 4. **Lead with real mistranscriptions** (the actionable signal) and group normalization cases under a separate "Normalization (benign, no information loss)" paragraph or sub-bullet. Headline counts should distinguish the two: e.g. "5 of 9 turns differ — 3 are STT normalization (benign), 2 are real mistranscriptions". If no diffs found, say so explicitly — silence is a finding.>

## 5. Conversation flow issues

<one entry per flow violation, fabrication, loop, role-confusion, etc., from Phase 5. Cite message indices and quote relevant snippets.>

## 6. Gold-replay cross-check

<DB hash match status + per-action diff if mismatch + nl_assertion verdict review, from Phase 6>

## 7. Root-cause attribution

| # | Issue | Primary | Secondary | Notes |
|---|-------|---------|-----------|-------|
| 1 | <one-line issue description> | user-sim/agent/ASR/framework | (optional) | (optional) |
...

## 8. Recommended actions

- <bulleted, ranked by impact: prompt tweak, framework fix, model investigation, rerun-after-restart, etc.>
```

## Tips

- **Never claim the agent "didn't do X" without checking `bot_logs_agent/llm_context.json` first.** The conversation_log shows audio turns only — tool calls, agent reasoning, and tool results all happen in the agent's processing time between audio turns and are completely absent from the conversation log. A scenario whose conversation_log shows 4 INTERRUPTED audio turns may still have a fully successful tool-call chain underneath. The same applies symmetrically to the user-sim — `bot_logs_user/llm_context.json` is the source of truth for what the user-sim's LLM generated, regardless of how much of it made it to TTS / STT / audio. See the [Authoritative source per question type](#authoritative-source-per-question-type--do-not-skip-this-table) table.
- **Read `scenario_config/metadata.json` for `success_signals` before judging "missing" anything.** This is the single most common source of false-positive framework findings. The five signals (`is_action_match`, `db_state_match`, `db_state_assertion`, `nl_assertion`, `judge_passed`) split into two classes per scenario — gating (in the whitelist) and informational (not in the whitelist). Missing on-disk artifacts for an *informational* signal are not findings. Examples: a tau2_telecom run will have `is_action_match: False` on most scenarios and that's fine (open-spec; `DB_STATE_ASSERTION` is the gate). An eva_airline run will have no `reference_answer.json` on scaffolded scenarios and that's fine (only `DB_STATE_MATCH` is gating; the reference is informational). See [Phase 0](#phase-0--read-the-scenarios-scoring-contract-first).
- **Prefer Bash + Python one-liners** over reading whole files when extracting numbers. `bridge_log.txt` can be 1 MB+; grep first, then parse.
- **Never read `conversation_log.wav`.** Just note it's available for human listening.
- **Don't trust `scenario_config/reference_answer.json` alone** — re-derive from the scenario class when DB hashes disagree, since the reference file is written at runtime from in-process state and may reflect older code if generated by a stale process.
- **When the agent emits zero tool calls**, this is almost always the root cause of every other failure in the scenario. Lead with it.
- **Token counts correlate with TTFB on the agent side.** A 10K-prompt-token turn with 2K completion tokens regularly produces 30-60 s TTFB on the Nemotron 120B model. Note this when calling out slow turns; don't blame the policy prompt size unless you have data.
- **Agent prompt size grows linearly across a conversation** as tool-call history accumulates. A retail run typically starts at ~5K prompt tokens and ends at ~10–11K by the last turn. If the agent's max prompt is >15K, investigate context bloat (verbose tool results, unconsolidated history). User-sim prompts stay flat (~1–2K) because the user-sim doesn't carry tool-call history. The Phase 2 token-usage table per side surfaces this; flag it in the report when prompt-mean is much higher than expected for the conversation length.
- **Token totals: prefer `metrics.json["token_usage"]` + `all_summary.txt` "Token Usage:" line when present.** The runner now snapshots `bridge.token_usage` into each scenario's `metrics.json` (under key `"token_usage"` with shape `{"agent": {n_calls, prompt, completion}, "user": {...}}`) and prints a run-level rollup in `all_summary.txt`. Use those canonical numbers for the headline rollup. **Per-LLM-call distributions** (mean / p90 / max prompt-and-completion size, largest single call) still require parsing `[AGENT/USER METRICS] tokens prompt=N completion=N` events from `bridge_log.txt` — the runner only sums; it doesn't track per-call distributions. So: canonical sums come from `metrics.json` / `all_summary.txt`; distributions still come from the log. If totals disagree between the two for the same scenario, the log is source-of-truth (the runner just sums those same lines).
- **Overwrite existing `analysis_report.md` without prompting.** When the skill is re-invoked on a directory that already has a report (e.g. after applying a fix and wanting fresh analysis), write the new content straight over the old file — do NOT skip the run, do NOT append, do NOT merge. Reports are point-in-time snapshots; the latest invocation always reflects current understanding. If the caller wants to preserve a historical version, they're responsible for renaming/archiving it before re-running.
- **Stop reason `[EXIT]` means the agent called `EndConversationTool` voluntarily.** `[TIMEOUT]` means the scenario duration cap was hit. Always report this — many failure modes hide behind a `[TIMEOUT]` because the conversation looped instead of progressing.
- **When restarting bots solves a bug**, that's a framework-class root cause (stale process), not a fix in itself. Flag it so the user knows what to verify.
- **Detect log "generation" by grepping for `[AGENT METRICS] ttfb`.** Logs from bots that include the `_log_rtvi_event` helper have side-tagged metrics lines; older logs don't. A single `grep -c '[AGENT METRICS] ttfb'` of `bridge_log.txt` tells you which extraction path to use. If the count is 0, fall back to the time-correlation path; if it's >0, prefer the side-tagged path (deterministic, no heuristics) and note that the report was generated against the new log format. This forward-compatibility matters when re-analyzing historical eval runs.

## What this skill should NOT do

- Don't propose code changes or apply edits as part of the analysis — the report is the deliverable. The user reads the report and decides what to change next.
- Don't run the eval pipeline. This skill is purely retrospective.
- Don't open the audio file or attempt acoustic analysis. Mention its presence if relevant.
