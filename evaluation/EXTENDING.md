# Extending the NeMo Voice Agent Evaluator

This guide is the author's reference for adding new scenarios, tools, and domains. For an operator's view (install, run, read results), see [`README.md`](README.md).

## Contents

- [Extension points](#extension-points)
- [Scenario Structure](#scenario-structure)
  - [The 8 properties (per side: user and agent)](#the-8-properties-per-side-user-and-agent)
  - [Scenario-level fields](#scenario-level-fields)
  - [Scoring contract (`success_signals`)](#scoring-contract-success_signals)
  - [Domain organization](#domain-organization)
- [Creating a New Scenario](#creating-a-new-scenario)
  - [1. Pick or create a domain](#1-pick-or-create-a-domain)
  - [2. Subclass the domain base](#2-subclass-the-domain-base)
  - [3. Verify](#3-verify)
- [Tool System](#tool-system)
  - [Tool configuration](#tool-configuration)
  - [Shared state](#shared-state)
  - [Termination contract — two patterns](#termination-contract--two-patterns)
  - [Creating a new tool](#creating-a-new-tool)

## Extension points

Extension points, in increasing order of scope. Each links to the detailed reference below.

1. **[New scenario](#creating-a-new-scenario)** — subclass an existing domain base and override the properties that differ. ~30–60 lines for in-repo smoke domains; tau2 and eva subclasses are often <20 lines because everything derives from a single `tau2_id` / `eva_id`.
2. **[New tool](#tool-system)** — subclass `StandardSchemaTool` (read-only), `WriteScenarioTool` (mutates `shared_state["db"]` + auto-records to the bridge-pulled action list), or `SendScenarioSummaryTool` (legacy LLM-callable summary, used only by in-repo smoke domains). Register with `@register_schema_tool_for_eval(domain="<domain>")` and side-import the module in `nemo_voice_agent/evaluation/tools/__init__.py`.
3. **[New domain](#scenario-structure)** — create a `{Domain}BaseScenario` under `nemo_voice_agent/evaluation/scenarios/data/{domain}/` (or as a single `{domain}.py` file for small domains). Set `domain = "{domain}"` as a ClassVar so the per-domain tool registry, fixture subdir, and bridge `tool_domain` argument all key off the same string. Add a `tools/{domain}_tools.py` for domain-specific tools and side-import it. The domain base should set `agent_resources` / `user_resources` defaults that subclasses can override, plus the right termination contract (`EndConversationTool` + either an LLM summary tool **or** the bridge-pull + `_record_action` pattern).
4. **(Advanced) Cross-side state & deterministic predicates** — opt into one or more per-domain registries: `@register_db_state_predicate(domain=...)` (`evaluation/db_state_predicates.py`) for deterministic outcome checks, `@register_initialization_function(domain=...)` (`evaluation/initialization_functions.py`) for fixture-seeding mutations, `@register_sync_applier(domain=...)` (`evaluation/sync_appliers.py`) + `Scenario.sync_state` override for dual-side DB propagation. See `tau2_telecom` for the canonical wiring of all four.

## Scenario Structure

A scenario fully specifies what both the user and the agent do during one evaluation run. Each is a Python class with 8 properties plus some scenario-level fields.

### The 8 properties (per side: user and agent)

| Property | Type | Purpose |
|----------|------|---------|
| `{side}_persona` | `Persona` | `role`, `name`, `background`, `personality`, optional `language`/`accent`. Rendered as the opening lines of the system prompt. |
| `{side}_task` | `Task` | `goal` and `background`. The single objective this side is trying to achieve. |
| `{side}_actions` | `Actions` | Ordered `instructions` (step-by-step script) and persistent `guidelines` (always-apply rules). |
| `{side}_resources` | `Resources` | `tools` dict (tool class name → constructor kwargs), `documents`, free-form `information` strings. |

### Scenario-level fields

| Field | Purpose |
|-------|---------|
| `name` | Unique scenario ID. Convention: `{domain}__{scenario_name}` (e.g., `restaurant__pizza_pepperoni`, `tau2_airline__7`). |
| `description` | Short human-readable summary. |
| `max_duration` | Max scenario duration in seconds. Overrides the CLI default. |
| `domain` | ClassVar string. Keys the per-domain tool registry namespace, the fixture subdir (e.g., `evaluation/data/{domain}/`), and the bridge's `tool_domain` argument to `update_system_prompt`. Set on the domain base class; subclasses inherit. Default is `"default"`. |
| `success_signals` | **Required.** Tuple of `SuccessSignal` members that gate the `is_successful` composite for this scenario. Set as a ClassVar on the domain base when uniform across the domain, or as a `cached_property` deriving from `self.nl_assertions` / other per-scenario fields when the domain has a mixed composition. `Scenario.__init_subclass__` validates this is non-empty at class-definition time for any class declaring `name`. See [Scoring contract](#scoring-contract-success_signals) below. |
| `reference_answer` | The expected action list (or `<final_response>` payload for legacy LLM-summary scenarios). Dict or list-of-dicts. Drives the action-list match signal. |
| `expected_scenario_db` | Optional `cached_property`. End-state DB fixture used by the DB-state hash match signal. Path-independent. |
| `db_state_assertions` | Optional list of deterministic predicate records. Used by the per-predicate `db_state_assertion_pass_rate` signal. See [Evaluation Metrics](README.md#evaluation-metrics). |
| `nl_assertions` | Optional list of natural-language claims about agent behavior, judged by an LLM. Drives the `nl_assertion_pass_rate` signal. |
| `initialization_actions` | Optional list of `{func_name, arguments, side}` records dispatched by the bot's `apply_initialization` handler to seed per-side DB state. |
| `ignore_capitalization` | String matching: case-insensitive. |
| `ignore_punctuation` | String matching: strip punctuation. |
| `clean_text` | String matching: apply ASR text cleaning. |
| `noise_config` | Optional `NoiseConfig` to inject background noise into the user→agent channel. |

Most subclasses only declare a handful of these — the rest are derived from a single `tau2_id` / `eva_id` ClassVar via `cached_property` on the domain base.

### Scoring contract (`success_signals`)

Every concrete scenario must resolve `success_signals` to a non-empty tuple of `SuccessSignal` members (the enum lives at `nemo_voice_agent.evaluation.scenarios.classes.SuccessSignal`):

| Member | Canonical key | When to use |
|---|---|---|
| `ACTION_MATCH` | `is_action_match` | Recursive match of the agent's prediction against `reference_answer`. Appropriate when the domain has a single canonical correct payload — whether that's a structured outcome summary (restaurant order dict, customer-service ticket) OR a tool-call trajectory where only one sequence is considered successful. **Not** appropriate when the solution space is open (multiple valid trajectories satisfying the same outcome — telecom). For domains that ship an `expected_scenario_db`, prefer `DB_STATE_MATCH` over `ACTION_MATCH` — it's path-independent. |
| `DB_STATE_MATCH` | `db_state_match` | Whole-DB SHA-256 equality against `expected_scenario_db`. Path-independent but trajectory-baking — appropriate when there's a single deterministic end state. |
| `DB_STATE_ASSERTION` | `db_state_assertion` | Per-predicate verdicts against the bridge-pulled DB. Use when the solution space is open (multiple valid end states satisfy the same outcome predicates — telecom-style). |
| `NL_ASSERTION` | `nl_assertion` | Per-claim LLM-judged verdicts on the transcript. Use when the domain has natural-language behavioral expectations. Requires `--judge-url`. |
| `JUDGE_PASSED` | `judge_passed` | Overall LLM judge score (binarized via `--judge-threshold`). Use as the **only** signal for domains where no deterministic check applies (free-form QA). |

**Principle:** `JUDGE_PASSED` gates only when no deterministic alternative exists. `ACTION_MATCH` gates only when the domain has a single canonical correct payload (open-spec domains use `DB_STATE_ASSERTION` instead). Signals NOT in `success_signals` are still computed and saved (visible in `metrics.json["success_breakdown"]["excluded"]`) — they just don't drive the verdict.

**Override patterns:**

```python
# Uniform signals across the domain → ClassVar tuple on the base.
class Tau2AirlineBaseScenario(Tau2BaseScenario):
    success_signals = (SuccessSignal.DB_STATE_MATCH,)

# Mixed composition (per-scenario opt-ins) → cached_property on the base.
class Tau2RetailBaseScenario(Tau2BaseScenario):
    @cached_property
    def success_signals(self):
        if self.nl_assertions:
            return (SuccessSignal.DB_STATE_MATCH, SuccessSignal.NL_ASSERTION)
        return (SuccessSignal.DB_STATE_MATCH,)

# Per-scenario outlier → declare directly on the subclass.
@register_eval_scenario
class WeirdEdgeCase(SomeBase):
    name = "..."
    success_signals = (SuccessSignal.DB_STATE_ASSERTION,)  # overrides base
```

**Escape hatch — custom verdict logic.** If a scenario needs more than strict-AND (weighted majority, threshold gating, OR-of-some), override `compute_is_successful(self, signals)` directly. The default implementation lives on `Scenario` and reads `self.success_signals` to filter.

**Authoring trap.** Forgetting `success_signals` on a new domain base raises `TypeError` at class-definition time (`Scenario.__init_subclass__` validation). An empty tuple `()` is also rejected — a scenario with zero gating signals is a bug, not a config.

### Domain organization

Scenarios are organized by domain using a **base class pattern**:

- A domain base class (e.g., `RestaurantBaseScenario`, `CustomerServiceBaseScenario`, `QABaseScenario`) implements all 8 properties with domain-level defaults. It is **not** registered.
- Concrete scenarios inherit the base and override only the properties that differ — typically `user_persona`, `user_task`, `user_actions`, `agent_actions`, `agent_resources`, and `reference_answer`.
- Each domain lives under `nemo_voice_agent/evaluation/scenarios/data/` — either as a single file (e.g., `restaurant.py`, `customer_service.py`, `qa.py`) or as a package (e.g., `eva_airline/`, `tau2_airline/`, `tau2_retail/`, `tau2_telecom/`) when the scenario set is large enough to split across multiple files.

## Creating a New Scenario

### 1. Pick or create a domain

Existing domains: `eva_airline`, `tau2_airline`, `tau2_retail`, `tau2_telecom`, `tau2_telecom_workflow` (primary benchmarks); `restaurant`, `customer_service`, `qa` (in-repo smoke sets); `default` (legacy / cross-domain harness tools, including the legacy `fastbite` and `simple_qa_*` scenarios).

- **Adding a scenario to an existing domain** — drop a new class into the matching file (`scenarios/data/{domain}.py`) or group module (`scenarios/data/{domain}/group_Nx.py` for packaged domains).
- **Creating a new domain** — add `scenarios/data/{domain}/` (package) or `scenarios/data/{domain}.py` (single file) with a `{Domain}BaseScenario` base class. Set `domain = "{domain}"` as a ClassVar. Side-import the new module from `scenarios/data/__init__.py`. Add `tools/{domain}_tools.py` for any domain-specific tools and side-import it from `tools/__init__.py`.

### 2. Subclass the domain base

Override only what's specific to your scenario. Inherited properties come from the base.

```python
from nemo_voice_agent.evaluation.scenarios import register_eval_scenario
from nemo_voice_agent.evaluation.scenarios.classes import Actions, Persona, Resources, Task
from nemo_voice_agent.evaluation.scenarios.data.restaurant import RestaurantBaseScenario

PIZZA_PALACE_MENU = """
## Pizza Palace Menu
Pepperoni Pizza - $9.99
Extra Cheese - $1.50
"""

@register_eval_scenario
class PizzaPepperoni(RestaurantBaseScenario):
    name = "restaurant__pizza_pepperoni"
    description = "Order a pepperoni pizza with extra cheese at Pizza Palace"
    reference_answer = {
        "items": [
            {"name": "Pepperoni Pizza", "unit_price": "9.99", "quantity": "1"},
            {"name": "Extra Cheese", "unit_price": "1.50", "quantity": "1"},
        ],
        "customer_name": "Charlie",
        "customer_phone": "314-527-8960",
        "total_price": "11.49",
    }

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="human user",
            name="Charlie",
            background="You work as a teacher. Your phone number is 314-527-8960.",
            personality="Communicative, friendly, decisive.",
        )

    @property
    def user_task(self) -> Task:
        return Task(goal="Order a pepperoni pizza with extra cheese.")

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Ask for pizza options.",
                "Order one pepperoni pizza.",
                "Ask if extra cheese is available and add it.",
                "Finish the order and ask for the price.",
            ],
        )

    @property
    def agent_resources(self) -> Resources:
        return Resources(
            tools={
                "GetMenuTool": {"menu": PIZZA_PALACE_MENU},
                "PlaceOrderTool": {"auto_validate": "False"},
                "EndConversationTool": {},
            },
        )
```

> **Note on `success_signals`** — this example inherits `(SuccessSignal.ACTION_MATCH,)` from `RestaurantBaseScenario`, so no declaration is needed. For a brand-new domain you'd set it on the base (or as a `cached_property` for mixed compositions); for a per-scenario outlier you'd declare it directly on the subclass. See [Scoring contract](#scoring-contract-success_signals) above.

### 3. Verify

```bash
python run_evaluation.py --list
# → restaurant__pizza_pepperoni should appear

python run_evaluation.py --scenarios restaurant__pizza_pepperoni
```

## Tool System

### Tool configuration

Tools are referenced by class name in `Resources.tools` as `{tool_name: constructor_kwargs}`. The bridge serializes this to JSON and the bot server instantiates each tool by calling `TheTool(**tool_args)`. Different scenarios can pass different kwargs to the same tool class — e.g., `GetMenuTool({"menu": PIZZA_PALACE_MENU})` vs. `GetMenuTool({"menu": BURGER_BARN_MENU})`.

### Shared state

Tools in the same scenario can share mutable state via a `shared_state` dict that is injected into their constructors if they declare it. The bridge creates one fresh dict per scenario and passes the same reference to every tool that accepts it.

Example: `JoinWaitListTool`, `DropWaitListTool`, and `GetWaitlistTool` all read and write `shared_state["waitlist"]`, so when the agent joins a customer via one tool, checking the list via another returns the updated data.

### Termination contract — two patterns

Every scenario needs a way for the bridge to (a) capture what the agent did and (b) end the conversation when work is complete. Two patterns coexist in the repo:

**Pattern A — bridge-pull (primary benchmarks: `eva_airline`, all `tau2_*`).** Write tools subclass `WriteScenarioTool` and call `self._record_action(...)` on success; the record is appended to `shared_state["actions"]`. At end of scenario, the bridge pulls `{actions, db_hash}` from BOTH bots via the `get_scenario_summary` RTVI action. The agent never emits an LLM-callable summary — no double-emit / forgot-to-call class of bugs. Recommended for new benchmarks.

**Pattern B — LLM summary (in-repo smoke sets: `restaurant`, `customer_service`, `qa`, `waitlist`).** A `SendScenarioSummaryTool` subclass wraps the agent's final structured result in `<final_response>` tags; the bridge captures the payload into `final_agent_response.json`. Examples: `PlaceOrderTool`, `ResolveTicketTool`, `SaveQuestionAnswerTool`, `JoinWaitListTool`. Kept for the original in-repo scenarios; not recommended for new work.

**Both patterns require `EndConversationTool`** — sends an `<exit>` tag that triggers the bridge to stop the scenario early. Without it, the bridge waits for the full `max_duration`, which can cause server-side WebSocket keepalive timeouts during idle periods. The domain base class should include `EndConversationTool` in `agent_resources.tools` and instruct the agent to call it.

### Creating a new tool

Subclass the right base for your tool's role, then register with `@register_schema_tool_for_eval(domain="<domain>")`:

- **Read-only tools** — `StandardSchemaTool`.
- **Write tools that should land in the bridge-pulled action list** — `WriteScenarioTool` (call `self._record_action({...})` on success). Pattern A.
- **Legacy LLM-callable summary** — `SendScenarioSummaryTool`. Pattern B.

```python
from nemo_voice_agent.evaluation.tools import register_schema_tool_for_eval
from nemo_voice_agent.utils.tool_calling import StandardSchemaTool

@register_schema_tool_for_eval(domain="restaurant")
class GetMenuTool(StandardSchemaTool):
    def __init__(self, *, menu: str = "", description: Optional[str] = None):
        super().__init__(description=description or "Get the restaurant menu.")
        self.menu = menu

    @property
    def properties(self):
        return {}

    @property
    def required_properties(self):
        return []

    async def _execute(self, params):
        await params.result_callback({"menu": self.menu})
```

Side-import the module from `nemo_voice_agent/evaluation/tools/__init__.py` so the decorator fires at import time. The constructor can accept:
- Any number of data kwargs (e.g., `menu`, `accounts`, `orders`).
- `shared_state: Optional[dict]` — auto-injected if declared. Mutations to `shared_state["db"]` are visible to other tools in the same scenario and to the bridge's end-of-scenario pull.
- `rtvi: Optional[RTVIProcessor]` — auto-injected if declared. Needed for `SendScenarioSummaryTool` subclasses and for emitting `action-applied` cross-side sync events (handled by `WriteScenarioTool._record_action` automatically when the bot has stashed the `__rtvi__` sentinel).

The bare-decorator form (`@register_schema_tool_for_eval` with no args) still works and registers into the `"default"` namespace — use it only for cross-domain harness tools like `EndConversationTool`, never for benchmark-specific tools.
