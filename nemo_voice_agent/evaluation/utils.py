# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import re
from typing import Dict, List, Optional, Union

import numpy as np
import requests
from dotenv import load_dotenv
from loguru import logger
from nemo.collections.asr.parts.utils.eval_utils import clean_label, remove_punctuations


def match_str_and_float(
    ref_value: Union[str, float],
    pred_value: Union[str, float],
    ignore_capitalization: bool = False,
    ignore_punctuation: bool = False,
    clean_text: bool = False,
) -> bool:
    """
    Match the reference and prediction value.

    Args:
        ref_value: The reference value, can be a string or a float.
        pred_value: The prediction value, can be a string or a float.
        ignore_capitalization: Whether to ignore capitalization when comparing strings.
        ignore_punctuation: Whether to ignore punctuation when comparing strings.
        clean_text: Whether to clean the text by replacing special characters before comparing.
    Returns:
        True if the reference and prediction value match, False otherwise.
    """
    try:
        # try to convert to float for input like "1.0"
        ref_value = float(ref_value)
        pred_value = float(pred_value)
        is_string = False
    except Exception:
        is_string = True

    if is_string:
        ref_value = str(ref_value)
        pred_value = str(pred_value)
        logger.debug(f"before processing: ref_value: {ref_value}, pred_value: {pred_value}")
        if ignore_capitalization:
            ref_value = ref_value.lower()
            pred_value = pred_value.lower()
        if ignore_punctuation:
            ref_value = remove_punctuations(ref_value)
            pred_value = remove_punctuations(pred_value)
        if clean_text:
            ref_value = clean_label(
                ref_value,
                langid="en",
                num_to_words=False,
                lowercase=ignore_capitalization,
            )
            pred_value = clean_label(
                pred_value,
                langid="en",
                num_to_words=False,
                lowercase=ignore_capitalization,
            )
        logger.debug(f"after processing: ref_value: {ref_value}, pred_value: {pred_value}")
        return ref_value == pred_value
    else:
        try:
            is_close = np.isclose(ref_value, pred_value)
            logger.debug(f"ref_value: {ref_value}, pred_value: {pred_value}")
            if isinstance(is_close, np.ndarray):
                is_close = all(is_close)
            return bool(is_close)
        except Exception as e:
            logger.error(f"Error checking for np.isclose(ref_value: {ref_value}, pred_value: {pred_value}): {e}")
            return False


def match_item(
    ref_value,
    pred_value,
    ignore_capitalization: bool = False,
    ignore_punctuation: bool = False,
    clean_text: bool = False,
) -> bool:
    """
    Recursively match a reference value against a prediction value.
    Handles dicts, lists, strings, and numbers.
    """
    if isinstance(ref_value, dict):
        if not isinstance(pred_value, dict):
            return False
        return match_dict(
            ref_value,
            pred_value,
            ignore_capitalization=ignore_capitalization,
            ignore_punctuation=ignore_punctuation,
            clean_text=clean_text,
        )
    elif isinstance(ref_value, list):
        if not isinstance(pred_value, list):
            return False
        return match_list(
            ref_value,
            pred_value,
            ignore_capitalization=ignore_capitalization,
            ignore_punctuation=ignore_punctuation,
            clean_text=clean_text,
        )
    else:
        return match_str_and_float(
            ref_value,
            pred_value,
            ignore_capitalization=ignore_capitalization,
            ignore_punctuation=ignore_punctuation,
            clean_text=clean_text,
        )


def match_dict(
    ref_dict: dict,
    pred_dict: dict,
    ignore_capitalization: bool = False,
    ignore_punctuation: bool = False,
    clean_text: bool = False,
) -> bool:
    """
    Check if pred_dict contains all keys and matching values from ref_dict.
    Additional keys in pred_dict are allowed.
    """
    for key, ref_val in ref_dict.items():
        if key not in pred_dict:
            return False
        if not match_item(
            ref_val,
            pred_dict[key],
            ignore_capitalization=ignore_capitalization,
            ignore_punctuation=ignore_punctuation,
            clean_text=clean_text,
        ):
            return False
    return True


def match_list(
    ref_list: list,
    pred_list: list,
    ignore_capitalization: bool = False,
    ignore_punctuation: bool = False,
    clean_text: bool = False,
) -> bool:
    """
    Check if each item in ref_list has a matching item in pred_list (order-independent).
    Each prediction item can only be matched once.
    """
    matched_indices = set()
    for ref_item in ref_list:
        found = False
        for i, pred_item in enumerate(pred_list):
            if i in matched_indices:
                continue
            if match_item(
                ref_item,
                pred_item,
                ignore_capitalization=ignore_capitalization,
                ignore_punctuation=ignore_punctuation,
                clean_text=clean_text,
            ):
                matched_indices.add(i)
                found = True
                break
        if not found:
            return False
    return True


def normalize_scenario_payload(payload):
    """Normalize a scenario payload (reference or prediction) to a canonical shape.

    Rules:
      - **List of exactly one dict** → unwrap to that single dict.
      - **Single dict** → return as-is.
      - **List of multiple dicts** → return as-is (legitimate push-path output:
        each entry is a separate ``<final_response>`` emission).
      - **Anything else** (scalar, ``None``, list of non-dicts) → return as-is.

    Used by both the deterministic comparator (``check_if_task_success``) and
    the LLM judge prep path (``runner.run_dynamic_evaluation``) so they apply
    the same shape-equivalence rule. Without this, the LLM judge reads the raw
    file text and deducts for cosmetic ``{...}`` vs. ``[{...}]`` differences
    that the deterministic comparator already treats as equivalent (its old
    "Situation 2" logic). One source of truth, applied to both scoring paths.
    """
    if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
        return payload[0]
    return payload


def check_if_task_success(
    *,
    reference: str,
    prediction: str,
    ignore_capitalization: bool = False,
    ignore_punctuation: bool = False,
    clean_text: bool = False,
    disallow_extra_items: bool = False,
) -> bool:
    """
    Check if the prediction is matches with the reference answer.

    Situations:
    1. If the reference is a dictionary, and the prediction is a dictionary:
      - The prediction should have the same keys and values as the reference.
      - Additional keys in prediction are allowed.

    2. If the reference is a dictionary, and the prediction is a list of dictionaries:
      -  the last dictionary in the prediction would be matched with the reference.

    3. If the reference is a list of dictionaries, and the prediction is a list of dictionaries:
      - For each dictionary in the reference, there should be a dictionary in the prediction that matches it
        according to the criteria in Situation 1.
      - The order of the dictionaries in the reference/prediction is not important.
      - All dictionaries in the reference should be matched with a dictionary in the prediction
        to be considered as a success.
      - If ``disallow_extra_items`` is True, the lengths must also match exactly
        (exact bijection — no extra prediction items tolerated).

    Both inputs are first passed through ``normalize_scenario_payload`` to
    collapse the list-of-1-dict / single-dict shape difference. The remaining
    situations 1-3 then handle the post-normalization shapes cleanly.

    Args:
        reference: The path to the reference json file.
        prediction: The path to the prediction json file.
        ignore_capitalization: Whether to ignore case when comparing strings.
        ignore_punctuation: Whether to ignore punctuation when comparing strings.
        clean_text: Whether to clean the text before comparing.
        disallow_extra_items: For list-of-dicts comparisons (Situation 3), require
            ``len(reference) == len(prediction)``. Default False preserves the
            lenient behavior where agent extras pass. Note: Situation 2 (single
            dict reference, list-of-dicts prediction) is unaffected — the last
            prediction dict is still picked and matched.
    Returns:
        True if the task is considered as successful, False otherwise.
    """
    with open(reference, "r") as f:
        reference_answer = json.load(f)
    with open(prediction, "r") as f:
        prediction_answer = json.load(f)

    # Apply the shape normalizer (list-of-1-dict → single dict) on both sides
    # so cosmetic wrapping doesn't bias the comparison or the downstream judge.
    reference_answer = normalize_scenario_payload(reference_answer)
    prediction_answer = normalize_scenario_payload(prediction_answer)

    # Situation 1: If the reference is a dictionary, and the prediction is a dictionary,
    # Convert to Situation 3
    if isinstance(reference_answer, dict):
        reference_answer = [reference_answer]
    if isinstance(prediction_answer, dict):
        prediction_answer = [prediction_answer]

    # Situation 2: If the reference is a dictionary, and the prediction is a list of dictionaries,
    # the last dictionary in the prediction would be matched with the reference.
    # Convert to Situation 3
    if len(reference_answer) == 1 and len(prediction_answer) > 1:
        prediction_answer = [prediction_answer[-1]]

    logger.debug(f"reference_answer: {reference_answer}")
    logger.debug(f"prediction_answer: {prediction_answer}")

    # Strict mode: exact bijection required. Combined with the existing
    # "each prediction matches at most one reference" constraint below,
    # equal lengths + every reference matched ⇒ every prediction matched.
    if disallow_extra_items and len(reference_answer) != len(prediction_answer):
        logger.debug(
            f"disallow_extra_items=True; length mismatch "
            f"(ref={len(reference_answer)}, pred={len(prediction_answer)}); fail"
        )
        return False

    result = True
    # Situation 3: For each reference dict, find a matching prediction dict (order-independent).
    matched_indices = set()
    for ref_dict in reference_answer:
        found = False
        for i, pred_dict in enumerate(prediction_answer):
            if i in matched_indices:
                continue
            if match_dict(
                ref_dict,
                pred_dict,
                ignore_capitalization=ignore_capitalization,
                ignore_punctuation=ignore_punctuation,
                clean_text=clean_text,
            ):
                matched_indices.add(i)
                found = True
                break
        if not found:
            result = False
            break
    logger.debug(f"success: {result}")
    return result


def _filtered_args(args: dict, compare_args: Optional[List[str]]) -> dict:
    """Filter a tool-call arguments dict by tau2's ``compare_args`` semantics.

    ``compare_args=None``  → return ``args`` verbatim (compare all).
    ``compare_args=[]``    → return ``{}`` (name-only match — empty == empty).
    ``compare_args=[k...]``→ return ``{k: args.get(k) for k in compare_args}``.

    Pitfall: do NOT write ``compare_args or "all"``-style fallbacks — ``[]`` is
    falsy and would silently collapse name-only matches into compare-all.
    Tau2 stores ``compare_args: None`` explicitly (the key is present), so
    ``ref.get("compare_args", "all")`` also doesn't fire the default. The
    explicit ``is None`` check below is the only safe sentinel.
    """
    if compare_args is None:
        return args
    return {k: args.get(k) for k in compare_args}


def _match_action(ref: dict, pred: dict) -> bool:
    """Deterministic action-record match honoring tau2's ``compare_args`` field.

    Name must always match. Argument comparison is delegated to ``_filtered_args``
    which applies the ``compare_args`` filter to both sides before equality.

    Reference records that omit ``compare_args`` (e.g. eva_airline records) get
    the default ``None`` from ``ref.get("compare_args")``, which means "compare
    all arguments" — preserving existing behavior for non-tau2 domains.

    Adapted from https://github.com/sierra-research/tau2-bench/tree/voice-user-sim-v1.0
        src/tau2/data_model/tasks.py:175-182
    """
    if ref.get("name") != pred.get("name"):
        return False
    compare_args = ref.get("compare_args")
    return _filtered_args(ref.get("arguments", {}), compare_args) == _filtered_args(
        pred.get("arguments", {}), compare_args
    )


class LLMJudge:
    """
    LLM-based judge for evaluating voice agent responses.

    Uses an OpenAI-compatible chat completions API to score how well a prediction
    matches a reference answer. Returns a float score between 0 and 1.

    Args:
        url: The URL of the OpenAI-compatible chat completions endpoint.
        model: The model name to use for judging.
        api_key: The API key. If None, will be loaded from environment variable.
        api_key_name: The environment variable name for the API key (default: "API_KEY").
        default_prompt: Custom default system prompt. If None, uses DEFAULT_PROMPT.
        **kwargs: Additional keyword arguments passed to the API payload (e.g., temperature, max_tokens).
    """

    DEFAULT_PROMPT = """You are a judge that evaluates the similarity between a reference answer and a prediction.
You will be given a reference and a prediction wrapped in XML tags.
Judge how well the prediction matches the reference in terms of correctness and completeness. If a field in prediction 
is not present in the reference, it means that the field is not required to check and can be ignored. 
Return a score between 0 and 1, where 0 means completely wrong and 1 means a perfect match.
You MUST return ONLY a JSON object in the following format, with no other text:
{"score": <score>, "reason": "<explanation of the score, also concrete with quoted evidence>"}"""

    SCENARIO_PROMPT = """You are a judge that evaluates voice agent performance in a conversational scenario.
You will be given some or all of the following XML-tagged inputs (only those available are included):
- <reference>: The reference answer (the expected outcome).
- <prediction>: The actual agent output (the "final response" the agent produced).
- <conversation>: The transcribed conversation turns between the user and the agent.
- <agent_context_history>: The agent's LLM context history, including system prompt and tool/function calls with their arguments and results.
- <user_context_history>: The simulated user's LLM context history, including the user-sim's own tool calls (e.g., phone-control tools in telecom scenarios where the user toggles airplane mode / data themselves).
- <nl_assertions>: A numbered list of natural-language assertions to consider when scoring.

Action `side` semantics. Each action in <reference> and <prediction> may carry a `side` field with one of two values:
- `side="agent"` — performed by the agent under test using its own LLM-callable tool. Score the agent on whether the corresponding tool call appears in <agent_context_history> with matching arguments.
- `side="user"` — performed by the simulated user using its own environment tool (e.g., telecom phone controls like `toggle_data`, `toggle_airplane_mode`). The agent does NOT have this tool and is NOT expected to invoke it. Score the agent on **guidance quality** — did it correctly diagnose the issue and clearly instruct the user to perform the action? Confirm the user-sim actually performed it by checking <user_context_history> for the corresponding tool call.
When `side` is absent on a reference action, default to `side="agent"` (legacy single-side domains: eva, tau2_airline, tau2_retail).

Strict attribution rule. Tool calls visible in <user_context_history> were made by the **simulated user**, NOT by the agent. Do NOT attribute them to the agent in your reasoning or deductions. The agent's tool calls live ONLY in <agent_context_history>. If a tool name appears in <user_context_history> but not in <agent_context_history>, the agent did NOT call it — treat it as the user-sim's action. Common cross-attribution mistake to avoid: claiming "the agent made an unnecessary call to `check_status_bar`" when that call actually appears only in <user_context_history>. Before deducting for any agent tool call, verify the call is in <agent_context_history> and quote its index.

Evaluate how well the agent performed by considering:
1. Whether <prediction> matches <reference>.
2. Whether the agent followed instructions correctly during the conversation.
3. For `side="agent"` reference actions: whether the agent called the corresponding tool with the correct arguments at the right time (use <agent_context_history> when present).
4. For `side="user"` reference actions: whether the agent correctly guided the user to perform that action (clear diagnosis + instruction). Do NOT deduct because "the agent didn't call the tool" — the agent has no such tool. Use <user_context_history> to confirm the user-sim actually executed the action.
5. Whether the agent avoided unnecessary or incorrect tool calls.
6. Whether the agent handled the conversation naturally and helpfully.
7. If <nl_assertions> is present, judge EACH numbered assertion independently against <conversation>, <agent_context_history>, and <user_context_history>. Emit a per-assertion verdict for each one.

Presentation-issue deduction cap. The following are "presentation issues" — they affect how the agent speaks, not whether the agent did the right thing:
- Missing or skipped `EndConversationTool` call. This is a voice-harness termination signal, not a domain policy requirement; the framework tracks termination separately via the conversation's stop reason.

When the agent successfully completed the task (reference actions matched, DB state correct, nl_assertions satisfied), **all presentation issues combined must not deduct more than 0.05 from the score**. Treat them as a single category capped at 0.05 total, regardless of how many individual presentation issues you find. The minimum score after only presentation issues is 0.95.

Minor issues that DO NOT deduct points from the score:
- Voice-realization violations: not spelling alphanumeric identifiers character-by-character (e.g. flight numbers, confirmation numbers, user IDs spoken as ordinary words instead of "S, K, seven, zero, three"), missing brand-specific farewells, prosody/formatting nits, etc.

Reason field requirements. The `reason` field MUST be concrete and debuggable:
- For each deduction, quote the specific phrase from <conversation> or <prediction> that was wrong (e.g. *Agent said "Flight SK703" instead of spelling it as "S, K, seven, zero, three"*).
- Cite the expected form alongside (e.g. *expected: "S, K, seven, zero, three"*).
- Group multiple instances of the same class of issue (e.g. "spelled 3 flight numbers and 2 confirmation numbers as ordinary words") instead of listing each individually, then quote 1-2 representative examples.
- Do NOT use vague phrases like "minor presentation issues" or "did not follow guidelines" without naming the specific guideline and the specific phrase that violated it.
- When deducting on a `side="user"` action: quote the agent's misguidance (e.g. *Agent said "Go to your phone's Settings..." instead of cuing the user to use `toggle_airplane_mode`*) — do NOT phrase the deduction as "the agent didn't call the tool", since the agent has no such tool.

Return a score between 0 and 1, where 0 means complete failure and 1 means perfect performance.

When <nl_assertions> is NOT present, return ONLY a JSON object with no other text:
{"score": <score>, "reason": "<explanation of the score, also concrete with quoted evidence>"}

When <nl_assertions> IS present, return ONLY a JSON object with no other text in this extended format:
{"score": <score>, "reason": "<explanation of the score, also concrete with quoted evidence>", "nl_assertion_verdicts": [{"index": <1-based assertion index>, "passed": <true|false>, "reason": "<per-assertion explanation, also concrete with quoted evidence>"}, ...]}
The ``nl_assertion_verdicts`` array MUST contain exactly one entry per assertion, with ``index`` matching the assertion's numbered position. ``passed`` is a strict boolean — only ``true`` if the assertion clearly holds given the evidence. Per-assertion ``reason`` follows the same concreteness rule: quote the specific evidence from the conversation, don't paraphrase."""

    def __init__(
        self,
        url: str,
        model: str,
        api_key: Optional[str] = None,
        api_key_name: str = "API_KEY",
        default_prompt: Optional[str] = None,
        **kwargs,
    ):
        self.url = url
        self.model = model
        self.api_key = api_key
        self.api_key_name = api_key_name
        if self.api_key is None:
            load_dotenv(override=True)
            self.api_key = os.getenv(self.api_key_name)
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.default_prompt = default_prompt or self.DEFAULT_PROMPT
        self.kwargs = kwargs

    def _get_payload(self, user_content: str, prompt: Optional[str] = None) -> dict:
        if not prompt:
            prompt = self.default_prompt
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content},
            ],
            **self.kwargs,
        }
        return payload

    def _parse_response(self, response: requests.Response) -> dict:
        """
        Parse the LLM response and extract the judgement JSON.

        Args:
            response: The HTTP response from the API.
        Returns:
            A dict with "score" (float) and optionally "reason" (str).
        Raises:
            ValueError: If the response cannot be parsed.
        """
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]

        # Try to parse JSON directly
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code block
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if match:
            return json.loads(match.group(1))

        # Try to find any JSON object in the content
        match = re.search(r'\{[^{}]*"score"\s*:\s*[\d.]+[^{}]*\}', content)
        if match:
            return json.loads(match.group(0))

        raise ValueError(f"Could not parse judgement JSON from LLM response: {content}")

    def judge(self, reference: str, prediction: str, prompt: Optional[str] = None) -> dict:
        """
        Judge the similarity between a reference and a prediction.

        Args:
            reference: The reference answer string.
            prediction: The prediction answer string.
            prompt: Optional custom system prompt. Uses default_prompt if not provided.
        Returns:
            A dict with "score" (float between 0 and 1) and "reason" (str).
            On error, returns {"score": 0.0, "reason": "<error message>"}.
        """
        user_content = f"<reference>\n{reference}\n</reference>\n\n<prediction>\n{prediction}\n</prediction>"
        payload = self._get_payload(user_content, prompt)
        # Attached to the returned dict (and saved into ``judge_result.json``
        # by the runner) so the exact text the judge saw is debuggable
        # without re-deriving from the source files — useful for triaging
        # surprising scores or iterating on the prompt.
        judge_input = {"system_prompt": prompt or self.default_prompt, "user_content": user_content}
        try:
            response = requests.post(self.url, headers=self.headers, json=payload)
            result = self._parse_response(response)
            result["score"] = float(result["score"])
            if "reason" not in result:
                result["reason"] = ""
            result["judge_input"] = judge_input
            logger.debug(f"LLMJudge result: {result}")
            return result
        except Exception as e:
            logger.error(f"LLMJudge error: {e}")
            return {"score": 0.0, "reason": f"Error: {e}", "judge_input": judge_input}

    def judge_file(self, reference: str, prediction: str, prompt: Optional[str] = None) -> dict:
        """
        Judge the similarity between a reference file and a prediction file.

        Args:
            reference: Path to the reference JSON file.
            prediction: Path to the prediction JSON file.
            prompt: Optional custom system prompt.
        Returns:
            A dict with "score" (float between 0 and 1) and "reason" (str).
        """
        with open(reference, "r") as f:
            reference_content = f.read()
            ref_json = json.loads(reference_content)
            # if the reference is a dictionary, convert it to a list of dictionaries
            if isinstance(ref_json, dict):
                ref_json = [ref_json]
            reference_content = json.dumps(ref_json)
        with open(prediction, "r") as f:
            prediction_content = f.read()
            pred_json = json.loads(prediction_content)
            if isinstance(pred_json, dict):
                pred_json = [pred_json]
            prediction_content = json.dumps(pred_json)
        logger.debug(f"reference_content: {reference_content}")
        logger.debug(f"prediction_content: {prediction_content}")
        return self.judge(reference_content, prediction_content, prompt)

    def judge_scenario(
        self,
        reference: str,
        prediction: str,
        conversation: Optional[list] = None,
        agent_context_history: Optional[list] = None,
        user_context_history: Optional[list] = None,
        nl_assertions: Optional[List[str]] = None,
        prompt: Optional[str] = None,
    ) -> dict:
        """
        Judge agent performance with full scenario context including conversation history.

        Args:
            reference: The reference answer string (or JSON string).
            prediction: The prediction answer string (or JSON string).
            conversation: List of conversation turns, each a dict with "role" and "text" keys.
            agent_context_history: Agent's LLM context messages (from
                ``bot_logs_agent/llm_context.json``). Contains the agent's
                tool calls + results. Rendered as ``<agent_context_history>``.
            user_context_history: Simulated user's LLM context messages
                (from ``bot_logs_user/llm_context.json``). Contains the
                user-sim's own tool calls — essential for dual-side
                domains like telecom where reference actions with
                ``side="user"`` are executed by the user-sim, not the
                agent. Rendered as ``<user_context_history>``.
            nl_assertions: Optional natural-language assertions (tau2 retail). When provided,
                each assertion is appended to the prompt and the LLM is instructed to emit
                a per-assertion verdict list. The returned dict gains a
                ``nl_assertion_verdicts`` field (one entry per assertion, ``{index, passed,
                reason}``) plus ``nl_assertion_pass_rate``. Missing/malformed verdicts are
                filled with ``passed=False`` so the runner can still aggregate cleanly.
                When ``None`` (or empty), the returned shape is the basic
                ``{score, reason}`` dict — no per-assertion fields.
            prompt: Optional custom system prompt. Uses SCENARIO_PROMPT if not provided.
        Returns:
            A dict with "score" (float between 0 and 1) and "reason" (str). When
            ``nl_assertions`` is non-empty, also includes ``nl_assertion_verdicts``
            (list of ``{index, passed, reason}``) and ``nl_assertion_pass_rate`` (float).
        """
        if not prompt:
            prompt = self.SCENARIO_PROMPT

        sections = [
            f"<reference>\n{reference}\n</reference>",
            f"<prediction>\n{prediction}\n</prediction>",
        ]

        if conversation:
            turns_text = "\n".join(f"[{turn.get('role', 'unknown')}]: {turn.get('text', '')}" for turn in conversation)
            sections.append(f"<conversation>\n{turns_text}\n</conversation>")

        if nl_assertions:
            numbered = "\n".join(f"{i + 1}. {a}" for i, a in enumerate(nl_assertions))
            sections.append(f"<nl_assertions>\n{numbered}\n</nl_assertions>")

        if agent_context_history:
            sections.append(
                f"<agent_context_history>\n{json.dumps(agent_context_history, indent=2)}\n</agent_context_history>"
            )
        if user_context_history:
            sections.append(
                f"<user_context_history>\n{json.dumps(user_context_history, indent=2)}\n</user_context_history>"
            )

        user_content = "\n\n".join(sections)
        payload = self._get_payload(user_content, prompt)
        # Attached to the returned dict (and saved into ``judge_result.json``
        # by the runner) so the exact text the judge saw is debuggable
        # without re-deriving from the source files — useful for triaging
        # surprising scores or iterating on the prompt.
        judge_input = {"system_prompt": prompt, "user_content": user_content}
        try:
            response = requests.post(self.url, headers=self.headers, json=payload)
            result = self._parse_response(response)
            result["score"] = float(result["score"])
            result.setdefault("reason", "")
            result["judge_input"] = judge_input
            if nl_assertions:
                # Normalize per-assertion verdicts: ensure exactly len(nl_assertions)
                # entries, in numbered order, with passed=False for any missing or
                # malformed entries (so the runner can aggregate without surprises).
                raw_verdicts = result.get("nl_assertion_verdicts") or []
                normalized: List[dict] = []
                by_index: Dict[int, dict] = {}
                for v in raw_verdicts:
                    if not isinstance(v, dict):
                        continue
                    try:
                        idx = int(v.get("index"))
                    except (TypeError, ValueError):
                        continue
                    if 1 <= idx <= len(nl_assertions):
                        by_index[idx] = v
                passes = 0
                for i in range(1, len(nl_assertions) + 1):
                    v = by_index.get(i)
                    passed = bool(v and v.get("passed") is True)
                    reason_text = (v or {}).get("reason", "") if v else "Missing verdict; treated as failed."
                    # Include the assertion text itself so judge_result.json
                    # is self-describing — operators can read a single file
                    # to see what was claimed AND what the judge decided,
                    # without cross-referencing scenario_config/metadata.json.
                    normalized.append(
                        {
                            "index": i,
                            "assertion": nl_assertions[i - 1],
                            "passed": passed,
                            "reason": reason_text,
                        }
                    )
                    if passed:
                        passes += 1
                result["nl_assertion_verdicts"] = normalized
                result["nl_assertion_pass_rate"] = passes / len(nl_assertions)
            return result
        except Exception as e:
            logger.error(f"LLMJudge error: {e}")
            err_result = {"score": 0.0, "reason": f"Error: {e}", "judge_input": judge_input}
            if nl_assertions:
                err_result["nl_assertion_verdicts"] = [
                    {"index": i + 1, "assertion": nl_assertions[i], "passed": False, "reason": f"Judge error: {e}"}
                    for i in range(len(nl_assertions))
                ]
                err_result["nl_assertion_pass_rate"] = 0.0
            return err_result
