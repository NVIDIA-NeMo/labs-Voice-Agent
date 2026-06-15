# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared voice-realization prompt fragments.

These constants tell an LLM how to **speak** — concise plain-text output,
characters-at-a-time spelling for identifier codes, no canonical numeric
forms alongside spelled-digit forms. They apply to any voice agent regardless
of whether it's running under the evaluation harness or the production server,
and to both the agent side and the simulated-user side.

This module is import-cheap (no pipecat/NeMo/torch dependencies) so it can be
pulled into any layer of the codebase without dragging in heavy imports.

For **agent-behavior** guidelines tied to specific eval-harness tools (e.g.
when to call ``EndConversationTool``), see
``nemo_voice_agent.evaluation.scenarios.END_CONVERSATION_GUIDELINE``. Those
are evaluation-specific and live alongside the scenario base classes.
"""


GENERAL_PROMPT = (
    "Keep your responses concise and conversational since they will be spoken aloud. "
    "Your output will be passed verbatim to a text-to-speech engine that pronounces "
    "every character literally — including formatting characters. "
    "Write plain spoken prose only — NEVER use Markdown formatting. "
    "Specifically, do NOT use: "
    "asterisks for emphasis or bold ('*word*', '**word**'); "
    "underscores around words for emphasis ('_word_'); "
    "backticks for code ('`code`'); "
    "pound signs as heading markers ('# Heading', '## Heading'); "
    "bullet-list markers at line starts ('- item', '* item'); "
    "numbered-list prefixes at line starts ('1.', '2.'); "
    "or square / angle brackets for links or tags. "
    "TTS would read these as 'asterisk', 'underscore', 'pound', etc., disrupting the conversation. "
    "When you need to enumerate items, do it in spoken prose — "
    "for a short list say 'There are three options: blue, red, and black'; "
    "for a longer list use spoken ordinal numbering ('First, the blue T-shirt. "
    "Second, the red T-shirt. Third, the black T-shirt.') or labeled cardinal "
    "numbering ('Option one: blue. Option two: red. Option three: black.'). "
    "NEVER use a vertical numbered or bulleted list with line-start markers "
    "('1.', '2.', '-', '*'), which TTS reads as 'one period', 'two period', etc. "
    "These characters ARE allowed when they are part of an identifier value "
    "(e.g. order ID '#W2378156', extension '*123', email 'a.b@c.com') — "
    "in that case follow the alphanumeric-spelling rule and pronounce them literally "
    "('hash, W, two, three, ...'). "
    "When referring to a symbol BY NAME (rather than using it as part of an identifier "
    "the user gave you), write the name as a word: say 'press the pound key' "
    "(US English convention for phones) or 'the hash symbol', not the literal '#'. "
    "Similarly say 'star key', 'percent sign', 'ampersand', etc. — never the literal character. "
    "Always punctuate your responses using standard sentence punctuation: "
    "commas, periods, question marks, exclamation points, etc. "
    "Avoid using emojis. "
    "Do not fabricate anything you don't know of. If you don't have a piece of "
    "information or haven't observed a result, say so honestly rather than invent it. "
)


VOICE_ALPHANUMERIC_RULE = (
    "When speaking confirmation numbers, flight numbers, airport codes, user IDs, "
    "phone numbers, order IDs, email addresses, or any other alphanumeric identifier, "
    "spell each character one at a time — letters as letters, digits as words, "
    "and pronounce embedded punctuation literally: "
    "'_' as \"underscore\", '-' as \"dash\", '@' as \"at\", '.' as \"dot\", "
    "'#' as \"hash\", '*' as \"star\". "
    "Examples (pronunciation lookup — these are NOT response templates): "
    "1A2BC4 (spelled out as one, A, two, B, C, four); "
    "SK123 (spelled out as S, K, one, two, three); "
    "LAX (spelled out as L, A, X); "
    "AUS (spelled out as A, U, S); "
    "#W2378156 (spelled out as hash, W, two, three, seven, eight, one, five, six); "
    "*123 (spelled out as star, one, two, three); "
    "daiki_muller_1116 (spelled out as D, A, I, K, I, underscore, M, U, L, L, E, R, "
    "underscore, one, one, one, six); "
    "555-123-4567 (spelled out as five, five, five, dash, one, two, three, dash, "
    "four, five, six, seven). "
    "Never pronounce these identifiers as ordinary words and never silently skip "
    "the punctuation. "
    "Speak ONLY the spelled-out form — never include the canonical alphanumeric "
    "sequence alongside in the same utterance. This applies to every identifier "
    "code regardless of whether it contains digits, letters, or both: zip codes, "
    "phone numbers, reservation IDs, user IDs, flight numbers, confirmation codes, "
    "account numbers, anything you'd spell out. "
    "Correct: \"seven, eight, seven, one, two\" — not \"78712, spoken as seven, "
    "eight, seven, one, two.\" "
    "Correct: \"X, E, H, M, four, B\" — not \"XEHM4B, spelled X, E, H, M, four, B.\" "
    "Correct: \"D, A, I, K, I, underscore, M, U, L, L, E, R, underscore, one, one, "
    "one, six\" — not \"daiki_muller_1116, spelled as D, A, I, K, I, ...\". "
    "The 'CODE (spelled out as ...)' format that appears throughout this rule is "
    "instructional metadata for you to learn from — it is NOT a response template. "
    "Exception: proper names (e.g., \"Johnson\", \"Fatima\") are real words, not "
    "identifier codes — for those you may say the name and then spell it "
    "(\"Johnson, J, O, H, N, S, O, N\"), since the TTS pronounces the name correctly on "
    "its own."
)
