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

"""Apache-2.0-licensed replacement for misaki's espeak-backed out-of-vocabulary (OOV)
word fallback.

misaki's dictionary-based English G2P leaves any word outside its lexicon with
``phonemes=None`` (silently dropped from synthesized audio) unless a ``fallback``
callable is supplied. Upstream, misaki/kokoro plug in an espeak-ng-backed fallback
(``misaki.espeak.EspeakFallback``), which requires the GPL-3.0-licensed
``phonemizer``/``espeak-ng`` stack — see ``_espeak_gpl_shim.py``.

This module provides an equivalent fallback built on ``g2p_en`` (Apache-2.0), which
predicts ARPAbet phonemes for arbitrary spellings via a small seq2seq model trained
on CMUdict, then maps ARPAbet onto misaki's own IPA-like phoneme inventory
(``misaki.en.US_VOCAB`` / ``GB_VOCAB``).
"""

from typing import Optional, Tuple


# ARPAbet (CMUdict) phone -> misaki phoneme symbol. Diphthongs collapse to the single
# characters misaki/Kokoro use (e.g. 'OW' -> 'O' for /oʊ/), matching the mapping
# misaki.espeak.EspeakFallback applies to espeak's raw IPA output.
_ARPABET_TO_MISAKI = {
    "AA": "ɑ",
    "AE": "æ",
    "AO": "ɔ",
    "AW": "W",
    "AY": "I",
    "B": "b",
    "CH": "ʧ",
    "D": "d",
    "DH": "ð",
    "EH": "ɛ",
    "EY": "A",
    "F": "f",
    "G": "ɡ",
    "HH": "h",
    "IH": "ɪ",
    "IY": "i",
    "JH": "ʤ",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "OW": "O",
    "OY": "Y",
    "P": "p",
    "R": "ɹ",
    "S": "s",
    "SH": "ʃ",
    "T": "t",
    "TH": "θ",
    "UH": "ʊ",
    "UW": "u",
    "V": "v",
    "W": "w",
    "Y": "j",
    "Z": "z",
    "ZH": "ʒ",
}
# AH ("hut"/schwa) and ER (rhotic vowel) map to different symbols depending on stress;
# ARPAbet doesn't split these into separate phones the way misaki's inventory does.
_AH_STRESSED = "ʌ"
_AH_UNSTRESSED = "ə"
_ER_STRESSED = "ɜɹ"
_ER_UNSTRESSED = "əɹ"

_STRESS_MARK = {"1": "ˈ", "2": "ˌ"}


def _arpabet_phone_to_misaki(phone: str) -> str:
    """Map one ARPAbet phone (optionally stress-suffixed, e.g. 'AH0') to misaki symbols."""
    base = phone.rstrip("012")
    stress = phone[len(base) :] or "0"
    if base == "AH":
        symbol = _AH_STRESSED if stress != "0" else _AH_UNSTRESSED
    elif base == "ER":
        symbol = _ER_STRESSED if stress != "0" else _ER_UNSTRESSED
    elif base in _ARPABET_TO_MISAKI:
        symbol = _ARPABET_TO_MISAKI[base]
    else:
        return ""  # Non-phoneme token (e.g. punctuation) — drop rather than corrupt output.
    return _STRESS_MARK.get(stress, "") + symbol


class ApacheG2PFallback:
    """Drop-in replacement for ``misaki.espeak.EspeakFallback`` with no GPL dependency.

    Assign an instance to ``KPipeline.g2p.fallback`` after construction.
    """

    def __init__(self):
        from g2p_en import G2p

        self._g2p = G2p()

    def __call__(self, token) -> Tuple[Optional[str], Optional[int]]:
        try:
            phones = self._g2p(token.text)
        except Exception:
            return None, None
        ps = "".join(_arpabet_phone_to_misaki(p) for p in phones if p.strip())
        if not ps:
            return None, None
        return ps, 2  # Rating 2 matches EspeakFallback's convention for fallback-resolved words.


_shared_fallback: Optional[ApacheG2PFallback] = None


def get_shared_fallback() -> ApacheG2PFallback:
    """Return a process-wide ``ApacheG2PFallback``, loading the g2p_en model once."""
    global _shared_fallback
    if _shared_fallback is None:
        _shared_fallback = ApacheG2PFallback()
    return _shared_fallback
