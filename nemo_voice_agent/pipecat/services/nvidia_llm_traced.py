# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Tracing subclass of the pipecat-shipped ``NvidiaLLMService``.

The parent (``pipecat.services.nvidia.llm.NvidiaLLMService``) goes straight
to the upstream ``OpenAILLMService.get_chat_completions`` for the actual
HTTP call to the NVIDIA OpenAI-compatible endpoint. That path logs nothing
between "request about to be built" and "first usage metric arrives" —
which means a server-side stall (no first token returned) is invisible in
the bot log. The hang appears as silence after the last
``FunctionCallResultFrame``, with no signal as to whether the request was
even sent.

This subclass adds four diagnostic log lines per LLM call so the stall can
be pinpointed:

- ``[NVIDIA LLM REQ]``        before the HTTP request is dispatched.
- ``[NVIDIA LLM RESP]``       when the stream object is returned
                              (headers / first byte from the endpoint).
- ``[NVIDIA LLM FIRST CHUNK]``the first time the model emits a chunk.
- ``[NVIDIA LLM DONE]``       after the stream is exhausted.

Plus:
- ``[NVIDIA LLM ERR]``        on dispatch failure (with HTTP status + body).
- ``[NVIDIA LLM STREAM ERR]`` if the stream raises mid-iteration.

Drop-in replacement: ``from nemo_voice_agent.pipecat.services.nvidia_llm_traced
import NvidiaLLMService`` instead of the pipecat import.
"""

from __future__ import annotations

import json
import time
from typing import AsyncIterator

from loguru import logger
from openai.types.chat import ChatCompletionChunk
from pipecat.services.nvidia.llm import NvidiaLLMService as _PipecatNvidiaLLMService
from pipecat.services.openai.base_llm import OpenAILLMInvocationParams


class NvidiaLLMService(_PipecatNvidiaLLMService):
    """Drop-in replacement for ``pipecat.services.nvidia.llm.NvidiaLLMService``
    that adds endpoint-call tracing without changing any other behavior.

    The parent class delegates to ``OpenAILLMService.get_chat_completions``
    for the actual HTTP call to NVIDIA's OpenAI-compatible endpoint
    (``https://integrate.api.nvidia.com/v1`` by default). Between
    pipecat's "request about to be built" log and the first
    ``LLMService#0 TTFB`` metric, there is no visibility into whether
    the request was dispatched, whether the server returned headers,
    whether the first token has been produced, or whether the request
    errored out. A server-side stall (very common with large prompts /
    reasoning models / contended endpoints) surfaces only as silence
    after the previous frame in the bot log, with no signal of what is
    actually frozen.

    This subclass adds six possible log lines per LLM call so the freeze
    can be pinpointed to a specific stage:

    ============================== ================================================
    Log line                       Stage
    ============================== ================================================
    ``[NVIDIA LLM REQ]``           Right before ``_client.chat.completions.create``
                                   is awaited. Includes model, message count, tool
                                   count, last/recent message roles, and an
                                   approximate JSON-encoded request body size in
                                   bytes (useful for spotting context-bloat).
    ``[NVIDIA LLM RESP]``          The OpenAI client has returned the stream
                                   object — HTTP response headers / first byte
                                   received from the endpoint. ``header_latency``
                                   measures only the network round-trip + server
                                   queueing, not the model generation.
    ``[NVIDIA LLM FIRST CHUNK]``   The model has emitted its first streaming
                                   chunk. ``first_token_latency`` is the canonical
                                   TTFB measured here.
    ``[NVIDIA LLM DONE]``          Stream exhausted normally. Reports total
                                   chunks, total elapsed, and whether at least
                                   one chunk was seen.
    ``[NVIDIA LLM ERR]``           Dispatch raised before returning a stream
                                   (network failure, 4xx/5xx). Includes HTTP
                                   status and the first 500 chars of the response
                                   body if available, then the original exception
                                   is re-raised so callers see the same type.
    ``[NVIDIA LLM STREAM ERR]``    Streaming raised mid-iteration (silent
                                   disconnect, malformed chunk, etc.). Reports
                                   how many chunks were received before the
                                   failure.
    ============================== ================================================

    **Reading a hang.** Which line is the LAST one present tells you
    where the freeze sits:

    - ``REQ`` only → the HTTP call itself is stuck (DNS / TLS /
      gateway-side queueing).
    - ``REQ`` + ``RESP`` only → server accepted the request but is
      producing no tokens — model-side stall.
    - ``REQ`` + ``RESP`` + ``FIRST CHUNK`` → streaming started but
      didn't reach ``DONE`` — the connection died mid-generation with
      no exception bubbled.
    - An ``ERR`` or ``STREAM ERR`` line → server returned an explicit
      error; the included status / body tells you why.

    **Behavior contract with the parent.** Identical: no state added, no
    side effects other than ``logger.info`` / ``logger.error`` calls, no
    change to method signatures or return types. The async generator
    returned by ``get_chat_completions`` yields the same
    ``ChatCompletionChunk`` objects in the same order as the underlying
    OpenAI stream. Existing code that consumes this service via the
    parent interface (e.g. ``_process_context`` inherited from the
    parent) needs no changes.

    Usage::

        # Replace this:
        from pipecat.services.nvidia.llm import NvidiaLLMService
        # with this:
        from nemo_voice_agent.pipecat.services.nvidia_llm_traced import NvidiaLLMService
    """

    async def get_chat_completions(
        self, params_from_context: OpenAILLMInvocationParams
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Dispatch the chat-completion HTTP request with full tracing.

        Overrides ``OpenAILLMService.get_chat_completions``. Builds the
        request via the inherited ``build_chat_completion_params``,
        emits a ``[NVIDIA LLM REQ]`` line, awaits
        ``openai.AsyncOpenAI.chat.completions.create``, emits a
        ``[NVIDIA LLM RESP]`` line, then returns a wrapping async
        iterator (``_traced_stream``) that emits ``[NVIDIA LLM FIRST
        CHUNK]`` and ``[NVIDIA LLM DONE]`` as the stream is consumed.

        On dispatch failure (network error, 4xx/5xx, timeout) emits a
        ``[NVIDIA LLM ERR]`` line via ``_log_request_error`` and
        re-raises the original exception so the calling pipeline sees
        the same exception type as the un-traced parent.

        Args:
            params_from_context: The invocation params (messages, tools,
                tool_choice) the pipecat LLM-context adapter produced
                from the current ``LLMContext``.

        Returns:
            An async iterator that yields the same
            ``ChatCompletionChunk`` objects as the underlying OpenAI
            stream, in the same order. Pipecat callers consume it via
            ``async for chunk in chunk_stream`` — semantically
            identical to the parent's ``AsyncStream`` return.

        Raises:
            openai.OpenAIError: Re-raised verbatim from
                ``chat.completions.create`` on dispatch failure (after
                the ``[NVIDIA LLM ERR]`` line is logged).
            Exception: Any other exception from the OpenAI client is
                re-raised unchanged.
        """
        params = self.build_chat_completion_params(params_from_context)
        msgs = params.get("messages") or []
        last_role = msgs[-1].get("role") if msgs else "?"
        roles_tail = [m.get("role") for m in msgs[-4:]]
        approx_body_bytes = len(json.dumps(params, default=str))
        tools_n = len(params.get("tools") or [])
        logger.info(
            f"[NVIDIA LLM REQ] model={params.get('model')!r} "
            f"messages={len(msgs)} tools={tools_n} last_role={last_role} "
            f"recent_roles={roles_tail} approx_body_bytes={approx_body_bytes}"
        )

        t_req = time.monotonic()
        try:
            stream = await self._client.chat.completions.create(**params)
        except Exception as err:
            self._log_request_error(err, t_req)
            raise

        logger.info(
            f"[NVIDIA LLM RESP] header_latency={time.monotonic() - t_req:.3f}s "
            f"(stream object received; first chunk pending)"
        )
        return self._traced_stream(stream, t_req)

    async def _traced_stream(
        self,
        stream: AsyncIterator[ChatCompletionChunk],
        t_req: float,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Yield chunks from the underlying OpenAI stream with arrival tracing.

        On the first chunk, emits ``[NVIDIA LLM FIRST CHUNK]`` with the
        elapsed time since dispatch (canonical TTFB). On exhaustion or
        early return, emits ``[NVIDIA LLM DONE]`` with the chunk count,
        total elapsed, and whether any chunk was observed. If iteration
        raises mid-stream, emits ``[NVIDIA LLM STREAM ERR]`` before
        re-raising — distinct from the dispatch-time ``[NVIDIA LLM
        ERR]`` so a silent mid-stream disconnect is identifiable in the
        log.

        Args:
            stream: The async iterator returned by
                ``openai.AsyncOpenAI.chat.completions.create``. Any
                ``AsyncIterator[ChatCompletionChunk]`` works; the
                ``AsyncStream`` returned by the OpenAI client satisfies
                this contract.
            t_req: ``time.monotonic()`` taken immediately before the
                dispatch ``await`` in ``get_chat_completions``. Used as
                the zero point for first-token and total latency.

        Yields:
            Each ``ChatCompletionChunk`` from ``stream``, in order, with
            no modification.

        Raises:
            Exception: Re-raises any exception raised by iterating
                ``stream`` (after logging ``[NVIDIA LLM STREAM ERR]``).
        """
        first_chunk_logged = False
        n_chunks = 0
        try:
            async for chunk in stream:
                n_chunks += 1
                if not first_chunk_logged:
                    logger.info(f"[NVIDIA LLM FIRST CHUNK] first_token_latency={time.monotonic() - t_req:.3f}s")
                    first_chunk_logged = True
                yield chunk
        except Exception as err:
            logger.error(
                f"[NVIDIA LLM STREAM ERR] {type(err).__name__}: {err} "
                f"(chunks_received={n_chunks}, elapsed={time.monotonic() - t_req:.3f}s)"
            )
            raise
        finally:
            logger.info(
                f"[NVIDIA LLM DONE] total_chunks={n_chunks} "
                f"total_elapsed={time.monotonic() - t_req:.3f}s "
                f"first_chunk_seen={first_chunk_logged}"
            )

    def _log_request_error(self, err: Exception, t_req: float) -> None:
        """Emit a single ``[NVIDIA LLM ERR]`` line on dispatch failure.

        Best-effort extraction of HTTP status and response body from the
        exception. The ``openai`` v1+ SDK attaches the underlying
        ``httpx.Response`` as ``err.response`` on
        ``APIStatusError`` / ``APITimeoutError`` / friends. For
        non-HTTP errors (DNS, TLS, ``asyncio.TimeoutError``) ``status``
        and ``body`` will be empty — only the exception type and
        message are logged. The function never raises; failures while
        formatting the body are swallowed so this helper can't mask the
        original error in the caller's ``raise``.

        Args:
            err: The exception raised by
                ``openai.AsyncOpenAI.chat.completions.create``.
            t_req: ``time.monotonic()`` taken immediately before the
                dispatch ``await`` in ``get_chat_completions``. Used to
                report how long the failing call took.
        """
        body = ""
        response = getattr(err, "response", None)
        if response is not None:
            try:
                body = f" body={response.text[:500]!r}"
            except Exception:
                body = ""
        status = getattr(response, "status_code", None)
        logger.error(
            f"[NVIDIA LLM ERR] {type(err).__name__} status={status} "
            f"elapsed={time.monotonic() - t_req:.3f}s msg={str(err)[:300]!r}{body}"
        )
