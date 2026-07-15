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

"""Shared bot-server runner.

Bots build their own services, RTVI processor, pipeline, and ``PipelineTask``;
this module handles the boilerplate that sits around the task — transport event
handlers, the RTVI ``on_client_ready`` kick-off, audio-logger finalization,
shutdown, and optional FastAPI ``/connect`` endpoint.

The runner makes no assumptions about pipeline contents: it only needs the
``task``, ``ws_transport``, and ``rtvi`` the bot already constructed. Per-bot
customization (what to reset on disconnect, which initial frame to queue on
client-ready) is passed via keyword arguments.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Awaitable, Callable, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pipecat.frames.frames import EndTaskFrame, Frame
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frameworks.rtvi import RTVIProcessor

from nemo_voice_agent.pipecat.processors.frameworks.rtvi_actions import TaskRef
from nemo_voice_agent.pipecat.services.nemo.audio_logger import AudioLogger
from nemo_voice_agent.pipecat.transports.network.websocket_server import (
    WebsocketServerTransport,
)
from nemo_voice_agent.utils.websocket_url import build_websocket_url


def _reset_services(services: Optional[List[Any]]) -> None:
    for service in services or []:
        if service is not None and hasattr(service, "reset"):
            service.reset()


async def run_bot_websocket_server(
    task: PipelineTask,
    ws_transport: WebsocketServerTransport,
    rtvi: RTVIProcessor,
    *,
    task_ref: Optional[TaskRef] = None,
    audio_logger: Optional[AudioLogger] = None,
    talk_first: bool = True,
    initial_frame_factory: Optional[Callable[[], Frame]] = None,
    on_disconnect_reset_services: Optional[List[Any]] = None,
    on_disconnect_hook: Optional[Callable[[], Awaitable[None]]] = None,
) -> None:
    """Wire event handlers onto ``ws_transport``/``rtvi`` and run the pipeline.

    Args:
        task: Fully constructed ``PipelineTask`` with observers already attached.
        ws_transport: Transport the task's pipeline uses for I/O.
        rtvi: RTVI processor embedded in the pipeline. Must already have any
            actions the bot wants registered.
        task_ref: Optional ``TaskRef`` passed to RTVI action factories. Populated
            here so handlers can queue ``EndTaskFrame`` onto the live task.
        audio_logger: Audio logger to finalize on disconnect / shutdown.
        talk_first: If True, queue ``initial_frame_factory()`` on client-ready.
        initial_frame_factory: Callable returning the Frame to kick off with
            (e.g. ``lambda: LLMRunFrame()``). Invoked once per client-ready.
        on_disconnect_reset_services: Services whose ``.reset()`` is called on
            client disconnect. ``None`` entries are skipped.
        on_disconnect_hook: Optional extra async cleanup called on disconnect
            after the default reset logic.
    """
    if task_ref is not None:
        task_ref.task = task
        task_ref.running = True

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi_proc: RTVIProcessor):  # noqa: ARG001
        logger.info(f"Pipecat client ready with talk_first={talk_first}.")
        await rtvi_proc.set_bot_ready()
        if talk_first and initial_frame_factory is not None:
            try:
                logger.info("Kicking off the conversation...")
                await task.queue_frames([initial_frame_factory()])
            except Exception as e:
                logger.error(f"Error queuing initial frame: {e}")
        else:
            logger.info("Pipecat client ready, listening...")

    @ws_transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):  # noqa: ARG001
        logger.info(f"Pipecat Client connected from {client.remote_address}")
        rtvi._client_ready = False
        rtvi._bot_ready = False

    @ws_transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):  # noqa: ARG001
        logger.info(f"Pipecat Client disconnected from {client.remote_address}")
        if audio_logger:
            audio_logger.finalize_session()
            logger.info("Audio logger session finalized")
        if task_ref is None or task_ref.running:
            try:
                await task.queue_frames([EndTaskFrame()])
                _reset_services(on_disconnect_reset_services)
                if on_disconnect_hook is not None:
                    await on_disconnect_hook()
            except Exception as e:
                if "ConnectionClosedOK" not in str(e) and "1005" not in str(e):
                    logger.warning(f"Error sending EndTaskFrame: {e}")
                else:
                    logger.info(f"Normal connection closure: {e}")

    @ws_transport.event_handler("on_session_timeout")
    async def on_session_timeout(transport, client):  # noqa: ARG001
        logger.info(f"Session timeout for {client.remote_address} (kept server alive)")
        if audio_logger:
            audio_logger.finalize_session()
            logger.info("Audio logger session finalized")
        if task_ref is None or task_ref.running:
            try:
                await task.queue_frames([EndTaskFrame()])
            except Exception as e:
                logger.error(f"Error sending EndTaskFrame: {e}")

    logger.info("Starting pipeline runner...")
    try:
        runner = PipelineRunner()
        await asyncio.wait_for(runner.run(task), timeout=None)
    except asyncio.TimeoutError:
        logger.info("Pipeline runner timeout (should not happen with no timeout)")
    except Exception as e:
        logger.error(f"Pipeline runner error: {e}")
    finally:
        if task_ref is not None:
            task_ref.running = False
        if audio_logger:
            audio_logger.finalize_session()
            logger.info("Audio logger session finalized on shutdown")
        logger.info("Pipeline runner stopped")


def create_fastapi_app(websocket_port: int, public_host: str = "127.0.0.1", ws_scheme: str = "ws") -> FastAPI:
    """FastAPI app with CORS + ``/connect`` + a stub ``/ws`` endpoint.

    ``/connect`` returns the ws URL the pipecat client should dial. ``/ws`` is
    kept as a placeholder mirroring the original bot scripts' unimplemented
    FastAPI websocket path.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # noqa: ARG001
        yield

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        logger.info("WebSocket connection accepted")
        try:
            raise NotImplementedError("FastAPI websocket endpoint is not implemented")
        except Exception as e:
            logger.info(f"Exception in run_bot: {e}")

    @app.post("/connect")
    async def bot_connect() -> Dict[Any, Any]:
        ws_url = build_websocket_url(public_host, websocket_port, ws_scheme)
        logger.info(f"Returning WebSocket URL: {ws_url}")
        return {"ws_url": ws_url}

    return app


async def run_bot_with_fastapi(
    ws_coro: Awaitable[None],
    app: FastAPI,
    host: str,
    fastapi_port: int,
) -> None:
    """Run the websocket server coroutine and uvicorn concurrently."""
    config = uvicorn.Config(app, host=host, port=fastapi_port)
    server = uvicorn.Server(config)
    try:
        await asyncio.gather(ws_coro, server.serve())
    except asyncio.CancelledError:
        logger.info("Tasks cancelled (probably due to shutdown).")


__all__ = [
    "run_bot_websocket_server",
    "create_fastapi_app",
    "run_bot_with_fastapi",
]
