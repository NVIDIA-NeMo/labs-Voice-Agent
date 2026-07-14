# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from pipecat.transports.base_transport import TransportParams as _TransportParams


class TransportParams(_TransportParams):
    can_create_user_frames: bool = True
