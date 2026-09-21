# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Canonical VTA backend selection helpers."""

import os


SUPPORTED_BACKENDS = ("fsim", "tsim")
LEGACY_SIMULATOR_TARGETS = ("sim", "tsim")


def is_simulator_backend(value):
    """Return whether *value* is a canonical simulator backend."""
    return value in SUPPORTED_BACKENDS


def normalize_backend(backend=None, simulator=None):
    """Resolve an explicit backend against the VTA_BACKEND contract."""
    if backend is not None and simulator is not None and backend != simulator:
        raise ValueError(
            f"backend mismatch: backend={backend!r}, simulator={simulator!r}"
        )
    selected = simulator if simulator is not None else backend
    configured = os.environ.get("VTA_BACKEND")
    if selected is not None and configured is not None and configured != selected:
        raise ValueError(
            f"backend mismatch: VTA_BACKEND={configured!r}, explicit backend={selected!r}"
        )
    if selected is None:
        selected = configured
    if is_simulator_backend(selected):
        return selected
    raise ValueError(
        "unsupported VTA backend {!r}; set VTA_BACKEND to fsim or tsim, "
        "or pass an explicit simulator/backend parameter".format(selected)
    )
