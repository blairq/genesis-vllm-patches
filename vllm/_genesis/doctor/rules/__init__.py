# SPDX-License-Identifier: Apache-2.0
"""Doctor rules sub-package."""
from vllm._genesis.doctor.rules.w4a16 import check_w4a16_artifact  # noqa: F401

__all__ = ["check_w4a16_artifact"]
