# SPDX-License-Identifier: MPL-2.0
"""Implemented/unreleased v174.0–v174.5 digital time-division switching."""
from .allocation import (AllocationInput, SlotCommand, SlotState, execute_slot_allocation,
                         validate_allocation_receipt)
from .core import (DigitalRequest, DigitalSwitchInput, FrameFault, demo_digital_input,
                   execute_digital_switch, validate_digital_receipt)
from .frames import (DigitalFrame, FrameObservation, FrameSchema, encode_frames,
                     interchange_frames, synchronize_frames, validate_interchange_receipt,
                     validate_sync_receipt)

__all__ = [
    "AllocationInput", "SlotCommand", "SlotState", "execute_slot_allocation", "validate_allocation_receipt",
    "DigitalRequest", "DigitalSwitchInput", "FrameFault", "demo_digital_input", "execute_digital_switch",
    "validate_digital_receipt", "DigitalFrame", "FrameObservation", "FrameSchema", "encode_frames",
    "interchange_frames", "synchronize_frames", "validate_interchange_receipt", "validate_sync_receipt",
]
from .evidence import (build_digital_report, run_digital_fault_battery, validate_digital_report,
                       validate_fault_battery)

__all__ += ["build_digital_report", "run_digital_fault_battery", "validate_digital_report", "validate_fault_battery"]
