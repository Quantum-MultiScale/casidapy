"""Backward-compat shim for older eDFTpy imports.

Prefer ``casidapy.adapter.qepy``; this module re-exports the same symbols so
``from casidapy.qepy_adapter import extract_casida_inputs_from_qepy_driver``
keeps working.
"""
from casidapy.adapter.qepy import *  # noqa: F401,F403
from casidapy.adapter.qepy import extract_casida_inputs_from_qepy_driver

__all__ = ["extract_casida_inputs_from_qepy_driver"]
