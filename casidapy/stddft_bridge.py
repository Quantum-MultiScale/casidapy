from __future__ import annotations

from dataclasses import replace
from typing import Optional

from casidapy.casida_api import CasidaInputs, CasidaOptions, CasidaResults
from casidapy.casida_engine import run_casida_in_memory
from casidapy.qepy_adapter import extract_casida_inputs_from_qepy_driver

# Class to bridge the STDDFT and Casida engines
class STDDFTBridge:
    def __init__(self, default_options: CasidaOptions):
        self.default_options = default_options
    def _merge_options(self, overrides: Optional[dict]) -> CasidaOptions:
        if not overrides:
            return self.default_options
        return replace(self.default_options, **overrides)
# Function to run the Casida engine from a QEpy driver
    def run_from_qepy_driver(
        self,
        driver,
        atoms,
        *,
        target_grid_nr=None,
        option_overrides: Optional[dict] = None,
        comm=None,
    ) -> CasidaResults:
        inputs = extract_casida_inputs_from_qepy_driver(
            driver=driver,
            atoms=atoms,
            target_grid_nr=target_grid_nr,
        )
        opts = self._merge_options(option_overrides)
        return run_casida_in_memory(inputs=inputs, options=opts, comm=comm)
# Function to run the Casida engine from arrays
    def run_from_arrays(
        self,
        inputs: CasidaInputs,
        *,
        option_overrides: Optional[dict] = None,
        comm=None,
    ) -> CasidaResults:
        opts = self._merge_options(option_overrides)
        return run_casida_in_memory(inputs=inputs, options=opts, comm=comm)