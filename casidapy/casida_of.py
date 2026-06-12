import numpy as np

from dftpy.functional import Functional, TotalFunctional
from dftpy.functional import LocalPseudo
from dftpy.functional.external_potential import ExternalPotential


def build_of_functional_context(rho_ks, ions, xc_name, pp_list):
    """
    Build an OF-inspired functional context for KS-Casida.

    This does not run a standalone OF-DFT ground state. Instead, it constructs
    an external potential from TFvW + XC + pseudo + Hartree + VW terms and
    injects it into TotalFunctional so CasidaKS_MPI can use an OF-style context.
    """
    pseudo = LocalPseudo(grid=rho_ks.grid, ions=ions, PP_list=pp_list)
    core = pseudo.core_density

    ke = Functional(type="KEDF", name="TFvW")
    xc = Functional(type="XC", name=xc_name, core_density=core)
    hartree = Functional(type="HARTREE")
   # v_w = Functional(type="VW")

    v_ext = -(
        ke(rho_ks).potential
        + xc(rho_ks).potential
        + pseudo(rho_ks).potential
        + hartree(rho_ks).potential
    )
   # vw = v_w(rho_ks).potential
   # v_ext = -(vs + vw)

    ext = ExternalPotential(v=v_ext)
    totalfunctional = TotalFunctional(KE=ke, XC=xc, PSEUDO=pseudo)
    totalfunctional.UpdateFunctional(keysToRemove=["HARTREE", "PSEUDO"])
    totalfunctional.UpdateFunctional(newFuncDict={"EXT": ext})

    return {
        "totalfunctional": totalfunctional,
        "pseudo": pseudo,
        "core_density": core,
        "v_ext_norm": float(np.linalg.norm(np.asarray(v_ext))),
    }
