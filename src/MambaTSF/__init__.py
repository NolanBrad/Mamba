# The root of the entire project

# This is the version number of your code.  This will change each time you make
# a new release.  The most popular approach is to use Semantic Versioning,
# described at https://semver.org/
__version__ = "0.1.18"


# Version History
# 0.1.1 Intial relase with tiny and min mamba models
# 0.1.2 Add iTransformer and utils
# 0.1.3 Module name changes and correct imports
# 0.1.4 Corrected imports
# 0.1.5 Export utils
# 0.1.6 Correct iTransformer import of utils
# 0.1.7 Corrected tools adjust_learning_rate
# 0.1.8 DataEmbedding_inverted forward not handling x_mark correctly
# 0.1.9 Add iTransConfig
# 0.1.10 Correct SDMamba caling Encoder forward
# 0.1.11 Correct model objects used in SDMamba Layers
# 0.1.12 Remove matplotlib and functions visuals
# 0.1.13 Add out_len to SDMamaba
# 0.1.14 Correct implementation of out_len to out_var in SDMamaba
# 0.1.15 Add linear layer for output variates
# 0.1.16 Relocate linear layer for output variates to after norm
# 0.1.17 Add RevIn, Shift and sparsermok
# 0.1.18 Correct sparsermok importing Int

import MambaTSF.utils

from MambaTSF.mamba_tiny import MambaTinyConfig
from MambaTSF.mamba_tiny import MambaBlock as MambaBlockTiny
from MambaTSF.mamba_tiny import Mamba as MambaTiny

from MambaTSF.mamba_min import MambaMinConfig
from MambaTSF.mamba_min import MambaBlock as MambaBlockMin
from MambaTSF.mamba_min import Mamba as MambaMin

from MambaTSF.mamba_sd import S_D_MambaConfig
from MambaTSF.mamba_sd import SDMamba

from MambaTSF.iTransformer import iTransModel, iTransConfig

from MambaTSF.RevIN import RevIN
from MambaTSF.Shift import Shift
from MambaTSF.sparsermok import SparseRMoK

__all__ = [ "utils", "iTransformer", "mamba_min", "mamba_tiny", "mamba_sd", "RevIN", "Shift", "sparsermok" ]
