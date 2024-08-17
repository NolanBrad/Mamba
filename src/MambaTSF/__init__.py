# The root of the entire project

# This is the version number of your code.  This will change each time you make
# a new release.  The most popular approach is to use Semantic Versioning,
# described at https://semver.org/
__version__ = "0.1.7"


# Version History
# 0.1.1 Intial relase with tiny and min mamba models
# 0.1.2 Add iTransformer and utils
# 0.1.3 Module name changes and correct imports
# 0.1.4 Corrected imports
# 0.1.5 Export utils
# 0.1.6 Correct iTransformer import of utils
# 0.1.7 Corrected tools adjust_learning_rate

import MambaTSF.utils

from MambaTSF.mamba_tiny import MambaTinyConfig
from MambaTSF.mamba_tiny import MambaBlock as MambaBlockTiny
from MambaTSF.mamba_tiny import Mamba as MambaTiny

from MambaTSF.mamba_min import MambaMinConfig
from MambaTSF.mamba_min import MambaBlock as MambaBlockMin
from MambaTSF.mamba_min import Mamba as MambaMin

from MambaTSF.mamba_sd import S_D_MambaConfig
from MambaTSF.mamba_sd import SDMamba

from MambaTSF.iTransformer import iTransModel


#__all__ = [ "utils", "iTransformer", "mamba_min", "mamba_tiny", "mamba_sd" ]
