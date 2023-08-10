REGISTRY = {}

from .basic_controller import BasicMAC
from .non_shared_controller import NonSharedMAC
from .maddpg_controller import MADDPGMAC
from .icm_controller import ICMMAC
from .go_explore_controller import GoExploreMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["non_shared_mac"] = NonSharedMAC
REGISTRY["maddpg_mac"] = MADDPGMAC
REGISTRY["icm_mac"] = ICMMAC
REGISTRY["go_explore_mac"] = GoExploreMAC