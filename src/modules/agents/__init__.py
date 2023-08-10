REGISTRY = {}

from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
from .icm_agent import ICMAgent
from .go_agent import GoExploreAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["icm"] = ICMAgent
REGISTRY["go_explore"] = GoExploreAgent