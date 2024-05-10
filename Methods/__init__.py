
from dataclasses import dataclass
from simple_parsing import mutable_field

# from .models.fixed_extractor import OvAInn

from .models.cnn_independent_experts import ExpertMixture
from .models.DTGN import DTGN_net
from .models.DTGN_components import LMC_conv_block
from .models.cnn_soft_gated_lifelong_dynamic import CNNSoftGatedLLDynamic

@dataclass
class ModelOptions():
    DTGN: DTGN_net.Options = mutable_field(DTGN_net.Options)
    Module: LMC_conv_block.Options = mutable_field(LMC_conv_block.Options)
    Experts: ExpertMixture.Options = mutable_field(ExpertMixture.Options)
    SGNet: CNNSoftGatedLLDynamic.Options = mutable_field(CNNSoftGatedLLDynamic.Options)
