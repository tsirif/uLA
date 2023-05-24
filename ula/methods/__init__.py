from ula.methods.base import BaseMethod
from ula.methods.sla import SupervisedLogitAdjustment
from ula.methods.ula import UnsupervisedLogitAdjustment
from ula.methods.lc import LogitCorrection
from ula.methods.mocov2plus import MoCoV2Plus


SUPERVISED_METHODS = {
    "sla": SupervisedLogitAdjustment,
    "ula": UnsupervisedLogitAdjustment,
    "lc": LogitCorrection,
    }


METHODS = {
    # base classes
    "base": BaseMethod,
    # methods
    "mocov2plus": MoCoV2Plus,
}
__all__ = [
    "BaseMethod",
    "SupervisedLogitAdjustment",
    "UnsupervisedLogitAdjustment",
    "LogitCorrection",
    "MoCoV2Plus",
]
