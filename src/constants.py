from src.tasks.mix_group_addition import MixCyclicGroupAddition
from src.tasks.mix_group_addition import MixRosetteGroupAddition
from src.tasks.mix_group_addition import MixDihedralGroupAddition
from src.tasks.mix_group_addition import MixMonoidAddition

TASK_MAP = {
    "mixcyclic": MixCyclicGroupAddition,
    "mixdihedral": MixDihedralGroupAddition,
    "mixrosette": MixRosetteGroupAddition,
    "mixmonoid": MixMonoidAddition
}
