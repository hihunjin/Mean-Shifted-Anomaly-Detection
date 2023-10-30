from typing import TYPE_CHECKING

from .base import (
    image_transform,
    available_models,
    load,
    VisionLanguageModel,
)

if TYPE_CHECKING:
    from . import clip
    from . import meru
else:
    from ..core import LazyModule

    clip = LazyModule(".clip", "clip", globals(), __package__)
    meru = LazyModule(".meru", "meru", globals(), __package__)




