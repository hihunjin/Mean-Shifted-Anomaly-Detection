from typing import Protocol, List, Union, Tuple, Optional, Callable, Literal

import torch
from torch import Tensor, IntTensor
from torch.types import _dtype
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from clip.clip import _convert_image_to_rgb

from PIL.Image import Image


__all__ = [
    "image_transform",
    "available_models",
    "load",
    "VisionLanguageModel",
]


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def image_transform(n_px) -> T.Compose:
    return T.Compose([
        T.Resize(n_px, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(n_px),
        _convert_image_to_rgb,
        T.ToTensor(),
    ])


class VisionLanguageModel(Protocol):

    def tokenize(self, texts: Union[str, List[str]]) -> IntTensor:
        ...

    def encode_image(
        self,
        images: Tensor,
        *,
        project: bool = False,
    ) -> Tensor:
        ...

    def encode_text(
        self,
        texts: Union[IntTensor, List[IntTensor], List[str], List[List[str]]],
        *,
        project: bool = False,
    ) -> Tensor:
        ...

    def similarity(
        self,
        image_features: Tensor,
        text_features: Tensor,
        *,
        project: bool = True,
        temperature: Optional[float] = None,
    ) -> Tensor:
        ...

    def forward(
        self,
        images: Tensor,
        texts: Union[IntTensor, List[IntTensor], List[str], List[List[str]]],
        *,
        temperature: Optional[float] = None,
    ) -> Tensor:
        ...

    @property
    def dtype(self) -> _dtype:
        ...

    @property
    def device(self) -> torch.device:
        ...


def available_models(impl: Literal["clip", "meru"]) -> List[str]:
    """Returns the names of available models for the given implementation"""
    if impl == "clip":
        from .clip import available_models as _clip_available_models
        return _clip_available_models()
    elif impl == "meru":
        from .meru import available_models as _meru_available_models
        return _meru_available_models()
    else:
        raise ValueError(f"Unknown implementation: {impl}")


def load(
    name: str,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    jit: bool = False,
    download_root: Optional[str] = None,
    impl: Optional[Literal["clip", "meru"]] = None,
) -> Tuple[VisionLanguageModel, Callable[[Image], Tensor]]:
    """Load a CLIP or MERU model

    Automatically determines the implementation from the model name if `impl` is not specified.
    - If the model name starts with "CLIP/", it uses the CLIP implementation.
    - If the model name starts with "MERU/", it uses the MERU implementation.

    Parameters
    ----------
    name: str
        A model name listed by `available_models(impl="clip")` or `available_models(impl="meru")`.

    device: Union[str, torch.device]
        The device to put the loaded model.

    jit: bool
        Whether to load the optimized JIT model or more hackable non-JIT model.

    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip" or "~/.cache/meru"
        depending on the implementation.

    impl: Optional[Literal["clip", "meru"]]
        The implementation to use. By default, it uses "clip" if "CLIP/*" is passed as the name,
        and "meru" if "MERU/*" is passed.

    Returns
    -------
    model: VisionLanguageModel
        The torch model compatible with the VisionLanguageModel protocol

    preprocess : Callable[[PIL.Image], Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if impl is None:
        if name.startswith("CLIP/"):
            impl = "clip"
        elif name.startswith("MERU/"):
            impl = "meru"
        else:
            raise ValueError(
                f"Could not determine the implementation from the model name: {name}. "
                "Please specify the implementation explicitly with the `impl` argument."
            )

    if impl == "clip":
        from .clip import load as _clip_load
        return _clip_load(name, device=device, jit=jit, download_root=download_root)
    elif impl == "meru":
        from .meru import load as _meru_load
        return _meru_load(name, device=device, jit=jit, download_root=download_root)
    else:
        raise ValueError(f"Unknown implementation: {impl}")
