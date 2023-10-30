# Code adapted from: https://github.com/openai/CLIP

import os
from typing import Callable, Tuple, Union, Optional, cast, List
from functools import wraps

import torch
from torch import Tensor, IntTensor
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from PIL.Image import Image

import clip
from clip.model import CLIP as _CLIP
from clip.clip import load as _load, _MODELS

from .base import image_transform, VisionLanguageModel


__all__ = [
    "load",
    "available_models",
    "CLIP",
]

_IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
_IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return [f"CLIP/{v}" for v in _MODELS.keys()]


class CLIP(_CLIP, VisionLanguageModel):

    def tokenize(self, texts: Union[str, List[str]]) -> IntTensor:
        tokens = clip.tokenize(texts, context_length=self.context_length, truncate=True)
        tokens = cast(IntTensor, tokens)
        return tokens

    def encode_image(
        self,
        images: Tensor,
        *,
        project: bool = False,
    ) -> Tensor:
        images = images.to(self.device)
        images = TF.normalize(images, mean=_IMAGE_MEAN, std=_IMAGE_STD)
        image_features = super().encode_image(images)

        if project:
            image_features = F.normalize(image_features, dim=-1)

        return image_features

    def encode_text(
        self,
        texts: Union[IntTensor, List[IntTensor], List[str], List[List[str]]],
        *,
        project: bool = False,
    ) -> Tensor:
        if isinstance(texts, list):
            if isinstance(texts[0], list):  # List[List[str]]
                texts = [self.tokenize(cast(str, t)) for t in texts]
            elif isinstance(texts[0], str):  # List[str]
                texts = self.tokenize(cast(str, texts))

        texts = cast(Union[IntTensor, List[IntTensor]], texts)

        if isinstance(texts, list):  # List[IntTensor]
            _clip_encode_text = super().encode_text
            if project:
                _encode_text = lambda t: F.normalize(_clip_encode_text(t), dim=-1)
            else:
                _encode_text = _clip_encode_text

            text_features = torch.stack([_encode_text(t.to(self.device)).mean(dim=0) for t in texts], dim=0)
        else:  # IntTensor
            text_features = super().encode_text(texts.to(self.device))

        if project:
            text_features = F.normalize(text_features, dim=-1)

        return text_features

    def encode_prompt(
        self,
        prompts: Tensor,
        eot_idxs: IntTensor,
        *,
        project: bool = False,
    ):
        x = prompts.type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), eot_idxs] @ self.text_projection

        if project:
            x = F.normalize(x, dim=-1)

        return x

    def similarity(
        self,
        image_features: Tensor,
        text_features: Tensor,
        *,
        project: bool = True,
        temperature: Optional[float] = None,
    ) -> Tensor:
        if project:
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

        image_features = image_features.type(self.dtype)
        text_features = text_features.type(self.dtype)

        logit_scale = self.logit_scale.exp() if temperature is None else (1 / temperature)
        image_logits = image_features @ text_features.t()
        image_probs = (logit_scale * image_logits).softmax(dim=-1)
        return image_probs

    def forward(
        self,
        images: Tensor,
        texts: Union[IntTensor, List[IntTensor], List[str], List[List[str]]],
        *,
        temperature: Optional[float] = None,
    ) -> Tensor:
        image_features = self.encode_image(images)
        text_features = self.encode_text(texts)
        image_probs = self.similarity(image_features, text_features, temperature=temperature)
        return image_probs

    @property
    def device(self) -> torch.device:
        return self.logit_scale.device


@wraps(_load)
def load(
    name: str,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    jit: bool = False,
    download_root: Optional[str] = None,
) -> Tuple[CLIP, Callable[[Image], Tensor]]:
    if name.startswith("CLIP/"):
        name = name[5:]
    elif os.path.isfile(name):
        pass
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    model, _ = _load(name, device=device, jit=jit, download_root=download_root)  # type: ignore
    model.__class__ = CLIP  # type: ignore
    model = cast(CLIP, model)

    if jit:
        preprocess = image_transform(model.input_resolution.item())  # type: ignore
    else:
        preprocess = image_transform(model.visual.input_resolution)

    return model, preprocess
