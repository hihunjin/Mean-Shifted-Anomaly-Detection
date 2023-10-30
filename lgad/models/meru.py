# Code adapted from: https://github.com/facebookresearch/meru, https://github.com/openai/CLIP

# TODO: Remove code adapted from "facebookresearch/meru" or dual license only this file under CC BY-NC 4.0.
#       "facebookresearch/meru" is licensed under CC BY-NC 4.0, which has some restrictions.
#       But "openai/CLIP" is licensed under MIT License, which is more permissive.
#       And much of the code is adapted from "openai/CLIP".
#       Following code is adapted from "facebookresearch/meru":
#       - _Mixin.tokenize
#       - _Mixin._encode_text
#       - CLIP.encode_text
#       - MERU.encode_text
#       - MERU.forward


try:
    import meru.models
except ImportError:
    raise ImportError(
        "Please install MERU to use this implementation: "
        "`pip install git+https://github.com/facebookresearch/meru.git`"
    )

import os
import hashlib
import urllib
import warnings
from typing import Tuple, Union, Optional, List, cast, Dict, Callable

import torch
from torch import Tensor, IntTensor
from torch.types import _dtype
from torch.nn import functional as F

from tqdm import tqdm
from PIL.Image import Image

import meru.lorentz as L
from meru.models import CLIPBaseline as _CLIP
from meru.models import MERU as _MERU
from meru.tokenizer import Tokenizer
from meru.encoders.image_encoders import build_timm_vit
from meru.encoders.text_encoders import TransformerTextEncoder

from .base import VisionLanguageModel, image_transform


__all__ = [
    "load",
    "available_models",
    "CLIP",
    "MERU",
]

_MODELS = {
    "CLIP/ViT-S/16": (
        "https://dl.fbaipublicfiles.com/meru/clip_vit_s.pth",
        "f3daebf0fddf1c4159d062888faf9e840be0c5646875f85c7056cf8f2c76dbe7",
    ),
    "CLIP/ViT-B/16": (
        "https://dl.fbaipublicfiles.com/meru/clip_vit_b.pth",
        "87e4566219cbedc2c95a31940a16a9c76370cdd9d71d0bad8945fa11990b41b4",
    ),
    "CLIP/ViT-L/16": (
        "https://dl.fbaipublicfiles.com/meru/clip_vit_l.pth",
        "6a691b352ab8c4db85ea0722108a445ff57cb6b3db84842fb3dca1faefbf75f1",
    ),
    "MERU/ViT-S/16": (
        "https://dl.fbaipublicfiles.com/meru/meru_vit_s.pth",
        "4466c04a790d9b5974928e7e9e528cbff2d82348f596e60443a88934b9a61de6",
    ),
    "MERU/ViT-B/16": (
        "https://dl.fbaipublicfiles.com/meru/meru_vit_b.pth",
        "b37dfc515eb3753a8da59ff16231c7c3d89b494bca0345ab48d296235c939651",
    ),
    "MERU/ViT-L/16": (
        "https://dl.fbaipublicfiles.com/meru/meru_vit_l.pth",
        "5abc7254e2a6e81c13367a4912a32e08f09b05184a1fc31e178b8948f7e598b9",
    ),
}
_IMAGE_SIZE = 224
_IMAGE_MEAN = (0.485, 0.456, 0.406)
_IMAGE_STD = (0.229, 0.224, 0.225)
_BASE_VISUAL_CONFIG = {
    "global_pool": "token",
    "use_sincos2d_pos": True,
}
_BASE_TEXTUAL_CONFIG = {
    "arch": "L12_W512",
    "vocab_size": 49408,
    "context_length": 77,
}
_BASE_CLIP_CONFIG = {
    "embed_dim": 512,
    "pixel_mean": _IMAGE_MEAN,
    "pixel_std": _IMAGE_STD,
}
_BASE_MERU_CONFIG = {
    "embed_dim": 512,
    "curv_init": 1.0,
    "learn_curv": True,
    "entail_weight": 0.2,
    "pixel_mean": _IMAGE_MEAN,
    "pixel_std": _IMAGE_STD,
}
_CONFIGS = {
    "CLIP/ViT-S/16": {
        "arch": "clip",
        "visual": {"arch": "vit_small_mocov3_patch16_224", **_BASE_VISUAL_CONFIG},
        "textual": _BASE_TEXTUAL_CONFIG,
        "kwargs": _BASE_CLIP_CONFIG,
    },
    "CLIP/ViT-B/16": {
        "arch": "clip",
        "visual": {"arch": "vit_base_patch16_224", **_BASE_VISUAL_CONFIG},
        "textual": _BASE_TEXTUAL_CONFIG,
        "kwargs": _BASE_CLIP_CONFIG,
    },
    "CLIP/ViT-L/16": {
        "arch": "clip",
        "visual": {"arch": "vit_large_patch16_224", **_BASE_VISUAL_CONFIG},
        "textual": _BASE_TEXTUAL_CONFIG,
        "kwargs": _BASE_CLIP_CONFIG,
    },
    "MERU/ViT-S/16": {
        "arch": "meru",
        "visual": {"arch": "vit_small_mocov3_patch16_224", **_BASE_VISUAL_CONFIG},
        "textual": _BASE_TEXTUAL_CONFIG,
        "kwargs": _BASE_MERU_CONFIG,
    },
    "MERU/ViT-B/16": {
        "arch": "meru",
        "visual": {"arch": "vit_base_patch16_224", **_BASE_VISUAL_CONFIG},
        "textual": _BASE_TEXTUAL_CONFIG,
        "kwargs": _BASE_MERU_CONFIG,
    },
    "MERU/ViT-L/16": {
        "arch": "meru",
        "visual": {"arch": "vit_large_patch16_224", **_BASE_VISUAL_CONFIG},
        "textual": _BASE_TEXTUAL_CONFIG,
        "kwargs": _BASE_MERU_CONFIG,
    },
}

# NOTE: Due to MERU's implementation
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")
warnings.filterwarnings("ignore", category=UserWarning, module="torch/nn/modules/activation")

_tokenizer = Tokenizer()


def _download(url: str, root: str, sha256: Optional[str] = None) -> str:
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target) and sha256 is not None:
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:  # type: ignore
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if sha256 and hashlib.sha256(open(download_target, "rb").read()).hexdigest() != sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def available_models() -> List[str]:
    """Returns the names of available CLIP and MERU models"""
    return list(_MODELS.keys())


class _Mixin(_CLIP):

    def tokenize(self, texts: Union[str, List[str]]) -> IntTensor:
        tokens = _tokenizer(texts)

        # Suppress type error. IntTensor is a subclass of Tensor.
        tokens = cast(List[Tensor], tokens)

        # Truncate tokens that are longer than context_length:
        for idx, inst_tokens in enumerate(tokens):
            if len(inst_tokens) > self.textual.context_length:
                eot_token = inst_tokens[-1]
                inst_tokens = inst_tokens[: self.textual.context_length]
                inst_tokens[-1] = eot_token
                tokens[idx] = inst_tokens

        # Pad all tokens on the right.
        tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
        tokens = cast(IntTensor, tokens)
        return tokens

    def encode_image(
        self,
        images: Tensor,
        *,
        project: bool = False,
    ) -> Tensor:
        images = images.to(self.device)
        image_features = super().encode_image(images, project=project)
        return image_features

    def _ensemble(
        self,
        features: Tensor,
        project: bool = False,
    ) -> Tensor:
        raise NotImplementedError

    def _text_projection(self, text_features: Tensor) -> Tensor:
        raise NotImplementedError

    def _encode_text(self, texts: IntTensor) -> Tensor:
        text_features = self.textual(texts.to(self.device))

        # Get features for [EOS] position and apply projection. `[EOS]` token ID
        # is the largest number in the vocabulary of tokenizer.
        _eos_indices = texts.argmax(dim=-1)
        batch_idxs = torch.arange(text_features.size(0))
        text_features = text_features[batch_idxs, _eos_indices]
        text_features = self.textual_proj(text_features)
        return text_features

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
            text_features = torch.stack([
                self._ensemble(self._encode_text(t), project=project)
                for t in texts
            ], dim=0)
        else:  # IntTensor
            text_features = self._encode_text(texts)

        if project:
            text_features = self._text_projection(text_features)

        return text_features

    def similarity(
        self,
        image_features: Tensor,
        text_features: Tensor,
        *,
        project: bool = True,
        temperature: Optional[float] = None,
    ) -> Tensor:
        raise NotImplementedError

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
    def dtype(self) -> _dtype:
        return self.visual.patch_embed.proj.weight.dtype  # type: ignore


class CLIP(_Mixin, _CLIP, VisionLanguageModel):

    def _ensemble(
        self,
        features: Tensor,
        project: bool = False,
    ) -> Tensor:
        if project:
            features = F.normalize(features, dim=-1)
        features = features.mean(dim=0)
        return features

    def _text_projection(self, text_features: Tensor) -> Tensor:
        text_features = F.normalize(text_features, dim=-1)
        return text_features

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


class MERU(_Mixin, _MERU, VisionLanguageModel):

    def _ensemble(
        self,
        features: Tensor,
        project: bool = False,
    ) -> Tensor:
        features = features.mean(dim=0)
        return features

    def _projection(
        self,
        features: Tensor,
        alpha: Tensor,
    ) -> Tensor:
        features = features * alpha.exp()
        with torch.autocast(self.device.type, dtype=torch.float32):  # type: ignore
            features = L.exp_map0(features, self.curv.exp())
        return features

    def _text_projection(self, text_features: Tensor) -> Tensor:
        return self._projection(text_features, self.textual_alpha)

    def similarity(
        self,
        image_features: Tensor,
        text_features: Tensor,
        *,
        project: bool = True,
        temperature: Optional[float] = None,
    ) -> Tensor:
        if project:
            image_features = self._projection(image_features, self.visual_alpha)
            text_features = self._projection(text_features, self.textual_alpha)

        image_features = image_features.type(self.dtype)
        text_features = text_features.type(self.dtype)

        with torch.autocast(self.device.type, dtype=torch.float32):  # type: ignore
            image_logits = -L.pairwise_dist(image_features, text_features, self.curv.exp())
            # text_logits = -L.pairwise_dist(text_features, image_features, self.curv.exp())

        logit_scale = self.logit_scale.exp() if temperature is None else (1 / temperature)
        image_probs = (logit_scale * image_logits).softmax(dim=-1)
        return image_probs


def build_model(config: Dict) -> Union[CLIP, MERU]:
    if config["arch"] == "clip":
        model_class = CLIP
    elif config["arch"] == "meru":
        model_class = MERU
    else:
        raise ValueError(f"Unknown model architecture: {config['arch']}")

    model = model_class(
        visual=build_timm_vit(**config["visual"]),
        textual=TransformerTextEncoder(**config["textual"]),
        **config["kwargs"],
    )
    return model


def load(
    name: str,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    jit: bool = False,
    download_root: Optional[str] = None,
) -> Tuple[Union[CLIP, MERU], Callable[[Image], Tensor]]:
    """Load a CLIP or MERU model

    Parameters
    ----------
    name: str
        A model name listed by `available_models()`.

    device: Union[str, torch.device]
        The device to put the loaded model.

    jit: bool
        Whether to load the optimized JIT model or more hackable non-JIT model.
        But this option is not supported for this implementation.

    download_root: str
        path to download the model files; by default, it uses "~/.cache/meru".

    Returns
    -------
    model: Union[CLIP, MERU]
        The CLIP or MERU model.

    preprocess : Callable[[PIL.Image], Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input.
    """
    if jit:
        raise NotImplementedError("JIT loading is not supported for this implementation.")

    if name in _MODELS:
        model_path = _download(
            _MODELS[name][0],
            root=download_root or os.path.expanduser("~/.cache/meru"),
            sha256=_MODELS[name][1],
        )
    elif os.path.isfile(name):
        raise NotImplementedError("Loading from a file is not supported for this implementation.")
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    state_dict = torch.load(model_path, map_location="cpu")
    model = build_model(_CONFIGS[name])
    model.load_state_dict(state_dict["model"])
    model.eval()
    model.to(device)

    preprocess = image_transform(_IMAGE_SIZE)

    return model, preprocess
