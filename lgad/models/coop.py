from typing import Literal, Optional, Sequence, cast

import torch
from torch import IntTensor
from torch import nn

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer


_tokenizer = _Tokenizer()


def prompt_constructor(
    class_token_position: Literal["front", "middle", "end"],
    n_cls: int,
    n_ctx: int,
    name_lens: Sequence[int],
):

    def _front(prefix, ctx, suffix):
        prompts = []
        for i in range(n_cls):
            name_len = name_lens[i]
            prefix_i = prefix[i : i + 1, :, :]
            class_i = suffix[i : i + 1, :name_len, :]
            suffix_i = suffix[i : i + 1, name_len:, :]
            ctx_i = ctx[i : i + 1, :, :]
            prompt = torch.cat(
                [
                    prefix_i,  # (1, 1, dim)
                    class_i,   # (1, name_len, dim)
                    ctx_i,     # (1, n_ctx, dim)
                    suffix_i,  # (1, *, dim)
                ],
                dim=1,
            )
            prompts.append(prompt)
        return torch.cat(prompts, dim=0)

    def _middle(prefix, ctx, suffix):
        half_n_ctx = n_ctx // 2
        prompts = []
        for i in range(n_cls):
            name_len = name_lens[i]
            prefix_i = prefix[i : i + 1, :, :]
            class_i = suffix[i : i + 1, :name_len, :]
            suffix_i = suffix[i : i + 1, name_len:, :]
            ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
            ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
            prompt = torch.cat(
                [
                    prefix_i,     # (1, 1, dim)
                    ctx_i_half1,  # (1, n_ctx//2, dim)
                    class_i,      # (1, name_len, dim)
                    ctx_i_half2,  # (1, n_ctx//2, dim)
                    suffix_i,     # (1, *, dim)
                ],
                dim=1,
            )
            prompts.append(prompt)
        return torch.cat(prompts, dim=0)

    def _end(prefix, ctx, suffix):
        return torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

    if class_token_position == "front":
        return _front
    elif class_token_position == "middle":
        return _middle
    elif class_token_position == "end":
        return _end
    else:
        raise ValueError(f"Invalid class_token_position: {class_token_position}")


class PromptLearner(nn.Module):
    def __init__(
        self,
        clip_model,
        classnames: Sequence[str],
        n_ctx: Optional[int] = None,
        ctx_init: Optional[str] = None,
        csc: bool = False,
        class_token_position: Literal["front", "middle", "end"] = "front",
    ):
        super().__init__()

        assert n_ctx is not None or ctx_init is not None, "Either n_ctx or ctx_init must be given"

        n_cls = len(classnames)
        dtype = clip_model.dtype
        device = clip_model.device

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)

            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)

            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            n_ctx = cast(int, n_ctx)
            ctx_dim = clip_model.ln_final.weight.shape[0]

            if csc:
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)

            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        ctx_vectors = ctx_vectors
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        with torch.no_grad():
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
            embedding = clip_model.token_embedding(tokenized_prompts.to(device))

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS
        self.register_buffer("_eot_idxs", tokenized_prompts.argmax(dim=-1))

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.name_lens = name_lens

        self.prompt_constructor = prompt_constructor(
            class_token_position=class_token_position,
            n_cls=n_cls,
            n_ctx=n_ctx,
            name_lens=name_lens,
        )

        self.to(device)

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompts = self.prompt_constructor(self.token_prefix, ctx, self.token_suffix)
        return prompts

    @property
    def prompt(self):
        return self()

    @property
    def eot_idxs(self) -> IntTensor:
        return cast(IntTensor, self._eot_idxs)
