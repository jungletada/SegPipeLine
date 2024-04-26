# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .modeling.efficient_sam import build_efficient_sam


def build_efficient_sam_vit_ti(checkpoint="weights/efficient_sam_vitt.pt"):
    return build_efficient_sam(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        checkpoint=checkpoint,
    ).eval()


def build_efficient_sam_vit_s(checkpoint="weights/efficient_sam_vits.pt",):
    return build_efficient_sam(
        encoder_patch_embed_dim=384,
        encoder_num_heads=6,
        checkpoint=checkpoint,
    ).eval()


efficient_sam_model_registry = {
    "vit_ti": build_efficient_sam_vit_ti,
    "vit_s": build_efficient_sam_vit_s,
}
