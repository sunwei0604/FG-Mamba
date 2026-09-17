"""Construct the full FG-Mamba model described in Section III."""

from .fg_mamba import FG_Mamba


def create_model(args, num_classes, num_bands):
    if args.model != 'fg_mamba':
        raise ValueError(f"Unsupported model: {args.model}")
    return FG_Mamba(
        hsi_channels=num_bands,
        num_classes=num_classes,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
    )
