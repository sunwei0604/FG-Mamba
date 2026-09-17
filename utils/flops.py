

from dataclasses import dataclass


@dataclass
class ProfileResult:
    params: int
    flops: str
    status: str


def _human_count(value: float) -> str:
    for unit in ("", "K", "M", "G", "T"):
        if abs(value) < 1000.0:
            return f"{value:.3f}{unit}"
        value /= 1000.0
    return f"{value:.3f}P"


def _remove_thop_buffers(model) -> None:
    for module in model.modules():
        module._buffers.pop("total_ops", None)
        module._buffers.pop("total_params", None)


def profile_model(model, inputs) -> ProfileResult:
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    was_training = model.training
    model.eval()
    errors = []
    try:
        try:
            from thop import profile
            flops, _ = profile(model, inputs=inputs, verbose=False)
            return ProfileResult(params, _human_count(float(flops)), "THOP")
        except Exception as exc:
            errors.append(f"THOP: {type(exc).__name__}: {exc}")
        try:
            from fvcore.nn import FlopCountAnalysis
            flops = FlopCountAnalysis(model, inputs).total()
            return ProfileResult(params, _human_count(float(flops)), "fvcore")
        except Exception as exc:
            errors.append(f"fvcore: {type(exc).__name__}: {exc}")
        return ProfileResult(params, "NA", " | ".join(errors))
    finally:
        _remove_thop_buffers(model)
        model.train(was_training)
