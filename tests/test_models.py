from pathlib import Path
import json
import pytest
import torch
from ieee14_fdia.models import build_model


@pytest.mark.parametrize("family", ["LSTM", "P_sLSTM", "xLSTM_Mixer"])
def test_complete_model_forward_backward_and_weight_roundtrip(family):
    torch.set_num_threads(2)
    plans = json.loads((Path(__file__).parents[1] / "configs/models.json").read_text())
    config = plans[family]["refine"]
    m = build_model(family, config)
    x = torch.randn(2, config["window"], 28)
    y = m(x)
    assert y.shape == (2, 27) and torch.isfinite(y).all()
    y.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
    other = build_model(family, config)
    other.load_state_dict(m.state_dict(), strict=True)
    torch.testing.assert_close(m.eval()(x), other.eval()(x))
