from mtm.models.gradient_based.protonet_cnn import ProtonetCNN
import torch


def test_smoke():
    # Smoke test protonet
    m = ProtonetCNN(50)
    # inp = torch.rand(64, 3, 256, 256)
    inp = torch.rand(64, 3, 84, 84)

    out = m(inp)
    print(out.shape)
