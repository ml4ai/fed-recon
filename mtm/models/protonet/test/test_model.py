import torch

from mtm.models.protonet.model import FederatedProtoNet


def test_running_prototype_averaging_store_embeddings():
    in_channels = 3
    hidden_size = 64
    embedding_size = 64
    distance_function = "euclidean"

    # Test storing all embeddings in memory
    model = FederatedProtoNet(
        in_channels=in_channels,
        out_channels=embedding_size,
        hidden_size=hidden_size,
        distance_function=distance_function,
        store_all_embeddings_in_memory=True,
    )

    embeddings = torch.rand(300, 1600)

    proto = torch.mean(embeddings, dim=0)

    for i in range(embeddings.shape[0]):
        model._update_embeddings_for_class(0, embeddings[i].unsqueeze(0))

    estimate = model.embeddings[0]
    estimate = estimate.mean(0)

    assert (estimate == proto).numpy().all()


def test_running_prototype_averaging():
    in_channels = 3
    hidden_size = 64
    embedding_size = 64
    distance_function = "euclidean"
    model = FederatedProtoNet(
        in_channels=in_channels,
        out_channels=embedding_size,
        hidden_size=hidden_size,
        distance_function=distance_function,
        store_all_embeddings_in_memory=False,
    )

    embeddings_dtype = torch.float32
    merge_ops_dtyp = torch.float64
    embeddings = torch.rand(15000, 1600, dtype=embeddings_dtype)

    proto = torch.mean(embeddings, dim=0)

    for i in range(embeddings.shape[0]):
        model.update_prototype_for_class(0, embeddings[i], 1)

    estimate = model.prototypes[0]

    if estimate.type != embeddings_dtype:
        estimate = estimate.type(embeddings_dtype)

    print("diff of means:")
    print((estimate.mean() - proto.mean()))
    import pdb

    pdb.set_trace()

    try:
        print("max of diffs")
        print((estimate - proto).max())
    except RuntimeError as e:
        print(e)
    # with 10 updates: tensor(1.1921e-07)
    # with 300 updates: tensor(5.9605e-07)
    # with 15000 updates:
    # with 15000 updates (running sum and final division): tensor(2.5332e-06)
    # TODO:
    #   pooled embeddings
    #   l2 normalized embeddings
    #   train with gaussian noise added to prototypes
    #   proto += epsilon, where epsilon ~ N(0, 0.0001)

    print("sum of diffs:")
    print((estimate - proto).sum())
    # tensor(2.2352e-07)
    # tensor(4.3809e-06)

    print("Norm of diff:")
    print(torch.norm(estimate - proto, 2))
    # 10 updates: tensor(1.6208e-06)
    # 300 updates: tensor(5.6697e-06)
    # with 1500 updates: (numerically stable version): tensor(3.8141e-05), tensor(1.2271e-05)
    # with 15000 updates: (numerically stable version): tensor(3.8885e-05)
    # with 15000 updates: (running sum and final division): tensor(3.3752e-05)

    # assert (estimate == proto).numpy().all()

    assert (estimate == proto).numpy().all()


test_running_prototype_averaging_store_embeddings()

test_running_prototype_averaging()
