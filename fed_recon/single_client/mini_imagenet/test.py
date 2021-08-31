from fed_recon.data.mini_imagenet.mini_imagenet import MiniImagenet


def test_mi_dataset():
    ds = MiniImagenet(
        root_dir="data/mini-imagenet/mini-imagenet",
        n_classes_per_task=5,
        k_shots_per_class=2,
        meta_split="train",
    )
    for i in range(10):
        mb = ds.sample_meta_batch(batch_size=3, sample_k_value=False)
        print("images:")
        print(mb[0])
        print("labels:")
        print(mb[1])


if __name__ == "__main__":
    test_mi_dataset()
