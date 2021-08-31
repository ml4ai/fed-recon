import torch


def plot_image_batch(images: torch.Tensor, labels: torch.Tensor):
    import matplotlib.pyplot as plt

    images = images.numpy()
    labels = labels.numpy()

    assert len(images.shape) == 4

    for i, (image, label) in enumerate(zip(images, labels)):
        image = image.transpose(1, 2, 0)
        image *= [0.229, 0.224, 0.225]
        image += [0.485, 0.456, 0.406]
        print(label)
        plt.imshow(image)
        plt.show()
        if i == 9:
            break
