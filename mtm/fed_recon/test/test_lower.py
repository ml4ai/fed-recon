from mtm.fed_recon.client_server import Server
from mtm.fed_recon.mini_imagenet import MiniImagenet
from mtm.models.mcfl.lower import Lower

num_clients = 10
images_path = "/home/user/data/miniImageNet/images"
environment = MiniImagenet(images_path)
n_classes = len(environment.base_labels)
model = Lower(n_classes)

server = Server(num_clients, environment, model)

for i in range(1):
    server.send_mission(i)
