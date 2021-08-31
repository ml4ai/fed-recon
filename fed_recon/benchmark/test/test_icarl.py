from fed_recon.benchmark.client_server import Server
from fed_recon.benchmark.mini_imagenet import MiniImagenet
from fed_recon.models.mcfl.icarl import ICaRL

num_clients = 10
images_path = "/home/user/data/miniImageNet/images"
environment = MiniImagenet(images_path)
n_classes = len(environment.base_labels)
budget = 2000
model = ICaRL(n_classes, budget)

server = Server(num_clients, environment, model)

# dataset, class_list = environment.get_mission_train_data()
for i in range(1):
    server.send_mission(i)
