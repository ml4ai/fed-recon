from mtm.fed_recon.mini_imagenet import MiniImagenet
from torch.utils.data.dataloader import DataLoader

images_path = "/home/user/data/miniImageNet/images"

hold_out_bool = True
base_val_bool = True

miniImagenet = MiniImagenet(
    images_path, hold_out_bool=hold_out_bool, base_val_bool=base_val_bool
)

print(f"base_classes: {miniImagenet.base_classes}")
print(f"base_labels: {miniImagenet.base_labels}")
if hold_out_bool:
    print(f"hold_out_classes: {miniImagenet.hold_out_classes}")
    print(f"hold_out_labels: {miniImagenet.hold_out_labels}")
print(f"field_classes: {miniImagenet.field_classes}")
print(f"field_labels: {miniImagenet.field_labels}")

class_id = "n03476684"
class_name = miniImagenet.get_class_name_from_id(class_id)
return_id = miniImagenet.get_class_id_from_name(class_name)
assert class_id == return_id

data, labels, labels_list = miniImagenet.get_mission_data_train_tensor()
dataset = miniImagenet.get_mission_data_test_tensor(labels_list)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

(
    base_train,
    base_val,
    base_test,
    hold_out_train,
    hold_out_val,
    field_train,
    field_test,
) = miniImagenet.get_datasets()

base_train_loader = DataLoader(base_train, batch_size=4, shuffle=True)

if base_val_bool:
    base_val_loader = DataLoader(base_val, batch_size=4, shuffle=True)

base_test_loader = DataLoader(base_test, batch_size=4, shuffle=True)

if hold_out_bool:
    hold_out_train_loader = DataLoader(hold_out_train, batch_size=4, shuffle=True)
    hold_out_val_loader = DataLoader(hold_out_val, batch_size=4, shuffle=True)

field_train_loader = DataLoader(field_train, batch_size=4, shuffle=True)
field_test_loader = DataLoader(field_test, batch_size=4, shuffle=True)

for inputs, labels in base_train_loader:
    print("BaseTrain")
    print(inputs.shape)
    print(labels)
    break

if base_val_bool:
    for inputs, labels in base_val_loader:
        print("BaseVal")
        print(inputs.shape)
        print(labels)
        break

for inputs, labels in base_test_loader:
    print("BaseTest")
    print(inputs.shape)
    print(labels)
    break

if hold_out_bool:
    for inputs, labels in hold_out_train_loader:
        print("HoldoutTrain")
        print(inputs.shape)
        print(labels)
        break

    for inputs, labels in hold_out_val_loader:
        print("HoldoutVal")
        print(inputs.shape)
        print(labels)
        break

for inputs, labels in field_train_loader:
    print("FieldTrain")
    print(inputs.shape)
    print(labels)
    break

for inputs, labels in field_test_loader:
    print("FieldTest")
    print(inputs.shape)
    print(labels)
    break
