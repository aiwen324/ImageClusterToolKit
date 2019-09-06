import numpy as np
import torch
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os, shutil
from sklearn.cluster import KMeans

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=(15,15))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def classify_Image(c_dict):
    for key in c_dict:
        #create a directory
        key_path = os.path.join(os.getcwd(), str(key))
        if not (os.path.isdir(key_path) and os.path.exists(key_path)):
            os.makedirs(str(key))
        for imgs_path in c_dict[key]:
#             print(os.path.basename(imgs_path))
            new_path = os.path.join(key_path, os.path.basename(imgs_path))
#             print(new_path)
            shutil.copyfile(imgs_path, new_path)
            
#               print(imgs_path)

def extract_Feature(model, device, dataloaders):
    output_list = None
    img_tensor = None
    model = model.to(device)
    with torch.no_grad():
        new_iter = iter(dataloaders)
        counter = 0
        for inputs, classes in new_iter:
            inputs_tensor = inputs.to(device)
            output = model(inputs_tensor)
            if device.type == 'cuda':
                output = output.cpu()
            output = output.numpy().reshape((output.shape[0], -1))
            if output_list is None:
                output_list = output
            else:
                output_list = np.vstack((output_list, output))
            print("Finishing one batch: current stacked shape", output_list.shape)
            # if img_tensor is None:
            #     img_tensor = inputs
            # else:
            #     img_tensor = torch.cat((img_tensor, inputs), 0)
            #     img_tensor = img_tensor.reshape([len(output_list), *(inputs.shape[1:])])
    return img_tensor, output_list


# ============================================= Main ============================
if __name__ == '__main__':
    # Define cpu or gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    my_model = models.resnet18(pretrained=True)
    my_model = torch.nn.Sequential(*(list(my_model.children())[:-1]))

    # Define transforms
    data_transforms = transforms.Compose([
        transforms.Resize((480, 480)),
#         transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Define image datasets, will transform to cli format in the future
    image_datasets = datasets.ImageFolder(os.path.join(os.getcwd(), 'dataset'), data_transforms)
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=50,
                                                shuffle=True, num_workers=4)
    
    # Generate classify table
    classify_table = {}

    img_tensor, output_list = extract_Feature(my_model, device, dataloaders)

    # Iteration for KMeans score
    # for i in range(5, output_list.shape[0] // 10):

    kmeans = KMeans(n_clusters=100, random_state=1).fit(output_list)
    print("Get kmeans distance: ", kmeans.inertia_)
    labels = kmeans.labels_
    # print(kmeans.labels_)
    for i in range(len(labels)):
        if labels[i] not in classify_table:
            classify_table[labels[i]] = [image_datasets.imgs[i][0]]
        else:
            classify_table[labels[i]].append(image_datasets.imgs[i][0])
    print(classify_table)
    print(kmeans.inertia_)

    # # IO operation for copying images to other folder
    classify_Image(classify_table)
    
    # # Show some sample image
    # out = torchvision.utils.make_grid(img_tensor.cpu())
    # imshow(out)