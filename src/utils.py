import os
import copy
import matplotlib.pyplot as plt
import torch
from datetime import datetime
from tqdm import tqdm
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
from sampling import sample_iid
from sklearn import datasets, preprocessing
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import colorsys

# for loading/processing the images
from keras.preprocessing.image import load_img
from keras.applications.vgg16 import preprocess_input

# models
from keras.applications.vgg16 import VGG16
from keras.models import Model


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    dir = os.path.dirname(__file__)

    if args.dataset == 'iris':
        data = datasets.load_iris(as_frame=True).frame
        train_dataset_labels = data['target']
        data = data.drop(columns='target')
        data = np.array(data)
        scaler = preprocessing.StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data)
        train_dataset = torch.Tensor(data)

        user_groups = None

    elif args.dataset == 'blob':
        seed = None
        if args.seed:
            seed = int(args.seed)
        data, labels = datasets.make_blobs(
            n_samples=args.samples,
            n_features=args.features,
            centers=args.components,
            random_state=seed
        )
        scaler = preprocessing.StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data)
        train_dataset = torch.Tensor(data)

        lb = preprocessing.LabelBinarizer()
        lb.fit(labels)
        labels = lb.transform(labels)
        train_dataset_labels = labels

        #IID
        user_groups = None
        if hasattr(args, 'K'): user_groups = sample_iid(train_dataset, args.K)

    elif args.dataset == 'custom':
        if args.path:
            dataset_name = os.path.basename(os.path.normpath(args.path))
            n_features = 210
            if n_features is None:
                features_filename = '{}_features_{}.pt'.format(dataset_name, 'full')
            else:
                features_filename = '{}_features_{}.pt'.format(dataset_name, 'pca' + str(n_features))
            filename = features_filename
            dir_name = os.path.join(dir, args.path, filename)

            loaded_dataset = None
            try:
                loaded_dataset = torch.load(dir_name)
                print("Features loaded.")
            except:
                print("Features not found.")

            if loaded_dataset is None:
                dir_img = 'images'
                dir_name = os.path.join(dir, args.path, dir_img)

                items = []

                with os.scandir(dir_name) as files:
                    # loops through each file in the directory
                    img_formats = ('.png', '.jpg', 'jpeg')
                    for file in files:
                        if file.name.endswith(img_formats):
                            item = os.path.join(dir_name, file.name)
                            items.append(item)

                data = extract_img_features(args, items, n_features)
                train_dataset = torch.Tensor(data)

                filename = features_filename
                dir_name = os.path.join(dir, args.path, filename)
                torch.save(train_dataset, dir_name)
            else:
                train_dataset = loaded_dataset

            try:
                filename = 'labels.csv'
                dir_name = os.path.join(dir, args.path, filename)
                train_dataset_labels = pd.read_csv(dir_name)
                train_dataset_labels = train_dataset_labels['label'].to_numpy()
            except:
                print('Labels not found.')
                train_dataset_labels = None

            user_groups = None

        else:
            print('Error: dataset path not specified!')
            train_dataset = None
            train_dataset_labels = None
            user_groups = None

    # if args.dataset == 'cifar':
    #     path = '../data/cifar'
    #     data_dir = os.path.join(dir, path)
    #     apply_transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #     ])

    #     train_dataset = datasets.CIFAR10(
    #         data_dir, train=True, download=True, transform=apply_transform
    #     )

    #     test_dataset = datasets.CIFAR10(
    #         data_dir, train=False, download=True, transform=apply_transform
    #     )

    #     # sample training data amongst users
    #     if args.iid:
    #         # Sample IID user data from Mnist
    #         user_groups = cifar_iid(train_dataset, args.num_users)
    #     else:
    #         # Sample Non-IID user data from Mnist
    #         if args.unequal:
    #             # Chose uneuqal splits for every user
    #             raise NotImplementedError()
    #         else:
    #             # Chose euqal splits for every user
    #             user_groups = cifar_noniid(train_dataset, args.num_users)

    #  elif args.dataset == 'mnist' or 'fmnist':
    #     if args.dataset == 'mnist':
    #         path = '../data/mnist'
    #     else:
    #         path = '../data/fmnist'

    #     data_dir = os.path.join(dir, path)

    #     apply_transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.1307,), (0.3081,))])

    #     train_dataset = datasets.MNIST(
    #         data_dir, train=True, download=True, transform=apply_transform
    #     )

    #     test_dataset = datasets.MNIST(
    #         data_dir, train=False, download=True, transform=apply_transform
    #     )

    #     # sample training data amongst users
    #     if args.iid:
    #         # Sample IID user data from Mnist
    #         user_groups = mnist_iid(train_dataset, args.num_users)
    #     else:
    #         # Sample Non-IID user data from Mnist
    #         if args.unequal:
    #             # Chose uneuqal splits for every user
    #             user_groups = mnist_noniid_unequal(
    #                 train_dataset, args.num_users)
    #         else:
    #             # Chose euqal splits for every user
    #             user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, train_dataset_labels, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return

def plot_PCA(ax, X, labels, pca_components=2, soft_clustering=True, title=None, random_state=None):  
    if X.shape[1] > 1:  
        if pca_components > 2: pca_components = 3
        else: pca_components = 2

        if random_state:
            random_state = int(random_state)
        pca = PCA(n_components=pca_components, random_state=random_state)
        pca.fit(X)
        pca_data = pca.transform(X)

        pc1 = pca_data[:, 0]
        pc2 = pca_data[:, 1]
        if pca_components == 3: pc3 = pca_data[:, 2]

        if not bool(soft_clustering):
            idxs = np.argmax(labels, axis=1)
            labels = np.zeros(labels.shape)
            for i in range(labels.shape[0]):
                labels[i, idxs[i]] = 1

        N = labels.shape[1]  
        HSV_tuples = [(i*1.0/N, 0.75, 0.75) for i in range(N)]
        RGB_tuples = np.array(list(map(lambda i: colorsys.hsv_to_rgb(*i), HSV_tuples)))
        colors = np.matmul(labels, RGB_tuples)

        if pca_components == 2:
            ax.set_aspect(1)
            ax.scatter(
                pc1,
                pc2,
                s=0.3,
                c=colors
            )
        else:
            ax.scatter(
                pc1,
                pc2,
                pc3,
                s=0.3,
                c=colors
            )

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        if pca_components == 3: ax.set_zlabel('PC3')

        if title:
            ax.set_title(title)
    else:
        print('Data have only 1 feature. PCA cannot be applied.')

    return

def prepare_output_dir():
    dir = os.path.dirname(__file__)
    path = '../output'
    dir_name = os.path.join(dir, path)
    os.makedirs(dir_name, exist_ok=True)

    date = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    output_dir = os.path.join(dir, path, date)
    os.makedirs(output_dir) 

    return output_dir

def show_img_clusters(args, labels, n_samples=None):
    if len(labels.shape) > 1:
        labels = np.argmax(labels, 1)

    dir = os.path.dirname(__file__)
    dir_img = 'images'
    dir_name = os.path.join(dir, args.path, dir_img)

    filenames = []

    with os.scandir(dir_name) as files:
        # loops through each file in the directory
        img_formats = ('.png', '.jpg', 'jpeg')
        for file in files:
            if file.name.endswith(img_formats):
                filenames.append(file.name)

    groups = {}
    for file, label in zip(filenames, labels):
        if label not in groups.keys():
            groups[label] = []
            groups[label].append(file)
        else:
            groups[label].append(file)

    unique_labels = list(set(labels))
    for cluster in unique_labels:
        plt.figure(figsize=(25, 25))
        files = groups[cluster]
    # only allow up to 30 images to be shown at a time
        if n_samples is not None and len(files) > n_samples:
            print(f"Clipping cluster size from {len(files)} to {n_samples}")
            files = files[:n_samples-1]
    # plot each image in the cluster
        for index, file in enumerate(files):
            plt.subplot(10, 10, index+1)
            dir_name = os.path.join(dir, args.path, dir_img, file)
            img = load_img(dir_name)
            img = np.array(img)
            plt.imshow(img)
            plt.axis('off')

        path = '../save/img'
        filename = 'cluster_{}.png'.format(cluster)
        dir_name = os.path.join(dir, path, filename)
        plt.savefig(dir_name, dpi=100)


def extract_img_features(args, files, n_features=None):
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    data = {}
    pbar = tqdm(files)
    pbar.set_description('Extracting features from images')
    for file in pbar:
        img = load_img(file, target_size=(224, 224))
        img = np.array(img)
        img = img.reshape(1, 224, 224, 3)
        img = preprocess_input(img)
        features = model.predict(img, use_multiprocessing=True)
        data[file] = features

    # filenames = np.array(list(data.keys()))
    features = np.array(list(data.values()))
    features = features.reshape(-1, 4096)

    if n_features:
        seed = None
        if args.seed:
            seed = int(args.seed)
        pca = PCA(n_components=n_features, random_state=seed)
        pca.fit(features)
        features = pca.transform(features)

    return features
