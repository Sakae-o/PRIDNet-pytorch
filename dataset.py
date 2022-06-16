import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset as BaseDataset


def normalize_input(data):
    return data/255


class Dataset(BaseDataset):
    def __init__(self, data_dir):
        """
        folder structure:
            - {data_dir}
                - origin
                    1.jpg, ..., n.jpg
                - noise
                    1.jpg, ..., n.jpg
        """

        self.data_dir = data_dir
        origin_dir = os.path.join(data_dir, 'origin')
        noise_dir = os.path.join(data_dir, 'noise')

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f'Folder {data_dir} does not exist')
        if not os.path.exists(origin_dir):
            raise FileNotFoundError(f'Folder {origin_dir} does not exist')
        if not os.path.exists(noise_dir):
            raise FileNotFoundError(f'Folder {noise_dir} does not exist')

        self.image_files = {}
        for opt in ['origin', 'noise']:
            folder = os.path.join(data_dir, opt)
            files = os.listdir(folder)

            self.image_files[opt] = [os.path.join(folder, f) for f in files]


    def __len__(self):
        return len(self.image_files['origin'])

    def __getitem__(self, index):
        origin, noise = self.load_images(index)
        return origin, noise

    def load_images(self, index):
        images = {}
        for opt in ['origin', 'noise']:
            fpath = self.image_files[opt][index]
            if opt == 'origin':
                images[opt] = cv2.imread(fpath, 0)
                images[opt] = images[opt][None, :]
            else:
                images[opt] = cv2.imread(fpath).transpose(2, 0, 1)
            images[opt] = self._transform(images[opt])
        return torch.tensor(images['origin']), torch.tensor(images['noise'])

    def _transform(self, img):
        img = img.astype(np.float32)
        return normalize_input(img)



if __name__ == '__main__':
    from torch.utils.data import DataLoader

    anime_loader = DataLoader(Dataset('D://CV//Denoise//dataset'), batch_size=2, shuffle=True)

    img, img_noise = iter(anime_loader).next()

    print(img.shape)

    cv2.imshow('sad', img[1].numpy().transpose(1, 2, 0))
    cv2.imshow('dq', img_noise[1].numpy().transpose(1, 2, 0))
    cv2.waitKey(0)
