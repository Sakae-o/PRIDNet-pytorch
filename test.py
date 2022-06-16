import torch
import os
import cv2
import numpy as np
from models.network import PRIDNet



if __name__ == '__main__':
    data_dir = 'test'
    save_dir = 'test_results'

    files = os.listdir(data_dir)

    model = PRIDNet()

    checkpoint = torch.load('checkpoint2/best_model.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    for file in files:
        if (file[-3:] in ['jpg', 'png', 'bmp'] or file[-4:] == 'jpeg'):
            img_dir = os.path.join(data_dir, file)
            img = cv2.imread(img_dir).transpose(2, 0, 1) / 255

            img = img[None, :]
            img = torch.tensor((img), dtype=torch.float)

            output = model(img)
            output = output.detach().numpy()[0].transpose(1, 2, 0)
            output = (output * 255).astype(np.uint8)

            # cv2.imshow('dq', output)
            # cv2.waitKey(0)

            cv2.imwrite(os.path.join(save_dir, file), output)







