import torch
import os
import numpy as np
import cv2


def save_checkpoint(model, optimizer, epoch, args):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }
    path = os.path.join(args.checkpoint_dir, f'{model.name}_epoch{epoch}.pth')
    torch.save(checkpoint, path)


def save_samples(model, epoch, args, sample_path):
    model.eval()

    img = cv2.imread(sample_path).transpose(2, 0, 1) / 255

    img = img[None, :]
    img = torch.tensor((img), dtype=torch.float).cuda()

    with torch.no_grad():
        output = model(img.cuda())
        output = output.detach().cpu().numpy()[0].transpose(1, 2, 0)
    output = (output * 255).astype(np.uint8)

    cv2.imwrite(os.path.join(args.save_image_dir, str(epoch) + '.jpg'), output)

