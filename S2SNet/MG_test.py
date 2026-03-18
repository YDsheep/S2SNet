import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from models.S2SNet import S2SNet
from data import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=448, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path',type=str,default='datasets/',help='test dataset path for RS dataset')
opt = parser.parse_args()

dataset_path = opt.test_path

if opt.gpu_id=='0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id=='1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')

#load the model
model = S2SNet()
checkpoint_path = 'save/ORSSD/epoch_best.pth'
state_dict = torch.load(checkpoint_path)
new_state_dict = {k: v for k, v in state_dict.items() if 'total_ops' not in k and 'total_params' not in k}
model.load_state_dict(new_state_dict)

model.cuda()
model.eval()


def save_tensor_as_image(tensor, save_dir, name, n):
    valid_ext = ['.jpg', '.png', '.jpeg', '.bmp']
    if not any(name.lower().endswith(ext) for ext in valid_ext):
        name += '.jpg'

    save_path = os.path.join(save_dir, name)

    numpy_array = tensor.cpu().squeeze(0).permute(1, 2, 0).detach().numpy()
    if numpy_array.dtype == np.float32:
        numpy_array = (numpy_array * 255).clip(0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path+n, numpy_array)

# test
# test_datasets = ['ORSSD', 'EORSSD','4199']
test_datasets = ['ORSSD/test']
for dataset in test_datasets:
    save_path = 'output/ORSSD/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/RGB/'
    gt_root = dataset_path + dataset + '/GT/'
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    for i in range(test_loader.size):
        image, gt, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        s1, s2, s3, s4 = model(image)
        # s1 = model(image)
        res = F.upsample(s1, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)


        print('save img to: ',save_path+name)
        cv2.imwrite(save_path+name,res*255)
    print('Test Done!')
