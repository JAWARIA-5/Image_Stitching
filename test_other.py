import argparse
import torch
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
import cv2
from network import get_stitched_result, Network, build_new_ft_model
import glob
import torchvision.transforms as T

resize_512 = T.Resize((512, 512))

# Loss functions integrated here
def l_num_loss(img1, img2, l_num=1):
    return torch.mean(torch.abs((img1 - img2) ** l_num))

def cal_lp_loss2(input1, warp_mesh, warp_mesh_mask):
    batch_size, _, img_h, img_w = input1.size()
    delta3 = (torch.sum(warp_mesh, [2, 3]) - torch.sum(input1 * warp_mesh_mask, [2, 3])) / torch.sum(warp_mesh_mask, [2, 3])
    input1_newbalance = input1 + delta3.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)
    lp_loss_2 = l_num_loss(input1_newbalance * warp_mesh_mask, warp_mesh, 1)
    lp_loss = 1.0 * lp_loss_2
    return lp_loss

def loadSingleData(data_path, img1_name, img2_name):
    # Load image1
    input1 = cv2.imread(data_path + img1_name)
    input1 = input1.astype(dtype=np.float32)
    input1 = (input1 / 127.5) - 1.0
    input1 = np.transpose(input1, [2, 0, 1])

    # Load image2
    input2 = cv2.imread(data_path + img2_name)
    input2 = input2.astype(dtype=np.float32)
    input2 = (input2 / 127.5) - 1.0
    input2 = np.transpose(input2, [2, 0, 1])

    # Convert to tensor
    input1_tensor = torch.tensor(input1).unsqueeze(0)
    input2_tensor = torch.tensor(input2).unsqueeze(0)
    return (input1_tensor, input2_tensor)

# Path of project
last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))

# Path to save the model files
MODEL_DIR = os.path.join(last_path, 'model')

# Create folders if it does not exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def train(args):
    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Define the network
    net = Network()
    if torch.cuda.is_available():
        net = net.cuda()

    # Define the optimizer and learning rate
    optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    # Load existing models if it exists
    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
    ckpt_list.sort()
    if len(ckpt_list) != 0:
        model_path = ckpt_list[-1]
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        scheduler.last_epoch = start_epoch
        print(f'Loaded model from {model_path}!')
    else:
        start_epoch = 0
        print('Training from scratch!')

    # Load dataset (only one pair of images)
    input1_tensor, input2_tensor = loadSingleData(data_path=args.path, img1_name=args.img1_name, img2_name=args.img2_name)
    if torch.cuda.is_available():
        input1_tensor = input1_tensor.cuda()
        input2_tensor = input2_tensor.cuda()

    input1_tensor_512 = resize_512(input1_tensor)
    input2_tensor_512 = resize_512(input2_tensor)

    loss_list = []
    print("################## Start Iteration #######################")
    for epoch in range(start_epoch, start_epoch + args.max_iter):
        net.train()
        optimizer.zero_grad()
        batch_out = build_new_ft_model(net, input1_tensor_512, input2_tensor_512)
        warp_mesh = batch_out['warp_mesh']
        warp_mesh_mask = batch_out['warp_mesh_mask']
        rigid_mesh = batch_out['rigid_mesh']
        mesh = batch_out['mesh']

        total_loss = cal_lp_loss2(input1_tensor_512, warp_mesh, warp_mesh_mask)
        total_loss.backward()
        # Clip the gradient
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3, norm_type=2)
        optimizer.step()

        current_iter = epoch - start_epoch + 1
        print(f"Training: Iteration[{current_iter:03}/{args.max_iter:03}] Total Loss: {total_loss:.4f} lr={optimizer.state_dict()['param_groups'][0]['lr']:.8f}")

        loss_list.append(total_loss)

        if current_iter == 1:
            with torch.no_grad():
                output = get_stitched_result(input1_tensor, input2_tensor, rigid_mesh, mesh)
            cv2.imwrite(args.path + 'before_optimization.jpg', output['stitched'][0].cpu().detach().numpy().transpose(1, 2, 0))
            cv2.imwrite(args.path + 'before_optimization_mesh.jpg', output['stitched_mesh'])

        if current_iter >= 4:
            if all(torch.abs(loss_list[i] - loss_list[i + 1]) <= 1e-4 for i in range(current_iter - 4, current_iter - 1)):
                with torch.no_grad():
                    output = get_stitched_result(input1_tensor, input2_tensor, rigid_mesh, mesh)
                path = args.path + f"iter-{epoch - start_epoch + 1:03}.jpg"
                cv2.imwrite(path, output['stitched'][0].cpu().detach().numpy().transpose(1, 2, 0))
                cv2.imwrite(args.path + f"iter-{epoch - start_epoch + 1:03}_mesh.jpg", output['stitched_mesh'])
                break

        if current_iter == args.max_iter:
            with torch.no_grad():
                output = get_stitched_result(input1_tensor, input2_tensor, rigid_mesh, mesh)
            path = args.path + f"iter-{epoch - start_epoch + 1:03}.jpg"
            cv2.imwrite(path, output['stitched'][0].cpu().detach().numpy().transpose(1, 2, 0))
            cv2.imwrite(args.path + f"iter-{epoch - start_epoch + 1:03}_mesh.jpg", output['stitched_mesh'])

        scheduler.step()

    print("################## End Iteration #######################")

if __name__ == "__main__":
    print('<==================== Setting Arguments ===================>\n')
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--max_iter', type=int, default=50)
    parser.add_argument('--path', type=str, default='Inputs/')
    parser.add_argument('--img1_name', type=str, default='input1.jpg')
    parser.add_argument('--img2_name', type=str, default='input2.jpg')
    args = parser.parse_args()
    print(args)
    train(args)

