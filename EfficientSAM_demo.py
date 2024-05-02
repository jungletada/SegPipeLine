import cv2
import re
import torch
import zipfile
import os.path as osp
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from efficient_sam.build_efficient_sam import efficient_sam_model_registry
from efficient_sam.utils import show_mask, show_points, show_box, \
    save_transparent_image_with_border, save_transparent_img


device = "cuda"
model_type = 'vit_s'

with zipfile.ZipFile("weights/efficient_sam_vits.pt.zip", 'r') as zip_ref:
    zip_ref.extractall("weights")
model = efficient_sam_model_registry[model_type]()

example_img_path = 'figs/examples'
output_img_path = 'figs/outputs'

img_name = 'motor-rotate.png'
img_name_no_ext = re.sub(r'\.[^.]*$', '', img_name)

sample_image_np = cv2.imread(osp.join(example_img_path, img_name))
sample_image_np = cv2.cvtColor(sample_image_np, cv2.COLOR_BGR2RGB)
sample_image_tensor = transforms.ToTensor()(sample_image_np)

    
def convert_bbox_to_points(input_boxes):
    # convert the bboxes into the point prompts
    num_queries = input_boxes.shape[0]
    input_points = torch.from_numpy(input_boxes).unsqueeze(0)    # [bs, num_queries, 4], bs = 1
    input_points = input_points.view(-1, num_queries, 2, 2)     # [bs, num_queries, num_pts, 2]
    input_labels = torch.tensor([2, 3])  # top-left, bottom-right
    input_labels = input_labels[None, None].repeat(1, num_queries, 1) # [bs, num_queries, num_pts]
    
    return input_points, input_labels


def run_points_sample():
    r"""
        掩码已按其预测的IoU进行排序。
        - 第一维是批处理大小
        - 第二维是我们要生成的掩码数量
        - 第三维是模型输出的候选掩码数量
        对于这个演示，我们使用第一个掩码
    """
    plt.cla()
    input_points = torch.tensor([[[[400, 160], [200, 360]]]])
    input_labels = torch.tensor([[[1, 1]]])

    plt.figure(figsize=(10,10))
    plt.imshow(sample_image_np)
    show_points(input_points.numpy(), input_labels.numpy(), plt.gca())
    plt.axis('on')
    plt.savefig(osp.join(output_img_path, f'{img_name_no_ext}_points.png'))
    
    print(f'使用{model_type}进行推理 ')
    
    predicted_logits, predicted_iou = model(
        sample_image_tensor[None, ...],
        input_points,
        input_labels,
    )
    sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
    predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
    predicted_logits = torch.take_along_dim(
        predicted_logits, sorted_ids[..., None, None], dim=2
    )
    mask = torch.ge(predicted_logits[0, 0, 0, :, :], 0).cpu().detach().numpy()
    binary_image = np.where(mask, 255, 0).astype(np.uint8)
    cv2.imwrite(osp.join(output_img_path, 
                         f"{img_name_no_ext}_{model_type}_pts_mask_binary.png"), 
                binary_image)
    # plt.cla()
    # plt.figure(figsize=(10, 10))
    # plt.imshow(sample_image_np)
    # show_mask(mask, plt.gca())
    # show_points(input_points.numpy(), input_labels.numpy(), plt.gca())
    # plt.axis('off')
    # plt.savefig(osp.join(output_img_path, f'{img_name_no_ext}_{model_type}_pts_mask.png'))
    
    # save_transparent_img(sample_image_np, mask, 
    #                     save_path=osp.join(output_img_path, f"{img_name_no_ext}_{model_type}_pts_mask_BGRA.png"))
    
    # save_transparent_image_with_border(sample_image_np, mask, 
    #                               save_path=osp.join(output_img_path, f"{img_name_no_ext}_{model_type}_pts_mask_BGRA_cropped.png"))
    print(f'{model_type}推理完成')
    

def run_bbox_sample():
    # bboxes of the sample
    input_boxes = np.array([[150, 20, 600, 780]])
    # convert the bboxes into the point prompts
    input_points, input_labels = convert_bbox_to_points(input_boxes)
    print(f'使用{model_type}进行推理')
    predicted_logits, predicted_iou = model(
        sample_image_tensor[None, ...],
        input_points,
        input_labels,
    )
    sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
    predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
    # [bs, num_queries, num_candidate_masks, img_h, img_w]
    predicted_logits = torch.take_along_dim(
        predicted_logits, sorted_ids[..., None, None], dim=2
    )
    masks = torch.ge(predicted_logits, 0).cpu().detach().numpy()
    mask = masks[0, :, 0, :, :]  # [bs, num_queries, num_candidate_masks, img_h, img_w] -> [num_queries, img_h, img_w]
    
    plt.cla()
    plt.figure(figsize=(10, 10))
    plt.imshow(sample_image_np)
    
    for i in range(input_boxes.shape[0]):
        show_mask(mask[i], plt.gca())
        show_box(input_boxes[i], plt.gca())
        plt.axis('off')
        plt.savefig(osp.join(output_img_path, f'{img_name_no_ext}_{model_type}_bbox_mask.png'))
        save_transparent_img(sample_image_np, mask[i], 
                            save_path=osp.join(output_img_path, 
                                               f"{img_name_no_ext}_{model_type}_box_mask{i}_BGRA.png"))
        save_transparent_image_with_border(sample_image_np, mask[i], 
                                    save_path=osp.join(output_img_path, 
                                                       f"{img_name_no_ext}_{model_type}_box_mask{i}_BGRA_cropped.png"))
    print(f'{model_type}推理完成')
    
    
if __name__ == '__main__':

    run_points_sample()
    # run_bbox_sample()