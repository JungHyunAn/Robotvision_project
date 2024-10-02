import torch
import torchvision
import mmcv
import mmengine
model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
pred_depth, confidence, output_dict = model.inference({'input': rgb})
pred_normal = output_dict['prediction_normal'][:, :3, :, :] # only available for Metric3Dv2 i.e., ViT models
normal_confidence = output_dict['prediction_normal'][:, 3, :, :] # see https://arxiv.org/abs/2109.09881 for details
