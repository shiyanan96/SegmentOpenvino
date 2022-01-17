import argparse
import os.path as osp
import sys
sys.path.insert(0, '.')
import os
import torch

from tools.miscs import model_factory, set_cfg_from_file

torch.set_grad_enabled(False)

root_model = '/home/yangw/sources/SegmentOpenvino/models'
src_prefix = 'model_fixsize_retrain'
dst_prefix = 'model_opset10'
parse = argparse.ArgumentParser()
parse.add_argument('--config', dest='config', type=str,
        default='configs/bisenetv2.py',)
parse.add_argument('--weight_path', dest='weight_pth', type=str,
        default=os.path.join(root_model, src_prefix + '.pth'))
parse.add_argument('--outpath', dest='out_pth', type=str,
        default=os.path.join(root_model, dst_prefix + '.onnx'))
parse.add_argument('--aux_mode', dest='aux_mode', type=str,
        default='pred') # 由 tools.isenetv2.py 得知
#parse.add_argument('--opset', type=int, default=10) # 客户原始设定
parse.add_argument('--opset', type=int, default=10)
args = parse.parse_args()


cfg = set_cfg_from_file(args.config)
if cfg.use_sync_bn: cfg.use_sync_bn = False
net = model_factory[cfg.model_type](cfg.n_cats, aux_mode=args.aux_mode)
net.load_state_dict(torch.load(args.weight_pth, map_location='cpu'), strict=False)
net.eval()


#  dummy_input = torch.randn(1, 3, *cfg.crop_size)
height = 640
width = 352
input_shape = (1, 3, height, width)
dummy_input = torch.randn(input_shape)
input_names = ['input_image']
output_names = ['preds',]
dynamic_axes = {
        'input_image': {2: 'height', 3:'width'},
        'preds': {2: 'height', 3:'width'}
}

torch.onnx.export(
        net, 
        dummy_input, 
        args.out_pth, 
        input_names=input_names, 
        output_names=output_names, 
        verbose=False, 
        opset_version=args.opset
        ) #, dynamic_axes=dynamic_axes)

