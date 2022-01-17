from bisenetv2 import BiSeNetV2
from easydict import EasyDict

def create_bisenetv2(n_classes, aux_mode='pred'):
  return BiSeNetV2(n_classes, aux_mode)


model_factory = {
  'bisenetv2' : create_bisenetv2
}


def set_cfg_from_file(f_cfg) -> EasyDict:
  cfg = EasyDict()
  cfg.use_sync_bn = False
  cfg.model_type='bisenetv2'
  cfg.n_cats = 13 # 由 onnx 文件得知
  return cfg
