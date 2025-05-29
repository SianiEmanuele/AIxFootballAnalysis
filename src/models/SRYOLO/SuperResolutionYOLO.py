import os
import glob
import cv2
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import numpy as np
from torch import nn
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from realesrgan import RealESRGANer
from ultralytics import YOLO



class SRYOLO(nn.Module):
    """
    Combines Real-ESRGAN super-resolution with YOLOv9c, applying SR on-the-fly
    during predict and val, maintaining original aspect ratio.

    Args:
        yolo_weights (str): Path or name of YOLO model weights/config.
        scale (int): Upscaling factor for Real-ESRGAN.
        model_path (str): Path to the ESRGAN generator .pth file.
        dni_weight (float): DNI weight.
        tile (int): Tile size for tiled inference.
        tile_pad (int): Tile padding.
        pre_pad (int): Pre-padding.
        max_size (int, optional): Maximum size for longer edge, preserving aspect ratio.
        device (str): Torch device identifier.
    """
    def __init__(
        self,
        yolo_weights: str,
        scale: int,
        model_path: str,
        dni_weight: float,
        tile: int,
        tile_pad: int,
        pre_pad: int,
        max_size: int = 640,
        device: str = 'cuda:0'
    ):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.max_size = max_size
        # Initialize Real-ESRGAN
        arch = SRVGGNetCompact(3, 3, 64, 32, scale, 'prelu')
        self.upsampler = RealESRGANer(
            scale=scale,
            model_path=model_path,
            dni_weight=dni_weight,
            model=arch,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            half=True,
            gpu_id=0 if 'cuda' in device else -1
        )
        # Initialize YOLOv9c
        self.yolo = YOLO(yolo_weights)
        self.yolo.model.to(self.device)
        self.stride = int(self.yolo.model.stride.max())

    def _load_and_preprocess(self, source):
        """
        Load images from path/folder, apply SR and resizing, return list of numpy HWC images.
        """
        paths = []
        if os.path.isdir(source):
            for ext in ('*.jpg', '*.png', '*.jpeg'):
                paths.extend(sorted(glob.glob(os.path.join(source, ext))))
        elif os.path.isfile(source):
            paths = [source]
        else:
            raise ValueError(f"Invalid source: {source}")

        processed = []
        for p in paths:
            img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)

            try:
                sr, _ = self.upsampler.enhance(img, outscale=1)
            except RuntimeError as e:
                print(f"OOM during SR for {p}: {e}")
                torch.cuda.empty_cache()
                continue

            torch.cuda.empty_cache()

            # Resize preserving aspect ratio
            if self.max_size:
                h0, w0 = sr.shape[:2]
                scale_ratio = self.max_size / max(h0, w0)
                new_w, new_h = int(w0 * scale_ratio), int(h0 * scale_ratio)
                sr = cv2.resize(sr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Pad to stride
            h, w = sr.shape[:2]
            ph, pw = (-h) % self.stride, (-w) % self.stride
            top, bottom = divmod(ph, 2)
            left, right = divmod(pw, 2)
            sr = np.pad(sr, ((top, bottom), (left, right), (0, 0)), mode='constant', constant_values=114)

            processed.append(sr)

            del img, sr  # Cleanup
            torch.cuda.empty_cache()

        return processed

    def predict(self, source=None, imgs=None, **kwargs):
        """
        Run inference with SR preprocessing.

        Args:
            source (str): Path to image or directory.
            imgs (List[np.ndarray]): List of HWC numpy images.
        Returns:
            List of ultralytics Results objects.
        """
        if source:
            hwc_imgs = self._load_and_preprocess(source)
        elif imgs is not None:
            hwc_imgs = []
            for img in imgs:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 and img.shape[2] == 3 else img
                try:
                    sr, _ = self.upsampler.enhance(img, outscale=1)
                except RuntimeError as e:
                    print(f"OOM during SR: {e}")
                    torch.cuda.empty_cache()
                    continue
                hwc_imgs.append(sr)
                del img, sr
                torch.cuda.empty_cache()
        else:
            raise ValueError("predict requires `source` or `imgs`")

        results = self.yolo.predict(source=hwc_imgs, half=True, **kwargs)
        return results

    def val(self, source=None, imgs=None, **kwargs):
        """
        Validation on SR images (alias to predict).
        """
        return self.predict(source=source, imgs=imgs, **kwargs)

# Example usage
if __name__ == '__main__':
    sr_yolo = SRYOLO(
        yolo_weights='yolov9c.pt',
        scale=4,
        model_path=r'src\models\esrgan\experiments\finetune_Realesr-general-x4v3_2\models\net_g_latest.pth',
        dni_weight=0.5,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        max_size=1280
    )
    preds = sr_yolo.predict(source='dataset/images')
    for r in preds:
        print(r.orig_img.shape, len(r.boxes))



# Example usage
# if __name__ == '__main__':
#     sr_yolo = SRYOLO(
#         yolo_weights='yolov9c.pt',
#         scale=4,
#         model_path=r'src\models\esrgan\experiments\finetune_Realesr-general-x4v3_2\models\net_g_latest.pth',
#         dni_weight=0.5,
#         tile=0,
#         tile_pad=10,
#         pre_pad=0
#     )
#     preds = sr_yolo.predict(source='path/to/images')
#     print(preds)


# Usage examples:

# sr_yolo.train(data='dataset.yaml', epochs=20)
# sr_yolo.val(data='dataset.yaml')
# sr_yolo.predict(source='image.jpg')






