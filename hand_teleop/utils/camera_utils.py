import sapien.core as sapien

import numpy as np


def fetch_texture(cam: sapien.CameraEntity, texture_name: str, return_torch=True):
    dlpack = cam.get_dl_tensor(texture_name)
    if not return_torch:
        shape = sapien.dlpack.dl_shape(dlpack)
        output_array = np.zeros(shape, dtype=np.float32)
        sapien.dlpack.dl_to_numpy_cuda_async_unchecked(dlpack, output_array)
        sapien.dlpack.dl_cuda_sync()
        return output_array
    else:
        import torch

        return torch.from_dlpack(dlpack)
