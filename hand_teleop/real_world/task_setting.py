from hand_teleop.env.rl_env.pc_processing import process_relocate_pc
from hand_teleop.real_world import lab
import numpy as np

CAMERA_CONFIG = {
    "relocate": {
        "relocate": dict(pose=lab.ROBOT2BASE * lab.CAM2ROBOT, fov=np.deg2rad(69.4), resolution=(128, 128)),
    }
}

empty_info = {}  # level empty dict for now, reserved for future
OBS_CONFIG = {
    "relocate": {
        "relocate": {"point_cloud": {"process_fn": process_relocate_pc, "num_points": 256}},
    }
}
