from vp_engine import Engine
import cv2
import matplotlib
import numpy as np
from time import sleep

if __name__ == '__main__':

  engine = Engine('configs/d455.json', 'model_registry/registry.jsonl')
  engine.build()
  engine.start_inference()
  was_success: bool = engine.test() 

  cmap = matplotlib.colormaps.get_cmap('Spectral')

  img_path = '/home/ubuntu/Anantak/Workspaces/Sites/MULTI/KOHLER-TXBNWD/ROSBAG/Images/000/frame_000016_ts1760658836298_fn1891.png'
  # img_path = 'tests/resources/object_detection/inputs/20230710_103312.png'

  for i in range(0,10):
    # load image
    image = cv2.imread(img_path)
    # img_resized = cv2.resize(image, (1920, 1080))

    # input image
    was_success: bool = engine.input_image(image) 

    sleep(0.1) # make sure all inputs are processed

    # get outputs
    depth: np.ndarray = engine.get_head_output(0)
    object_detection: list[np.ndarray] = engine.get_head_output(1) # [labels, scores, normalized boxes (i.e. to [0,1] range)]
    raw_descriptors: np.ndarray = engine.get_head_output(2)


    # # Normalize for visualization
    # depth_colormap = cv2.applyColorMap(
    #     cv2.convertScaleAbs(depth, alpha=0.03), 
    #     cv2.COLORMAP_JET
    # )
    # cv2.imshow('Depth', depth_colormap)
    # cv2.waitKey(0)

    # Normalize and colorize for visualization
    depth_img = depth[:, :, 0]
    depth_img = depth_img.astype(np.float32)
    depth_img = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())
    depth_img = (cmap(depth_img)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    cv2.imshow('Depth', depth_img)
    cv2.waitKey(0)

    print(f"b Object Type: {type(raw_descriptors)}")
    print(f"b Element Data Type (dtype): {raw_descriptors.dtype}")
    print(f"b Shape: {raw_descriptors.shape}")

    # # Create colored segmentation mask
    # colored_seg = cv2.applyColorMap(seg_image.astype(np.uint8) * 10, cv2.COLORMAP_HSV)
    # cv2.imshow('Segmentation', colored_seg)
    # cv2.waitKey(0)

  cv2.destroyAllWindows()
