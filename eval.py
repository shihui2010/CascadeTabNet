from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import json
import glob
import numpy as np
from tqdm import tqdm


OUT_DIR="/home/ubuntu/WikiDTEval"


def load_model():
    config_file = 'Config/cascade_mask_rcnn_hrnetv2p_w32_20e.py'
    checkpoint_file = 'epoch_36.pth'
    return init_detector(config_file, checkpoint_file, device='cuda:0')


def get_image(subset):
    path = f"/home/ubuntu/Dataset/{subset}/sub_page/images/*.png"
    return glob.glob(path)


def inference_main(model, subset):
    images = get_image(subset)
    fout = open(f"{OUT_DIR}/{subset}/cascade_tablenet_pred.json", "w") 
    for im in tqdm(images):
        try:
            bbox_result, _ = inference_detector(model, im)
        except Exception as e:
            print(e)
            continue
        labels = [np.full(bbox.shape[0], i, dtype=np.int32) 
                for i, bbox in enumerate(bbox_result)]
        labels = np.concatenate(labels).tolist()
        bboxes = np.vstack(bbox_result).tolist()
        fout.write(json.dumps({"pred": list(zip(bboxes, labels)), "page_id": im.split("/")[-1][:-4]}))
        fout.write("\n")
    fout.close()


if __name__ == "__main__":
    model = load_model()
    inference_main(model, "WTQ")
    inference_main(model, "TFC")
