# +
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import json
import glob
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageColor, ImageFilter, ImageDraw
import xml.etree.ElementTree as ET
from tqdm import tqdm


class F1Score:
    def __init__(self, IoU_thresh=0.5, area_thresh=1000):
        self.true_positive = 0
        self.false_positive = 0
        self.truth_count = 0
        self._iou_thresh = IoU_thresh
        self._area_thresh = area_thresh
        self._iou_total = 0
        self._iou_count = 0

    def update_state(self, n_truth, IoUs, areas=None):
        if areas is not None:
            IoUs = [i for (i, a) in zip(IoUs, areas) if a > self._area_thresh]
        self._iou_total += sum(IoUs)
        self._iou_count += len(IoUs)
        n_tp = sum(1 for i in IoUs if i >= self._iou_thresh)
        n_fp = len(IoUs) - n_tp
        self.true_positive += n_tp
        self.false_positive += n_fp
        self.truth_count += n_truth

    @property
    def precision(self):
        return self.true_positive / max(1, self.true_positive + self.false_positive)

    @property
    def recall(self):
        return self.true_positive / max(1, self.truth_count)

    @property
    def f1(self):
        return (2 * self.precision * self.recall) / max(1, self.precision + self.recall)

    @property
    def iou(self):
        return self._iou_total / max(1, self._iou_count)
    
    
def read_sample(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    bboxes = []
    for object_ in root.iter('object'):
        ymin, xmin, ymax, xmax = None, None, None, None
        for box in object_.findall("bndbox"):
            ymin = int(box.find("ymin").text)
            xmin = int(box.find("xmin").text)
            ymax = int(box.find("ymax").text)
            xmax = int(box.find("xmax").text)

        bbox = [xmin, ymin, xmax, ymax] # PASCAL VOC
        bboxes.append(bbox)
    im_file = root.find("filename").text
    return im_file, bboxes


# -


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
