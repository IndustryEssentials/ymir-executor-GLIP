from ymir_exc.code import ExecutorState, ExecutorReturnCode
from ymir_exc import monitor
import urllib
from easydict import EasyDict as edict
import os, shutil
import json,yaml
import logging
from typing import Any, List
from PIL import Image
import numpy as np
import torch.utils.data as td
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.miscellaneous import  save_config
from pycocotools import mask as maskUtils

def split_into_train_val(index_file,imgdir,output_json_file):
    '''
    /out/tmp/assets/2e/b1a44898eefd6e08ee28e455ab046187eaed1a2e.png      /out/tmp/annotations/coco-annotations.json
    '''

    if not os.path.isdir(imgdir):
        logging.info(f'make dir for train image in {imgdir}')
        os.makedirs(imgdir)

    img_names = []
    with open(index_file,'r') as f:
        for line in f.readlines():
            img_path, input_json_file = line.strip().split()  
            shutil.copy(img_path,imgdir)
            img_name = os.path.basename(img_path)
            img_names.append(img_name)

    with open(input_json_file, 'r') as f:
        data = json.load(f)
    new_categories=[]


    new_data = {
        # 'info': data['info'],
        # 'licenses': data['licenses'],
        'images': [],
        'annotations': [],
        'categories': data['categories']
    }
    
    image_ids = set()
    for image in data['images']:
        if image['file_name'] in img_names:
            new_data['images'].append(image)
            image_ids.add(image['id'])
    
    for annotation in data['annotations']:
        if annotation['image_id'] in image_ids:
            new_data['annotations'].append(annotation)
    
    with open(output_json_file, 'w') as f:
        json.dump(new_data, f)
    return new_data['categories']


def create_ymir_dataset_config(ymir_cfg):
    epochs: int = int(ymir_cfg.param.epochs)
    batch_size_per_gpu: int = int(ymir_cfg.param.batch_size_per_gpu)

    gpu_id = (ymir_cfg.param.get('gpu_id'))
    assert gpu_id != None,'Invalid CUDA, GPU id needed'
    gpu_id = str(gpu_id)
    gpu_count: int = len(gpu_id.split(',')) if gpu_id else 0
    batch_size: int = batch_size_per_gpu * max(1, gpu_count)

    training_index_file = ymir_cfg.ymir.input.training_index_file
    val_index_file = ymir_cfg.ymir.input.val_index_file
    
    split_train_img_dir = '/out/tmp/train/images/'
    split_val_img_dir = '/out/tmp/val/images/'

    val_output_json_file='/out/tmp/val/val.json'
    train_output_json_file='/out/tmp/train/train.json'

    dataset_category = split_into_train_val(training_index_file,split_train_img_dir,train_output_json_file)
    _ = split_into_train_val(val_index_file,split_val_img_dir,val_output_json_file)

    YMIR_DATASET = dict()
    DATASETS = dict()
    DATALOADER = dict()
    REGISTER  = dict()
    
    DATALOADER['ASPECT_RATIO_GROUPING'] = False
    DATALOADER['SIZE_DIVISIBILITY'] = 32 

    REGISTER['test'] = dict(ann_file=val_output_json_file,img_dir=split_val_img_dir)
    REGISTER['train'] = dict(ann_file=train_output_json_file,img_dir=split_train_img_dir)
    REGISTER['val'] = dict(ann_file=val_output_json_file,img_dir=split_val_img_dir)


    DATASETS['OVERRIDE_CATEGORY']=str(dataset_category)
    DATASETS['GENERAL_COPY']=16
    DATASETS['REGISTER'] = REGISTER
    DATASETS['TEST']=str(("val",))
    DATASETS['TRAIN']=str(("train",))
    
    INPUT = dict()

    INPUT['MAX_SIZE_TEST'] = int(ymir_cfg.param.MAX_SIZE_TEST)
    INPUT['MAX_SIZE_TRAIN'] = int(ymir_cfg.param.MAX_SIZE_TRAIN)
    INPUT['MIN_SIZE_TEST'] = int(ymir_cfg.param.MIN_SIZE_TEST)
    INPUT['MIN_SIZE_TRAIN'] = int(ymir_cfg.param.MIN_SIZE_TRAIN)

    MODEL = dict()
    NUM_CLASSES = len(dataset_category) +1 # if  len(dataset_category)>1 else 1
    MODEL['ATSS'] = dict(NUM_CLASSES = NUM_CLASSES)
    MODEL['DYHEAD'] = dict(NUM_CLASSES = NUM_CLASSES)
    MODEL['FCOS'] = dict(NUM_CLASSES = NUM_CLASSES)
    MODEL['ROI_BOX_HEAD'] = dict(NUM_CLASSES = NUM_CLASSES)
    
    TEST = dict()
    TEST['IMS_PER_BATCH'] = batch_size

    SOLVER = dict()
    SOLVER['CHECKPOINT_PERIOD'] = int(ymir_cfg.param.CHECKPOINT_PERIOD )
    SOLVER['IMS_PER_BATCH'] = batch_size
    SOLVER['MAX_EPOCH'] = epochs
    SOLVER['BASE_LR'] = ymir_cfg.param.BASE_LR

    YMIR_DATASET['DATALOADER'] = DATALOADER
    YMIR_DATASET['DATASETS']=DATASETS
    YMIR_DATASET['INPUT'] = INPUT
    YMIR_DATASET['MODEL'] = MODEL
    YMIR_DATASET['SOLVER'] = SOLVER
    YMIR_DATASET['TEST'] = TEST
    
    YMIR_DATASET['OUTPUT_DIR'] = ymir_cfg.ymir.output.models_dir
    YMIR_DATASET['TENSORBOARD_EXP'] = ymir_cfg.ymir.output.tensorboard_dir


    
    with open("configs/ymir_dataset.yaml", "w") as f:
        yaml.safe_dump(YMIR_DATASET, f)


def modefy_task_config(model_cfg,ymir_cfg):
    test_img_dir = '/out/tmp/test/images/'
    if not os.path.isdir(test_img_dir):
        os.makedirs(test_img_dir)
    with open(ymir_cfg.ymir.input.candidate_index_file,'r') as f:
        for line in f.readlines():
            img_path = line.strip()
            shutil.copy(img_path,test_img_dir)
    cfg.merge_from_file(model_cfg)
    cfg.DATASETS.REGISTER.test.ann_file=''
    cfg.DATASETS.REGISTER.test.img_dir=test_img_dir
    cfg.DATASETS.TEST = ['test']
    save_config(cfg, ymir_cfg.param.task_config)







def process_error(e,msg='defult'):
    print(type(e),e,'=========')
    if msg=='dataloader' or 'dataloader' in e.args:
        crash_code = ExecutorReturnCode.RC_EXEC_DATASET_ERROR 
    elif type(e) == urllib.error.HTTPError:
        crash_code = ExecutorReturnCode.RC_EXEC_NETWORK_ERROR 
    elif type(e) ==FileNotFoundError:
        crash_code = ExecutorReturnCode.RC_EXEC_CONFIG_ERROR 
    elif 'CUDA out of memory' in repr(e):
        crash_code = ExecutorReturnCode.RC_EXEC_OOM 
    elif 'Invalid CUDA' in repr(e):
        crash_code = ExecutorReturnCode.RC_EXEC_NO_GPU 
    else:
        crash_code = ExecutorReturnCode.RC_CMD_CONTAINER_ERROR
    monitor.write_monitor_logger(percent=1,
                                state=ExecutorState.ES_ERROR,
                                return_code=crash_code)
    raise RuntimeError(f"App crashed with code: {crash_code}")

def load_image_file(img_path):

    pil_image = Image.open(img_path).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return dict(image=image, img_path=img_path)


class YmirDataset(td.Dataset):
    def __init__(self, images: List[Any], load_fn=None):
        super().__init__()
        self.images = images
        self.load_fn = load_fn

    def __getitem__(self, index):
        return self.load_fn(self.images[index])

    def __len__(self):
        return len(self.images)

def combine_caption(captions):
    return captions.replace(';',' . ')



def gen_anns_from_dets(top_predictions,ymir_infer_result,caption,img_path):
    """Generates json annotations from detections."""

    # Load the detections
    # all_boxes = top_predictions.convert('xywh')
    all_boxes = top_predictions.convert('xywh')
    all_boxes_covered = all_boxes.bbox
    anns = []
    imgs=[]
    cats=[]
    print(top_predictions.size)

    if 'images' in ymir_infer_result:
        img_id = len(ymir_infer_result['images'])
    else:
        img_id = 0
    img = {
            "id": img_id,
            "file_name": img_path.split('/')[-1],
            "width": top_predictions.size[0],
            "height": top_predictions.size[1],
            "date_captured": "",
            "license": 1,
            "coco_url": "",
            "flickr_url": ""
        }
    
    if 'images' in ymir_infer_result:
        ymir_infer_result['images'].append(img)
    else:
        ymir_infer_result['images']=[img]

    caption= caption.split(' . ')
    for i in range(len(caption)):
        cat= {
            "id": i+1,
            "name": caption[i],
            "supercategory": "object"
        }
        cats.append(cat)
    ymir_infer_result['categories'] = cats

    if 'annotations' in ymir_infer_result:
        len_anno = len(ymir_infer_result['annotations'])
    else:
        len_anno = 0

    for j in range(all_boxes_covered.shape[0]):
        bbox = (list(map(int,all_boxes_covered[j].numpy().tolist())))
        ann_i_j = {
            'id': len_anno + j,
            'image_id': img_id,
            'category_id': top_predictions.get_field('labels')[j].item(),
            # 'area':  ,
            'bbox': bbox,
            'confidence':  top_predictions.get_field('scores')[j].item(),
            'iscrowd': 0
        }
       
        if 'annotations' in ymir_infer_result:
            ymir_infer_result['annotations'].append(ann_i_j)
        else:
            ymir_infer_result['annotations']=[ann_i_j]

    return ymir_infer_result