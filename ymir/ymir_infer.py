import argparse
import datetime
import os

from easydict import EasyDict as edict
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
import matplotlib.pyplot as plt
from maskrcnn_benchmark.config import cfg
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from tqdm import tqdm
from ymir_exc import result_writer as rw
from ymir_exc.util import  get_merged_config ,write_ymir_monitor_process,YmirStage
from ymir.util import process_error, combine_caption, get_weight_file,gen_anns_from_dets


def init_distributed_mode(args):
    """Initialize distributed training, if appropriate"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print("| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True)

    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank,
        timeout=datetime.timedelta(0, 7200)
    )
    dist.barrier()
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def imshow(file,img, caption):
    plt.figure("Image")
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.figtext(0.5, 0.09, caption, wrap=True, horizontalalignment='center', fontsize=5)
    plt.savefig('infer_result/'+file.split('/')[-1])
    plt.show()


def load(img_path):

    pil_image = Image.open(img_path).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image


def run(ymir_cfg: edict, args):
    confidence = ymir_cfg.param.get('conf_thres')
    MAX_SIZE_TEST = ymir_cfg.param.get('MAX_SIZE_TEST')
    MIN_SIZE_TEST = ymir_cfg.param.get('MIN_SIZE_TEST')

    gpu_id = (ymir_cfg.param.get('gpu_id'))
    assert gpu_id != None,'Invalid CUDA, GPU id needed'
    gpu_id = str(gpu_id)
    gpu_count: int = len(gpu_id.split(',')) if gpu_id else 0
    distributed = gpu_count > 1
    if distributed:
        init_distributed_mode(args)
        print("Passed distributed init")

    config_file = "configs/pretrain/glip_A_Swin_T_O365.yaml"
    weight_file = "MODEL/glip_a_tiny_o365.pth"

    task_weight = get_weight_file(ymir_cfg)

    captions = ymir_cfg.param.prompt

    cfg.local_rank = args.local_rank
    cfg.num_gpus = gpu_count

    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
    cfg.merge_from_list(["INPUT.MAX_SIZE_TEST", MAX_SIZE_TEST])
    cfg.merge_from_list(["INPUT.MIN_SIZE_TEST", MIN_SIZE_TEST])
    cfg.freeze()
    log_dir = cfg.OUTPUT_DIR
    logger = setup_logger("maskrcnn_benchmark", log_dir, get_rank())
    logger.info("Using {} GPUs".format(gpu_count))
    if not task_weight:
        raise FileNotFoundError('task_weight not found')

    with open(ymir_cfg.ymir.input.candidate_index_file, 'r') as f:
        images = [line.strip() for line in f.readlines()]

    if  args.rank != -1:
        images_rank = images[args.rank:: args.world_size]
    else:
        images_rank = images

    glip_demo = GLIPDemo(cfg,
                         task_weight,
                         min_image_size=MIN_SIZE_TEST,
                         confidence_threshold=confidence,
                         show_mask_heatmaps=False)
    glip_demo.color=(255,0,255)
    caption = combine_caption(captions)

    monitor_gap = max(1, len(images_rank) // 1000)
    results = []
    pbar = tqdm(images_rank) if args.rank in [0, -1] else images_rank
    for idx, img_path in enumerate(pbar):
        # top_predictions.mode : xyxy
        # batch: /in/assets/41/68624cc85d7515e9649d324d78bf875ed6dd9c41.jpg
        image = load(img_path)
        top_predictions = glip_demo.inference(image, caption)
        if idx % monitor_gap == 0:
            write_ymir_monitor_process(ymir_cfg,
                                       task='infer',
                                       naive_stage_percent=idx  / len(images_rank),
                                       stage=YmirStage.TASK)

        results.append(dict(img_path=img_path, top_predictions=top_predictions,caption =caption))
    torch.save(results, f'/out/infer_results_{max(0,args.rank)}.pt')


def main() -> int:
    parser = argparse.ArgumentParser(description="PyTorch Detection to Grounding Inference")
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--rank", type=int, default=-1)

    args = parser.parse_args()
    ymir_cfg = get_merged_config()

    run(ymir_cfg,args)
    if args.world_size > 1:
        dist.barrier()
    if args.rank in [0,-1]:
        results = []
        for rank in range(args.world_size):
            results.append(torch.load(f'/out/infer_results_{rank}.pt'))
        ymir_infer_result = dict()

        for result in results:
            for img_data in result:
                top_predictions = img_data['top_predictions']
                img_path = img_data['img_path']
                caption = img_data['caption']

                ymir_infer_result = gen_anns_from_dets(top_predictions,ymir_infer_result,caption,img_path)
        if 'annotations' not in ymir_infer_result:
            ymir_infer_result['annotations'] = []
        rw.write_infer_result(infer_result=ymir_infer_result,algorithm='segmentation')
    return 0


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        process_error(e)
