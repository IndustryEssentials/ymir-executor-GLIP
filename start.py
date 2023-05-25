import logging
import os
import subprocess
import sys

from easydict import EasyDict as edict

from ymir_exc import monitor
from ymir_exc.util import YmirStage, find_free_port, get_bool, get_merged_config, write_ymir_monitor_process
from ymir.util import process_error, create_ymir_dataset_config


def start(cfg: edict) -> int:
    logging.info(f'merged config: {cfg}')

    os.makedirs('/app/OUTPUT', exist_ok=True)

    if cfg.ymir.run_training:
            _run_training(cfg)
    else:
        if cfg.ymir.run_mining:
            _run_mining(cfg)
        if cfg.ymir.run_infer:
            _run_infer(cfg)

    return 0


def _run_training(cfg: edict) -> None:
    """
    function for training task
    1. convert dataset
    2. training model
    3. save model weight/hyperparameter/... to design directory
    """
    # 1. convert dataset
    out_dir = cfg.ymir.output.root_dir
    # out_dir = 'in'

    # logging.info(f'generate {out_dir}/data.yaml')
    write_ymir_monitor_process(cfg, task='training', naive_stage_percent=1.0, stage=YmirStage.PREPROCESS)

    # 2. training model
    print(cfg.param)

    gpu_id = (cfg.param.get('gpu_id'))
    assert gpu_id != None,'Invalid CUDA, GPU id needed'
    gpu_id = str(gpu_id)
    gpu_count: int = len(gpu_id.split(',')) if gpu_id else 0
  
    port: int = find_free_port()
    epochs: int = int(cfg.param.epochs)
    custom_shot_and_epoch_and_general_copy = f"0_{epochs}_1"
    create_ymir_dataset_config(cfg)

    # models_dir = cfg.ymir.output.models_dir
    # project = os.path.dirname(models_dir)
    # name = os.path.basename(models_dir)
    # assert os.path.join(project, name) == models_dir

    commands = ['python3']


    commands.extend(f'-m torch.distributed.launch --nproc_per_node {gpu_count} --master_port {port}'.split())
    # commands.extend([
    #     'tools/finetune.py', '--skip-test', '--config-file', 'configs/pretrain/glip_A_Swin_T_O365.yaml',
    #     '--ft-tasks', 'configs/ymir_dataset.yaml',
    #     '--custom_shot_and_epoch_and_general_copy', '0_200_1',
    #     '--evaluate_only_best_on_test', '--push_both_val_and_test',
    #     'MODEL.WEIGHT', 'MODEL/glip_a_tiny_o365.pth', 'SOLVER.USE_AMP', 'True', 'TEST.DURING_TRAINING', 'True',
    #     'TEST.IMS_PER_BATCH', str(batch_size),'SOLVER.IMS_PER_BATCH', str(batch_size),
    #     'SOLVER.WEIGHT_DECAY', '0.05', 'TEST.EVAL_TASK', 'detection','DATASETS.TRAIN_DATASETNAME_SUFFIX', '_grounding',
    #     'MODEL.BACKBONE.FREEZE_CONV_BODY_AT', '-1', 'MODEL.DYHEAD.USE_CHECKPOINT', 'True', 'SOLVER.FIND_UNUSED_PARAMETERS', 'False',
    #   'SOLVER.TEST_WITH_INFERENCE', 'True', 'SOLVER.USE_AUTOSTEP', 'True', 'DATASETS.USE_OVERRIDE_CATEGORY' ,'True', 'SOLVER.SEED' ,'10',
    #   'DATASETS.SHUFFLE_SEED' ,'3' ,'DATASETS.USE_CAPTION_PROMPT', 'True' ,'DATASETS.DISABLE_SHUFFLE', 'True', 
    #   'SOLVER.STEP_PATIENCE' ,'2' ,'SOLVER.CHECKPOINT_PER_EPOCH', '1.0', 'SOLVER.AUTO_TERMINATE_PATIENCE', '4' ,'SOLVER.BASE_LR', '0.0001',
    #   'SOLVER.MODEL_EMA', '0.0' ,'SOLVER.MAX_EPOCH',str(epochs),'SOLVER.TUNING_HIGHLEVEL_OVERRIDE', 'full', 'DATALOADER.DISTRIBUTE_CHUNK_AMONG_NODE' ,'False'
    # ])
    commands.extend([
        'tools/finetune.py', '--skip-test', '--config-file', 'configs/pretrain/glip_A_Swin_T_O365.yaml',
        '--ft-tasks', 'configs/ymir_dataset.yaml',
        '--custom_shot_and_epoch_and_general_copy', custom_shot_and_epoch_and_general_copy,
        '--evaluate_only_best_on_test', '--push_both_val_and_test',
        'MODEL.WEIGHT', 'MODEL/glip_a_tiny_o365.pth', 'SOLVER.USE_AMP', 'True', 'TEST.DURING_TRAINING', 'True',
        'SOLVER.WEIGHT_DECAY', '0.05', 'TEST.EVAL_TASK', 'detection','DATASETS.TRAIN_DATASETNAME_SUFFIX', '_grounding',
        'MODEL.BACKBONE.FREEZE_CONV_BODY_AT', '-1', 'MODEL.DYHEAD.USE_CHECKPOINT', 'True', 'SOLVER.FIND_UNUSED_PARAMETERS', 'False',
      'SOLVER.TEST_WITH_INFERENCE', 'True', 'SOLVER.USE_AUTOSTEP', 'True', 'DATASETS.USE_OVERRIDE_CATEGORY' ,'True', 'SOLVER.SEED' ,'10',
      'DATASETS.SHUFFLE_SEED' ,'3' ,'DATASETS.USE_CAPTION_PROMPT', 'True' ,'DATASETS.DISABLE_SHUFFLE', 'True', 
      'SOLVER.STEP_PATIENCE' ,'2' ,'SOLVER.CHECKPOINT_PER_EPOCH', '1.0', 'SOLVER.AUTO_TERMINATE_PATIENCE', '4' ,
      'SOLVER.MODEL_EMA', '0.0' ,'SOLVER.TUNING_HIGHLEVEL_OVERRIDE', 'full', 'DATALOADER.DISTRIBUTE_CHUNK_AMONG_NODE' ,'False'
    ])



    logging.info(f'start training: {commands}')
    
    try:
        subprocess.run(commands, check=True)
    except Exception as e:
        print(e)
    write_ymir_monitor_process(cfg, task='training', naive_stage_percent=1.0, stage=YmirStage.TASK)

    # if task done, write 100% percent log
    monitor.write_monitor_logger(percent=1.0)


def _run_mining(cfg: edict) -> None:
    # generate data.yaml for mining
    try:

        write_ymir_monitor_process(cfg, task='mining', naive_stage_percent=1.0, stage=YmirStage.PREPROCESS)
        gpu_id: str = str(cfg.param.get('gpu_id', '0'))
        assert gpu_id != None,'Invalid CUDA, GPU id needed'
        gpu_count: int = len(gpu_id.split(',')) if gpu_id else 0

        mining_algorithm = cfg.param.get('mining_algorithm', 'entropy')
        support_mining_algorithms = ['entropy']
        port: int = find_free_port()
        if mining_algorithm not in support_mining_algorithms:
            raise FileNotFoundError(f'unknown mining algorithm {mining_algorithm}, not in {support_mining_algorithms}')

        command = f'python3 -m torch.distributed.launch --nproc_per_node={gpu_count} --master_port {port} ymir/mining/ymir_mining_{mining_algorithm}.py'

    except Exception as e:
            process_error(e)
            exit()
    logging.info(f'mining: {command}')
    subprocess.run(command.split(), check=True)
    write_ymir_monitor_process(cfg, task='mining', naive_stage_percent=1.0, stage=YmirStage.POSTPROCESS)


def _run_infer(cfg: edict) -> None:
    # generate data.yaml for infer

    write_ymir_monitor_process(cfg, task='infer', naive_stage_percent=1.0, stage=YmirStage.PREPROCESS)

    gpu_id = (cfg.param.get('gpu_id'))
    assert gpu_id != None,'Invalid CUDA, GPU id needed'
    gpu_id = str(gpu_id)
    gpu_count: int = len(gpu_id.split(',')) if gpu_id else 0


    task_weight = cfg.param.model_params_path

    port: int = find_free_port()
    if not task_weight:
        raise FileNotFoundError('task_weight not found')
    command = f'python3 -m torch.distributed.launch --nproc_per_node={gpu_count} --master_port {port} ymir/ymir_infer.py'

    logging.info(f'infer: {command}')
    subprocess.run(command.split(), check=True)

    write_ymir_monitor_process(cfg, task='infer', naive_stage_percent=1.0, stage=YmirStage.POSTPROCESS)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout,
                        format='%(levelname)-8s: [%(asctime)s] %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        level=logging.INFO)
    try:
        # with open('in/config.yaml', "r") as f:
        #     cfg = yaml.safe_load(f)
        cfg = get_merged_config()
    except Exception as e:
        process_error(e)

    os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

    # merged_cfg = edict()
    # merged_cfg.param = cfg
    sys.exit(start(cfg))
