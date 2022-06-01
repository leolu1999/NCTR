import argparse
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from build_scheduler import build_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from load_data import OxfordTrain, Oxfordval, collate_batch
from models.NCTR import NCTR
from models.superpoint_e2e import SuperPoint
from utils.common import increment_path, init_seeds, clean_checkpoint, time_synchronized, test_model
from utils.preprocess_utils import torch_find_matches
from torch.cuda.amp import autocast, GradScaler


def train(config):
    save_dir = Path(config['train_params']['save_dir'])
    weight_dir = save_dir / "weights"
    weight_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / 'config.yaml', 'w') as file:
        yaml.dump(config, file, sort_keys=False)

    init_seeds(config['train_params']['init_seed'])

    config['NCTR_params']['GNN_layers'] = ['self', 'cross'] * config['NCTR_params']['num_layers']
    NCTR_model = NCTR(config['NCTR_params']).to(device)
    superpoint_model = SuperPoint(config['superpoint_params']).to(device)

    start_epoch = config['train_params']['start_epoch'] if config['train_params']['start_epoch'] > -1 else 0

    if config['NCTR_params']['restore_path']:
        restore_dict = torch.load(config['NCTR_params']['restore_path'], map_location=device)
        NCTR_model.load_state_dict(
            clean_checkpoint(restore_dict['model'] if 'model' in restore_dict else restore_dict))
        print("Restored NCTR weights..")
        superpoint_model.load_state_dict(
            clean_checkpoint(restore_dict['model_sp'] if 'model_sp' in restore_dict else restore_dict))
        print("Restored SuperPoint weights..")
        if config['train_params']['start_epoch'] < 0:
            start_epoch = restore_dict['epoch'] + 1

    optimizer = torch.optim.AdamW([{'params':filter(lambda p: p.requires_grad, superpoint_model.parameters()), 'lr':config['optimizer_params']['sp_lr']},
                                   {'params':NCTR_model.parameters(), 'lr':config['optimizer_params']['lr']}])
    print('SuperPoint: %g parameters' % (len(optimizer.param_groups[0]['params'])))
    print('NCTR: %g parameters' % (len(optimizer.param_groups[1]['params'])))
    scaler = GradScaler()

    if config['NCTR_params']['restore_path']:
        if ('optimizer' in restore_dict) and config['train_params']['restore_opt']:
            optimizer.load_state_dict(restore_dict['optimizer'])
            print("Restored optimizer...")

    train_dataset = OxfordTrain(config['dataset_params'], typ="train")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config['train_params']['batch_size'],
                                                   num_workers=config['train_params']['num_workers'],
                                                   shuffle=True,
                                                   sampler=None,
                                                   collate_fn=collate_batch,
                                                   pin_memory=True)
    num_batches = len(train_dataloader)
    lr_scheduler = build_scheduler(config, optimizer, num_batches)
    val_dataset = Oxfordval(config['dataset_params'])
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                                 num_workers=0,
                                                 sampler=None,
                                                 collate_fn=collate_batch,
                                                 pin_memory=True,
                                                 shuffle=False)

    start_time = time.time()
    num_epochs = config['train_params']['num_epochs']
    best_val_score = 1e-10
    print("Started training for {} epochs".format(num_epochs))
    print("Number of batches: {}".format(num_batches))
    log_dir = "tf-logs\\final_model\\"+config['train_params']['experiment_name']
    print("log_dir: " + log_dir)
    writer = SummaryWriter(log_dir)
    for epoch in range(start_epoch, num_epochs):
        NCTR_model.train()
        print("Started epoch: {}".format(epoch))
        epoch_loss = 0
        pbar = enumerate(train_dataloader)
        pbar = tqdm(pbar, total=num_batches)
        optimizer.zero_grad()
        mloss = torch.zeros(6, device=device)
        print(('\n' + '%10s' * 10) % (
        'Epoch', 'gpu_mem', 'Iteration', 'PosLoss', 'NegLoss', 'TotLoss', 'Dtime', 'Ptime', 'Mtime', 'lr'))
        t5 = time_synchronized()
        for i, (orig_warped, homographies) in pbar:
            optimizer.zero_grad()
            with autocast():
                t1 = time_synchronized()
                orig_warped = orig_warped.to(device, non_blocking=True)
                homographies = homographies.to(device, non_blocking=True)
                midpoint = len(orig_warped) // 2
                all_match_index_0, all_match_index_1, all_match_index_2 = torch.empty(0, dtype=torch.int64,
                                                                                      device=homographies.device), torch.empty(
                    0, dtype=torch.int64, device=homographies.device), torch.empty(0, dtype=torch.int64,
                                                                                   device=homographies.device)
                t2 = time_synchronized()
                superpoint_results = superpoint_model.forward_train({'homography': homographies, 'image': orig_warped})
                keypoints = torch.stack(superpoint_results['keypoints'], 0)
                descriptors = torch.stack(superpoint_results['descriptors'], 0)
                scores = torch.stack(superpoint_results['scores'], 0)
                keypoints0, keypoints1 = keypoints[:midpoint, :, :], keypoints[midpoint:, :, :]
                descriptors0, descriptors1 = descriptors[:midpoint, :, :], descriptors[midpoint:, :, :]
                scores0, scores1 = scores[:midpoint, :], scores[midpoint:, :]
                images0, images1 = orig_warped[:midpoint, :, :, :], orig_warped[midpoint:, :, :, :]
                for k in range(midpoint):
                    ma_0, ma_1, miss_0, miss_1 = torch_find_matches(keypoints0[k], keypoints1[k], homographies[k],
                                                                    dist_thresh=3, n_iters=1)
                    all_match_index_0 = torch.cat([all_match_index_0,
                                                   torch.empty(len(ma_0) + len(miss_0) + len(miss_1), dtype=torch.long,
                                                               device=ma_0.device).fill_(k)])
                    all_match_index_1 = torch.cat([all_match_index_1, ma_0, miss_0,
                                                   torch.empty(len(miss_1), dtype=torch.long,
                                                               device=miss_1.device).fill_(-1)])
                    all_match_index_2 = torch.cat([all_match_index_2, ma_1, torch.empty(len(miss_0), dtype=torch.long,
                                                                                        device=miss_0.device).fill_(-1),
                                                   miss_1])
                match_indexes = torch.stack([all_match_index_0, all_match_index_1, all_match_index_2], -1)
                gt_vector = torch.ones(len(match_indexes), dtype=torch.float32, device=match_indexes.device)
                t3 = time_synchronized()
                keypoints0, keypoints1 = keypoints0.transpose(0, 1), keypoints1.transpose(0, 1)
                descriptors0, descriptors1 = descriptors0.permute(2, 0, 1), descriptors1.permute(2, 0, 1)
                scores0, scores1 = scores0.transpose(0, 1), scores1.transpose(0, 1)
                input = {
                    'keypoints0': keypoints0, 'keypoints1': keypoints1,
                    'descriptors0': descriptors0, 'descriptors1': descriptors1,
                    'image0': images0, 'image1': images1,
                    'scores0': scores0, 'scores1': scores1,
                    'matches': match_indexes,
                    'gt_vec': gt_vector
                }
                total_loss, pos_loss, neg_loss = NCTR_model(input)
            epoch_loss += total_loss.item()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step_update(epoch * num_batches + i)
            t4 = time_synchronized()
            data_time, preprocess_time, model_time = torch.tensor(t1 - t5, device=device), torch.tensor(t3 - t2,
                                                                                                        device=device), torch.tensor(
                t4 - t3, device=device)
            loss_items = torch.stack((pos_loss, neg_loss, total_loss, data_time, preprocess_time, model_time)).detach()
            mloss = (mloss * i + loss_items) / (i + 1)
            lr = optimizer.param_groups[0]['lr']
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.4g' * 8) % (str(epoch), mem, i, *mloss, lr)
            pbar.set_description(s)
            if ((i + 1) % 20000) == 0:
                ckpt = {'epoch': epoch,
                        'iter': i,
                        'model': NCTR_model.state_dict(),
                        'model_sp': superpoint_model.state_dict(),
                        'optimizer': optimizer.state_dict()}
                print("\nDoing evaluation..")
                loss = mloss[2].item()
                with torch.no_grad():
                    eval_NCTR = NCTR_model
                    results = test_model(val_dataloader, superpoint_model, eval_NCTR,
                                         config['train_params']['val_images_count'], device)
                    writer.add_scalar('ransac_auc', results['ransac_auc'][2], epoch * 4 + (i + 1) / 20000)
                    writer.add_scalar('dlt_auc', results['dlt_auc'][2], epoch * 4 + (i + 1) / 20000)
                    writer.add_scalar('precision', results['precision'], epoch * 4 + (i + 1) / 20000)
                    writer.add_scalar('recall', results['recall'], epoch * 4 + (i + 1) / 20000)
                    writer.add_scalar('loss', loss, epoch * 4 + (i + 1) / 20000)
                    writer.add_scalar('f1_score', results['f1_score'], epoch * 4 + (i + 1) / 20000)
                    writer.add_scalar('learning_rate', lr, epoch * 4 + (i + 1) / 20000)
                torch.save(ckpt, weight_dir / 'lastiter.pt')
                if results['f1_score'] > best_val_score:
                    best_val_score = results['f1_score']
                    print("Saving best model at epoch {} with f1-score {}".format(epoch, best_val_score))
                    torch.save(ckpt, weight_dir / 'best.pt')
            t5 = time_synchronized()
        print("\nDoing evaluation..")
        with torch.no_grad():
            eval_NCTR = NCTR_model
            results = test_model(val_dataloader, superpoint_model, eval_NCTR,
                                 config['train_params']['val_images_count'], device)
            writer.add_scalar('epoch_ransac_auc', results['ransac_auc'][2], epoch)
            writer.add_scalar('epoch_dlt_auc', results['dlt_auc'][2], epoch)
            writer.add_scalar('epoch_precision', results['precision'], epoch)
            writer.add_scalar('epoch_recall', results['recall'], epoch)
            writer.add_scalar('epoch_f1_score', results['f1_score'], epoch)
            # writer.add_scalar('learning_rate', lr, epoch)
        ckpt = {'epoch': epoch,
                'iter': -1,
                'model': NCTR_model.state_dict(),
                'model_sp': superpoint_model.state_dict(),
                'optimizer': optimizer.state_dict(), 'metrics': results}
        torch.save(ckpt, weight_dir / 'last.pt')
        if results['f1_score'] > best_val_score:
            best_val_score = results['f1_score']
            print("Saving best model at epoch {} with f1-score {}".format(epoch, best_val_score))
            torch.save(ckpt, weight_dir / 'best.pt')
        epoch_loss /= num_batches
        writer.add_scalar('epoch_loss', epoch_loss, epoch)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="configs/config.yaml", help="Path to the config file")
    parser.add_argument('--local_rank', type=int, default=-1, help="Rank of the process incase of DDP")
    opt = parser.parse_args()
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if opt.local_rank >= 0:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
    else:
        if "cpu" not in device: torch.cuda.set_device(device)
    with open(opt.config_path, 'r') as file:
        config = yaml.full_load(file)
    config["train_params"]['save_dir'] = increment_path(
        Path(config['train_params']['output_dir']) / config['train_params']['experiment_name'])
    if opt.local_rank in [0, -1]:
        for i, k in config.items():
            print("{}: ".format(i))
            print(k)
    train(config)
