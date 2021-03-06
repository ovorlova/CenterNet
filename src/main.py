from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory
from tensorboardX import SummaryWriter
from os import system
import math

def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test 
  torch.backends.cudnn.deterministic = True
  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)

  logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv)
  lr = opt.lr
  if (opt.cycle_exp):
    lr = opt.max_lr
  optimizer = torch.optim.Adam(model.parameters(), lr)
  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.resume, lr, opt.lr_step)

  Trainer = train_factory[opt.task]
  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  print('Setting up data...')
  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), 
      batch_size=1, 
      shuffle=False,
      num_workers=1,
      pin_memory=True
  )

  if opt.test:
    _, preds, _, hms = trainer.val(0, val_loader, None)
    val_loader.dataset.run_eval(preds, '', hms=hms)
    return

  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), 
      batch_size=opt.batch_size, 
      shuffle=True,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=True
  )

  print('Starting training...')
  exp_id = 'exp/'+opt.exp_id
  writer = SummaryWriter(exp_id)

  best = 1e10
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    print('learning rate: ', lr)
    writer.add_scalar('Learning rate', lr, epoch)
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _, writer, _ = trainer.train(epoch, train_loader, writer)
    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer)
    
      with torch.no_grad():
        log_dict_val, preds, writer, hms = trainer.val(epoch, val_loader, writer)
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
      if log_dict_val[opt.metric] < best:
        best = log_dict_val[opt.metric]
        save_model(os.path.join(opt.save_dir, 'model_best.pth'), 
                   epoch, model)
      if opt.not_hm_hp:
        ap, pckh = val_loader.dataset.run_eval(preds, '', hms=None, score=opt.score)
      else:
        ap, pckh = val_loader.dataset.run_eval(preds, '', hms=hms, score=opt.score)
      for name in ap:
          writer.add_scalar('Test_AP/'+ name, ap[name], epoch)
      for name in pckh:
          writer.add_scalar('Test_PCKh/'+name, pckh[name], epoch)

    else:
      save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                 epoch, model, optimizer)
    logger.write('\n')
    if (opt.cycle_exp):
      iteration = epoch+opt.step_size
      cycle = math.floor(1.+(iteration)/(2.*opt.step_size))
      x = abs(iteration/opt.step_size - 2.*cycle + 1.)
      lr= opt.base_lr + (opt.max_lr-opt.base_lr)*max(0., (1.-x))*gamma**(iteration)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr

    if epoch in opt.lr_step:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                 epoch, model, optimizer)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
    if opt.exp_lr:
      lr *= opt.gamma
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
    if epoch % 5 == 0:
      save_model(os.path.join(opt.save_dir, 'model_val{}.pth'.format(str(epoch))), 
                 epoch, model, optimizer)
  logger.close()
  writer.close()


if __name__ == '__main__':
  opt = opts().parse()
  main(opt)
