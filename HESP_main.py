import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# print("os.environ['CUDA_LAUNCH_BLOCKING']=" + os.environ['CUDA_LAUNCH_BLOCKING'])
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import argparse
import datetime
import time
import csv
import pandas as pd
import importlib
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

from utils import Logger, save_networks, load_networks
from core import train, test

from prompt.prompt import *

from loss.ConLoss import *

import clip

from datasets.dataset_MAFW import train_data_loader, test_data_loader, out_data_loader 

parser = argparse.ArgumentParser("Training")

# Dataset
parser.add_argument('--dataset', type=str, default='MAFW', help="AFEW | MAFW | OURDATA | MULTIDATA | DFEW")
parser.add_argument('--dataroot', type=str, default='/home/suyf/datasets/MAFW')
parser.add_argument('--outf', type=str, default='./log')
parser.add_argument('--workers', type=int, default=4, help='workers')

# optimization
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.01, help="learning rate for model")
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--stepsize', type=int, default=30)
parser.add_argument('--temp', type=float, default=1.0, help="temp")
parser.add_argument('--num-centers', type=int, default=1)

# model
parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for center loss")
parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")
parser.add_argument('--model', type=str, default='HESP')

# misc
parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=20)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=10)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='../log')
parser.add_argument('--loss', type=str, default='MYARPLoss')
parser.add_argument('--eval', action='store_true', help="Eval", default=False)

# 获取当前时间，格式化为 "YYYYMMDD_HHMMSS"
now = datetime.datetime.now()
timestamp = now.strftime("%y%m%d-%H%M")
print("datetime is " + timestamp)

def main_worker(options):
    # now = datetime.datetime.now()
    # time_str = now.strftime("%y%m%d-%H%M")
    # print(time_str)

    print('************************')
    for k, v in options.items():
        print(k, '=', v)
    print('************************')

    # with open(log_txt_path, 'a') as f:
    #     for k, v in options.items():
    #         f.write(str(k) + '=' + str(v) + '\n')

    # print("*********** MAFW Dataset Fold  " + " ***********")
    # log_txt_path = './log/' + 'MAFW-' + time_str + '-set'  + '-log.txt'
    # checkpoint_path = './checkpoint/' + 'MAFW-' + time_str + '-set'  + '-model.pth'
    # best_checkpoint_path = './checkpoint/' + 'MAFW-' + time_str + '-set'  + '-model_best.pth'

    torch.manual_seed(options['seed'])
    os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu']
    use_gpu = torch.cuda.is_available()
    if options['use_cpu']: use_gpu = False
    
    known = options['known']
    unknown = options['unknown']

    if use_gpu:
        print("Currently using GPU: {}".format(options['gpu']))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(options['seed'])
    else:
        print("Currently using CPU")

    # Dataset
    print("{} Preparation".format(options['dataset']))
    # with open(log_txt_path, 'a') as f:
    #     f.write("{} Preparation".format(options['dataset']))
    if 'AFEW' in options['dataset']:
        train_data = train_data_loader(known, unknown)
        test_data = test_data_loader(known, unknown)
        out_data = out_data_loader(known, unknown)

        trainloader = torch.utils.data.DataLoader(train_data,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.workers,
                                                pin_memory=True)
        testloader = torch.utils.data.DataLoader(test_data,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.workers,
                                                pin_memory=True)
        outloader = torch.utils.data.DataLoader(out_data,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.workers,
                                                pin_memory=True)
    elif 'MAFW' in options['dataset']:
        train_data = train_data_loader(known, unknown)
        test_data = test_data_loader(known, unknown)
        out_data = out_data_loader(known, unknown)

        trainloader = torch.utils.data.DataLoader(train_data,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.workers,
                                                pin_memory=True)
        testloader = torch.utils.data.DataLoader(test_data,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.workers,
                                                pin_memory=True)
        outloader = torch.utils.data.DataLoader(out_data,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.workers,
                                                pin_memory=True)
    elif 'OURDATA' in options['dataset']:
        train_data = train_data_loader()
        test_data = test_data_loader()
        out_data = out_data_loader()

        trainloader = torch.utils.data.DataLoader(train_data,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.workers,
                                                pin_memory=True)
        testloader = torch.utils.data.DataLoader(test_data,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.workers,
                                                pin_memory=True)
        outloader = torch.utils.data.DataLoader(out_data,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.workers,
                                                pin_memory=True)
    elif 'MULTIDATA' in options['dataset']:
        train_data = train_data_loader()
        test_data = test_data_loader()
        out_data = out_data_loader()

        trainloader = torch.utils.data.DataLoader(train_data,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.workers,
                                                pin_memory=True)
        testloader = torch.utils.data.DataLoader(test_data,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.workers,
                                                pin_memory=True)
        outloader = torch.utils.data.DataLoader(out_data,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.workers,
                                                pin_memory=True)
    elif 'DFEW' in options['dataset']:
        train_data = train_data_loader(known, unknown)
        test_data = test_data_loader(known, unknown)
        out_data = out_data_loader(known, unknown)

        trainloader = torch.utils.data.DataLoader(train_data,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.workers,
                                                pin_memory=True)
        testloader = torch.utils.data.DataLoader(test_data,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.workers,
                                                pin_memory=True)
        outloader = torch.utils.data.DataLoader(out_data,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.workers,
                                                pin_memory=True)
    

    options['num_classes'] = len(known)
    options['ids'] = ''.join(str(x) for x in options['known'])

    clip_model, preprocess = clip.load("ViT-B/32") 
    feat_dim = 512
    
    classes_names = list(set(train_data.label_text))
    classes_names.sort()
    options['classes_names']=classes_names
    clip_model = clip_model.cuda()
    net0 = clip_model.cuda() # 同一个 CLIP 模型,net0 只是 clip_model 的引用，它们指向同一块 GPU 上的内存
    # net0 = torch.nn.DataParallel(net0).cuda()

    # 冻结图像编码器的参数
    for param in clip_model.visual.parameters():
        param.requires_grad = False

    # 冻结文本编码器的参数
    for param in clip_model.transformer.parameters():
        param.requires_grad = False
        

    for param in clip_model.token_embedding.parameters():
        param.requires_grad = False


    options['patch_size'] = 56 
    
    normalization = preprocess.transforms[-1]# 从 preprocess 变换列表中提取最后一个变换（通常是归一化）
    prompt_size = options['patch_size']
    prompt = PromptCLIP(prompt_size, clip_model, classes_names)

    # prompt = torch.nn.DataParallel(prompt).cuda()

    # Loss
    options.update(
        {
            'feat_dim': feat_dim,
            'use_gpu':  use_gpu
        }
    )

    Loss = importlib.import_module('loss.'+options['loss'])
    criterion = getattr(Loss, options['loss'])(**options)
    
    prompt_CEloss = torch.nn.CrossEntropyLoss()
    contrastive_loss = ConLoss()

    if use_gpu:
        criterion = criterion.cuda()

    model_path = os.path.join(options['outf'], 'models', options['dataset'])
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    file_name = '{}_{}_{}'.format(options['model'], options['loss'], options['item'])

    if options['eval']:
        prompt, criterion = load_networks(prompt, model_path, file_name, criterion=criterion)
        results = test(prompt, normalization, criterion, testloader, outloader, epoch=0, **options)
        print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'], results['OSCR']))

        return results

    params_list = [{'params': prompt.parameters()},
                {'params': criterion.parameters()}]
    
    optimizer = torch.optim.SGD(params_list, lr=options['lr'], momentum=0.9, weight_decay=1e-4)

    if options['stepsize'] > 0:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90,120])

    start_time = time.time()

    best_score = 0
    best_results = {}
    for epoch in range(options['max_epoch']):
        print("==> Epoch {}/{}".format(epoch+1, options['max_epoch']))
        # with open(log_txt_path, 'a') as f:
        #     f.write("==> Epoch {}/{}".format(epoch+1, options['max_epoch']) + '\n')
        if epoch == 0:
            _, logits_list, [cy,cx] = train(net0, preprocess, prompt, normalization, criterion, prompt_CEloss, contrastive_loss, optimizer, trainloader, epoch=epoch, **options)
            options['patch_position'] = [cy,cx]
        else:
            _, logits_list = train(net0, preprocess, prompt, normalization, criterion, prompt_CEloss, contrastive_loss, optimizer, trainloader, epoch=epoch, **options)

        if options['eval_freq'] > 0 and (epoch+1) % options['eval_freq'] == 0 or (epoch+1) == options['max_epoch']:
            print("==> Test", options['loss'])
            results_e = test(prompt, normalization, criterion, testloader, outloader, epoch=epoch, **options)
            print("Acc_close (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results_e['ACC_CLOSE'], results_e['AUROC'], results_e['OSCR']))
            options['writer'].add_scalar('Acc_close (%)', results_e['ACC_CLOSE'], epoch)
            options['writer'].add_scalar('AUROC (%)', results_e['AUROC'], epoch)
            options['writer'].add_scalar('OSCR (%)', results_e['OSCR'], epoch)

            if results_e['OSCR'] +  results_e['AUROC'] > best_score:
                best_score = results_e['OSCR'] +  results_e['AUROC']
                best_results = results_e
                save_networks(prompt, model_path, file_name, criterion=criterion)
        
        if options['stepsize'] > 0: scheduler.step()

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

    # return results
    return best_results

if __name__ == '__main__':
    args = parser.parse_args()
    options = vars(args)
    # options['dataroot'] = os.path.join(options['dataroot'], options['dataset'])
    img_size = 224
    results = dict()
    
    from split import splits_2024 as splits 
    
    for i in range(len(splits[options['dataset']])): 
        options['writer'] = SummaryWriter(f"results/{options['dataset']}_{i}")
        known = splits[options['dataset']][len(splits[options['dataset']])-i-1]
        print('known classes:', known)
        if options['dataset'] == 'AFEW':
            unknown = list(set(list(range(0, 7))) - set(known))
        elif options['dataset'] == 'MAFW':
            unknown = list(set(list(range(0, 11))) - set(known))
        elif options['dataset'] == 'OURDATA': 
            unknown = [7,8,9,10]
        elif options['dataset'] == 'MULTIDATA': 
            unknown = [7]
        elif options['dataset'] == 'DFEW':
            unknown = list(set(list(range(0, 7))) - set(known))
        else:
            unknown = list(set(list(range(0, 10))) - set(known))

        options.update(
            {
                'item':     i,
                'known':    known,
                'unknown':  unknown,
                'img_size': img_size
            }
        )

        dir_name = '{}_{}'.format(options['model'], options['loss'])
        dir_path = os.path.join(options['outf'], 'results', dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # if options['dataset'] == 'cifar100':
        #     file_name = '{}_{}.csv'.format(options['dataset'], options['out_num'])
        # else:
        #     file_name = options['dataset'] + '.csv'


        # timestamp = datetime.now().strftime("%y%m%d-%H%M")

        if options['dataset'] == 'cifar100':
            file_name = '{}_{}_{}.csv'.format(options['dataset'], options['out_num'], timestamp)
        else:
            file_name = '{}_{}.csv'.format(options['dataset'], timestamp)

        res = main_worker(options)
        res['unknown'] = unknown
        res['known'] = known
        results[str(i)] = res
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(dir_path, file_name))
