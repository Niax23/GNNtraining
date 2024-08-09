import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.seed import seed_everything
import json
import time
import os
from torch.optim import Adam
import argparse
from ogb.graphproppred import Evaluator
from torch.nn import BCEWithLogitsLoss
import torch
from torch.optim import Adam
from modules.model import MyGNN
from modules.DataLoading import pyg_dataset
from torch.nn import BCEWithLogitsLoss








cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()


def get_work_dir(args):
    return args.dataset


def get_file_name(args):
    file_name = [(f'bs_{args.batch_size}')]
    file_name.append(f'lr_{args.lr}')
    file_name.append(f'dp_{args.drop_ratio}')
    file_name.append(f'ep{args.epochs}')
    file_name.append(f'layernum{args.num_layer}')
    current_time = time.time()
    file_name.append(f'{current_time}')
    return '-'.join(file_name) + '.json', current_time


def train(model, device, loader, optimizer, task_type):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type: 
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss.backward()
            optimizer.step()

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)





def init_args():
    parser = argparse.ArgumentParser('Parser For Experiment on OGB')

    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')

    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                        help='dataset name (default: ogbg-molhiv)')

    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    
    parser.add_argument(
        '--exp_name', type=str, default='',
        help='the name of logging file'
    )

    args = parser.parse_args()

    args.work_dir = get_work_dir(args)
    if args.exp_name == '':
        args.exp_name, args.time = get_file_name(args)
    return args



if __name__ == '__main__':
    args = init_args()
    print(args)
    if not os.path.exists('log'):
        os.mkdir('log')
    if not os.path.exists(os.path.join('log', args.work_dir)):
        os.makedirs(os.path.join('log', args.work_dir))


    dataset = pyg_dataset(args.dataset)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    # define the model
    model = MyGNN(num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)


    # dataset and dataloader

    evaluator = Evaluator(args.dataset)
    data_split_idx = dataset.get_idx_split()
    train_idx = data_split_idx['train']
    valid_idx = data_split_idx['valid']
    test_idx = data_split_idx['test']

    train_dataset = dataset[train_idx]
    valid_dataset = dataset[valid_idx]
    test_dataset = dataset[test_idx]

    train_loader = DataLoader(dataset[data_split_idx["train"]], batch_size=32, shuffle=True)
    valid_loader = DataLoader(dataset[data_split_idx["valid"]], batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset[data_split_idx["test"]], batch_size=32, shuffle=False)

    # optimizer and loss
    optimizer = Adam(model.parameters(), lr=args.lr)
    optimizer = Adam(model.parameters(), lr=0.001)

    cls_criterion = torch.nn.BCEWithLogitsLoss()
    reg_criterion = torch.nn.MSELoss()

    valid_curve = []
    test_curve = []
    train_curve = []

    best_valid_score = None  # 初始化最佳验证分数
    best_model_state = None  # 用于保存最佳模型的参数

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train(model, device, train_loader, optimizer, dataset.task_type)

        print('Evaluating...')
        train_perf = eval(model, device, train_loader, evaluator)
        valid_perf = eval(model, device, valid_loader, evaluator)
        test_perf = eval(model, device, test_loader, evaluator)

        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])

        print("cur valid score: {}".format(valid_perf[dataset.eval_metric]))
        print("cur epoch: {}".format(epoch))


        if best_valid_score is None or valid_perf[dataset.eval_metric] > best_valid_score:
            best_valid_score = valid_perf[dataset.eval_metric]
            best_model_state = model.state_dict()  # 保存当前最优模型的参数
            

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    print('Finished training!')
    print('Best epoch: {}'.format(best_val_epoch + 1))
    print('Train score: {}'.format(train_curve[best_val_epoch]))
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))



    with open(os.path.join('log', args.work_dir, args.exp_name), 'w') as Fout:
        json.dump({
        'best_epoch': int(best_val_epoch + 1),
        'config': args.__dict__,
        'train': train_curve,
        'valid': valid_curve,
        'test': test_curve,
        'best': [
            int(best_val_epoch + 1),
            train_curve[best_val_epoch],
            valid_curve[best_val_epoch],
            test_curve[best_val_epoch]
            ],
        }, Fout)
        


    if not args.filename == '':
        torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch], 'Train': train_curve[best_val_epoch], 'BestTrain': best_train}, args.filename)






