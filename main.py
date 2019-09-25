import fire
import numpy as np
import torch
import torch.optim as optim

from tqdm import trange
from torch.optim.lr_scheduler import LambdaLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import dataset
import models
import utils

from config import opt

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(**keward):

    setup_seed(9979)

    opt.parse(keward)
    if opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    model = getattr(models, opt.model)(opt)
    if opt.use_gpu:
        model.cuda()

    # ----------------修改处1: 数据加载 ------------------- #
    train_data = datasets.MNIST(root="./dataset", train=True, transform=transforms.ToTensor(), download=True)
    test_data = datasets.MNIST(root="./dataset", train=False, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(1+0.95*epoch))
    train_steps = (len(train_data) + opt.batch_size - 1) // opt.batch_size
    test_steps = (len(test_data) + opt.batch_size - 1) // opt.batch_size

    print("train data num:{}; test data num: {}".format(len(train_data), len(test_data)))
    print("start training...")
    for epoch in range(opt.num_epochs):
        print("{}; epoch:{}/{}; training....".format(utils.now(), epoch, opt.num_epochs))
        model.train()
        scheduler.step()
        dataIterator = enumerate(train_loader)
        lossAll = utils.RunningAverage()
        t = trange(train_steps)
        for i in t:
            # -----------------修改处3: 设置数据----------------#
            idx, data = next(dataIterator)
            x, y = data
            y = utils.to_categorical(y, opt.num_class)
            if opt.use_gpu:
                x = x.cuda()
                y = y.cuda()
            loss = model(x, y)

            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=opt.clip_grad)
            optimizer.step()

            lossAll.update(loss.item())
            t.set_postfix(allLoss='{:05.3f}/{:05.3f}'.format(lossAll(), loss.item()))

        print("evaluate just one image...")
        print("predict train data: ")
        predict_one(model, train_loader, train_steps, opt, epoch)
        print("predict test data: ")
        predict_one(model, test_loader, test_steps, opt, epoch)

        print("evaluate two images...")
        print("predict train data: ")
        predict_two(model, train_loader, train_steps, opt, epoch)
        print("predict test data...")
        predict_two(model, test_loader, test_steps, opt, epoch)

        print('\n')
def predict_one(model, dataLoader, steps, opt, epoch):
    model.eval()
    dataIterator = enumerate(dataLoader)
    t = trange(steps)
    y_true, y_pred = [], []
    for i in t:
        # -----------------修改处3: 设置数据----------------#
        idx, data = next(dataIterator)
        x, y = data
        y_true.extend(y.tolist())
        if opt.use_gpu:
            x = x.cuda()
        out = model(x)
        p = torch.max(out, 1)[1]
        y_pred.extend(p.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    print("single image acc:{}".format(np.sum(y_true == y_pred) / len(y_true)))

def predict_two(model, dataLoader, steps, opt, epoch):
    model.eval()
    dataIterator = enumerate(dataLoader)
    t = trange(steps)

    y_true, y_pred, greater = [], [], []

    for i in t:
        idx, data = next(dataIterator)
        x, y = data
        index = list(range(x.size(0)))
        np.random.shuffle(index)

        x = torch.cat([x, x[index]], 2)
        y = torch.cat([y.unsqueeze(1), y[index].unsqueeze(1)], 1)
        x = x[y[:, 0] != y[:, 1]]
        y = y[y[:, 0] != y[:, 1]]

        y = torch.sort(y, 1)[0]
        y_true.extend(y.tolist())

        if opt.use_gpu:
            x = x.cuda()

        out = model(x)
        g = torch.sort(out, 1)[0][:, -2] > 0.5
        greater.extend(g.tolist())

        p = torch.argsort(out, 1)[:, -2:]
        p = torch.sort(p, 1)[0]
        y_pred.extend(p.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    greater = np.array(greater)

    acc = np.prod(y_true == y_pred, axis=1).sum() / len(y_true)
    print("double image acc(无置信度):{}".format((acc)))
    acc = (np.prod(y_true == y_pred, axis=1) * greater).sum() / len(y_true)
    print("double image acc(有置信度):{}".format((acc)))


if __name__ == '__main__':
    fire.Fire()
