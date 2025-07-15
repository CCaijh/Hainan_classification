import datetime
import os
import json
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
torch.cuda.current_device()
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import datetime, os, json, random, numpy as np
from pathlib import Path
from model.model_resnet_448_sa import resnet50
from torch.utils.data import Dataset, DataLoader, Sampler
from configs_0_5t import train_configs
from logger_0 import log_creater
from utils import get_params_groups

SEED = train_configs["seed"]  #

# ---------- Fixed seeds ----------
def set_seed(seed: int):
    """Fix all random sources for full reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def worker_init_fn(worker_id: int):
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)
    torch.manual_seed(SEED + worker_id)

class FixedOrderSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices
    def __iter__(self):
        return iter(self.indices)
    def __len__(self):
        return len(self.indices)

def main():


    set_seed(SEED)

    g = torch.Generator().manual_seed(SEED)

    logger = log_creater(train_configs["output_path"])
    logger.info('start training! (seed={})'.format(SEED))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor()]),
        "val": transforms.Compose([transforms.ToTensor()]),
    }

    train_dataset = datasets.ImageFolder(root=os.path.join(train_configs["input_path"],"train"), transform=data_transform["train"])
    train_num = len(train_dataset)

    order_file = Path("train_order.txt")
    if order_file.exists():
        shuffled_idx = [int(line.strip()) for line in order_file.open()]
    else:
        g = torch.Generator().manual_seed(train_configs["seed"])
        shuffled_idx = torch.randperm(len(train_dataset), generator=g).tolist()
        with order_file.open("w") as f:
            for idx in shuffled_idx:
                f.write(f"{idx}\n")
    print(f"fixed order length: {len(shuffled_idx)}")

    SF_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in SF_list.items())
    sf_classes = len(cla_dict)

    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    nw = min([os.cpu_count(), train_configs["batch_size"] if train_configs["batch_size"] > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))


    train_loader = DataLoader(
        train_dataset,
        batch_size=train_configs["batch_size"],
        sampler=FixedOrderSampler(shuffled_idx),
        num_workers=nw,
        worker_init_fn=worker_init_fn
    )

    validate_dataset = datasets.ImageFolder(root=os.path.join(train_configs["input_path"], "val"),
                                            transform=data_transform["val"])

    validate_loader = torch.utils.data.DataLoader(
        validate_dataset,
        batch_size=train_configs["batch_size"],
        shuffle=True,
        num_workers=nw,
        worker_init_fn=worker_init_fn,  # ★
        generator=g)  # ★
    val_num = len(validate_dataset)


    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # model_name = "VIT"
    # model_name="convnext_base"
    # model_name = "swin_base"
    model_name="resnet50"
    net = resnet50(num_classes=sf_classes)
    # net = vit_base_patch16_224_in21k(num_classes=sf_classes)
    net.to(device)
    # total_steps=200

    loss_function = nn.CrossEntropyLoss()
    pg = get_params_groups(net, weight_decay=train_configs["weight_decay"])
    optimizer = optim.SGD(pg, lr=train_configs["lr"], momentum=train_configs["momentum"], nesterov=True, weight_decay=train_configs["weight_decay"])

    scheduler=optim.lr_scheduler.OneCycleLR(optimizer,max_lr=train_configs["max_lr"],pct_start=train_configs["pct_start"],steps_per_epoch=len(train_loader),epochs=train_configs["epochs"],anneal_strategy="cos")


    best_acc = 0.0

    save_path='./weight//shuffle_add_sa_cuda0_layer34_seed56_newpara'
    os.makedirs(save_path, exist_ok=True)
    train_steps = len(train_loader)

    print("the type of model:{}".format(model_name))
    print("the start time of training: {}".format(datetime.datetime.now()))
    logger.info('the type of model:{}'.format(model_name))
    logger.info('the start time of training: {}'.format(datetime.datetime.now()))
    logger.info('batch_size: {}\t max_lr:{}\t weight_decay:{}\t momentum:{}\t pct_start:{}'.format(train_configs["batch_size"],train_configs["max_lr"],train_configs["weight_decay"],train_configs["momentum"],train_configs["pct_start"]))


    for epoch in range(train_configs["epochs"]):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data

            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.8f}".format(epoch + 1,train_configs["epochs"],loss)
            logger.info('Epoch:[{}/{}]\t lr={:.8f}\t loss={:.8f}'.format(epoch + 1, train_configs["epochs"], scheduler.get_last_lr()[0],loss))

        net.eval()
        acc = 0.0  # accumulate accurate number / epoch

        class_correct=list(0. for i in range(sf_classes+1))
        class_total=list(0. for i in range(sf_classes+1))
        with torch.no_grad():
            for data in validate_loader:
                images,labels=data
                outputs=net(images.to(device))
                _,predicted=torch.max(outputs,1)
                c=(predicted == labels.to(device)).squeeze()
                acc += torch.eq(predicted, labels.to(device)).sum().item()
                for i in range(len(labels)):
                    label=labels[i]
                    class_correct[label]+=c[i].item()
                    class_total[label]+=1

        val_accurate = acc / val_num
        print('[epoch %d] train_ave_loss: %.8f  val_accuracy: %.8f' %(epoch + 1, running_loss / train_steps, val_accurate))
        logger.info('Epoch:[{}/{}]\t train_ave_loss={:.8f}\t acc={:.8f}'.format(epoch+1, train_configs["epochs"], running_loss / train_steps, val_accurate))

        for i in range(sf_classes):

            if class_total[i] > 0:
                acc = 100.0 * class_correct[i] / class_total[i]
                print('Accuracy of %5s : %.3f %%' % (cla_dict[i], acc))
                logger.info(
                    'Epoch:[{}/{}]\t Accuracy of {:5s} : {:.3f} %'.format(
                        epoch + 1,
                        train_configs["epochs"],
                        cla_dict[i],
                        acc
                    )
                )
            else:

                print('Accuracy of %5s : ---- (no samples)' % cla_dict[i])
                logger.info(
                    'Epoch:[{}/{}]\t Accuracy of {:5s} : ---- (no samples)'.format(
                        epoch + 1,
                        train_configs["epochs"],
                        cla_dict[i]
                    )
                )

        if val_accurate > best_acc:
           best_acc = val_accurate
           torch.save(net.state_dict(), save_path+'/'+'{}Net_{}_best.pth'.format(model_name,(datetime.datetime.now()).strftime("%Y%m%d")))

        if (epoch+1) % 10 == 0:
            torch.save(net.state_dict(), save_path+'/'+'{}Net_{}_{}.pth'.format(model_name,epoch+1,(datetime.datetime.now()).strftime("%Y%m%d")))

    print("the end time of training: {}".format(datetime.datetime.now()))
    print('Finished Training')
    logger.info("the end time of training: {}".format(datetime.datetime.now()))
    logger.info('finish training!')



if __name__ == '__main__':
    main()
