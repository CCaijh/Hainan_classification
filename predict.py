import os
import json
from shutil import move, rmtree, copy


import torch
from PIL import Image
from torchvision import transforms
import datetime
# from efficient import efficientnet_b1
# from model_resnet_x import resnet50X
# from model_resnet_x import resnet34X
# from configs import train_configs
# from new_model import Model
# from conv import convnext_base
# from Swim_transformer import swin_base_patch4_window7_224_in22k
# from vit_model import vit_base_patch16_224_in21k
# from model.model_vgg_448 import vgg
#from MobileNet_v2 import MobileNetV2
# from model_resnet_448_3 import resnet34
from model_resnet_448 import resnet50
# from model.model_resnet_modified import resnet34_modified
#from model.residual_attention_network import ResidualAttentionModel_448input as ResidualAttentionModel
from tqdm import tqdm  # 引入tqdm库


def mk_file(file_path: str):

    if os.path.exists(file_path):
        rmtree(file_path)
    os.makedirs(file_path)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([transforms.ToTensor()])

    # load image
    imgs_root = "Z:\CJH\ConvTRM_code_exp\Predict\\result\\5_32_4_33\\results-2016"

    assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."

    img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if (i.endswith(".PNG") or i.endswith('.png'))]
    img_path_list.sort()

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' dose not exist."

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)


    for k, v in class_indict.items():
        mk_file(os.path.join(imgs_root, v))

    # create model

    # model = vgg(model_name="vgg16", num_classes=5).to(device)
    #model = ResidualAttentionModel().to(device)
    model = resnet50(num_classes=5).to(device)
    # model = MobileNetV2(num_classes=5).to(device)

    # load model weights

    # weights_path = "./vgg16Net.pth"
    # weights_path = "./last_model_92_sgd_25_5.pkl"
    weights_path = r"Z:\ZQY\Hainan\Hainan_train_5t\weight_qy\noshuffle_ccv1\resnet50Net_20250624_best.pth"
    # weights_path = "./epoch_200.pth"

    assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction+
    model.eval()
    fid = open('./2016_convtrm_resnet50Net_99.pth.txt'.format(datetime.datetime.now().strftime("%Y%m%d")),'w')
    with torch.no_grad():
        for img_path in tqdm(img_path_list, desc="Processing images"):  # 使用tqdm包装img_path_list以显示进度条
            assert os.path.exists(img_path), f"file: '{img_path}' does not exist."
            img_list = []
            img = Image.open(img_path)
            #img = img.resize((448, 448))
            img = data_transform(img)

            img_list.append(img)

            # batch img batch is always 1
            # Package all the images in the img_list into one batch
            batch_img = torch.stack(img_list, dim=0)
            # predict class
            output = model(batch_img.to(device)).cpu()
            predict = torch.softmax(output, dim=1)
            probs, classes = torch.max(predict, dim=1)

            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                fid.writelines("{}\t{}\t{:.3}\n".format(img_path,
                                                        class_indict[str(cla.numpy())],
                                                        pro.numpy()))
               


if __name__ == '__main__':
    main()
