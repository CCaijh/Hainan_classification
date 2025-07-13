from matplotlib import rcParams
import matplotlib.pyplot as plt
import re
import numpy as np

logFile = 'Z:\ZQY\Hainan\Hainan_train_5t\logs\\2025-06-28-14-55_0.log'

text = ''

file = open(logFile)
for line in file:
    text += line
file.close()

loss_list = re.findall('loss=.*[0-9]',text)
lr_list=re.findall('lr=.*[0-9].*?\t',text)
acc_list=re.findall('acc=.*[0-9]',text)


train_loss = []
lr=[]
acc=[]

for i in loss_list:
    train_loss.append(float(i.split('acc')[0].split('=')[1]))
for j in lr_list:
    lr.append((float((j.strip()).split('acc')[0].split('=')[1])))
for k in acc_list:

    acc.append(float(k.split('acc')[1].split('=')[1]))


with open("./train_loss.txt", 'w') as train_los:
    train_los.write(str(train_loss))
with open("./lr.txt",'w') as lr_str:
    lr_str.write(str(lr))
with open("./acc.txt",'w') as acc_str:
    acc_str.write(str(acc))


dir_path = "./train_loss.txt"
dir_path1="./lr.txt"
dir_path2="./acc.txt"

def data_read(dir_path):
    with open(dir_path, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(", ")

    return np.asfarray(data, float)
def data_read(dir_path1):
    with open(dir_path1, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(", ")

    return np.asfarray(data, float)


train_loss_path = r"./train_loss.txt"
lr_path=r"./lr.txt"
acc_path=r"./acc.txt"

y_train_loss = data_read(train_loss_path)
y_lr=data_read(lr_path)
y_acc=data_read(acc_path)
x_train_loss = range(len(y_train_loss))
x_lr=range(len(y_lr))
x_acc=range(len(y_acc))


a1=plt.subplot(311)
a1.set_title("loss")
a1.plot(x_train_loss,y_train_loss,label="train loss",color="blue")

a1=plt.subplot(313)
a1.set_title("acc")
a1.plot(x_acc,y_acc,label="train acc",color="red")


a2=plt.subplot(312)
a2.set_title("lr")
a2.plot(x_lr,y_lr,label="lr",color="black")

plt.tight_layout()
plt.show()


