import cv2
import glob
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import time
import PIL.Image as Image

img_path = r'./test_img/Smoke-3'
imgs_name = glob.glob('{}/*.jpg'.format(img_path))
imgs_name.sort()
model_ft = torch.load('resnet18_50.pth')
model_ft = model_ft.cuda()
model_ft.train(False)
data_transforms = {
    'test': transforms.Compose([
        # transforms.Resize(256),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),}
smoke_add = []
for img_name in imgs_name:
    num = 3
    img0 = cv2.imread(img_name)#读取彩色图片
    img = cv2.resize(img0, (224 * num, 224 * num))#缩放

    detection_results = []
    crop = []
    all = np.zeros(((2*num-1)*(num+1) + 1,3,224,224))
    start_time = time.time()#设置初始时间

    a = 0
    for i in range(2*num-1):#在整张图片上做滑动窗口检测
        for j in range(num+1):
            crop = img[i * 112: (i + 2) * 112, j * 112: (j + 2) * 112, :]
            crop = Image.fromarray(crop.astype('uint8')).convert('RGB')
            crop = data_transforms['test'](crop)
            all[a,:,:,:]=crop
            a = a + 1
    img1 = cv2.resize(img0,(224,224))
    img1 = Image.fromarray(img1.astype('uint8')).convert('RGB')
    img1 = data_transforms['test'](img1)

    all[a,:,:,:] = img1
    all = torch.from_numpy(all).type(torch.FloatTensor)
    all = Variable(all.cuda())
    outputs = model_ft(all)
    outputs = outputs.cpu().detach().numpy()
    outputs0 = outputs[:,0]
    outputs1 = outputs[:,1]
    results = outputs1 - outputs0
    smoke_result_1_0 = np.max(results)

    end_time = time.time()#记录结束时间
    use_time = end_time - start_time
    print(use_time)

    if smoke_result_1_0 > 1:
        smoke = 1
    else:
        smoke = 0


    smoke_add.append(smoke)

    # 统计最近五帧检测出烟雾的帧数，五帧内有四帧检测出烟雾，则在界面显示‘Smoke’,否则显示‘No smoke’
    if len(smoke_add) > 5:
        smoke_add = smoke_add[-5:]
    add = np.sum(smoke_add)
    if add > 4:
        smoke_result = 1
    else:
        smoke_result = 0


    cv2.putText(img0,#关于文字的显示的设置
                '{0}'.format('smoke' if smoke_result_1_0 > 1.0 else 'no smoke'),
                (20, 60),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                2)

    cv2.imshow('', img0)#显示图像
    key = cv2.waitKey(5)#设置每张照片显示五毫秒
