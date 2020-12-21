import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data as data
from torchvision import datasets, models, transforms
import os
import time


def train_model(model, criterion, optimizer, num_epochs):
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        time1 = time.time()
        print('-' * 20)
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                inputs,labels = data

                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = torch.argmax(outputs, 1)

                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()


        time2 = time.time()
        print('epoch time: {}s'.format(time2 - time1))


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomCrop((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),   #(H,W,C),(0,225)->(C,H,W),(0.0,1.0)
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])   #image=(image-mean)/std
    ]),
    'val': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.ToTensor(),#（H，W，C），（0，225）->(C,H,W),(0.0->1.0)
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


data_dir = 'train_img'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

model_ft = models.resnet18(pretrained=True)
model_dict = model_ft.state_dict()
print(model_ft)
dict_name = list(model_dict)
for i, p in enumerate(dict_name):
    print(i, p)
for i,p in enumerate(model_ft.parameters()):
    if i < 90:
        p.requires_grad = False

fc_features = model_ft.fc.in_features
model_ft.fc = nn.Linear(fc_features, 2)

model_ft = model_ft.cuda()
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.9,nesterov=True)


train_model(model_ft,criterion,optimizer_ft,50)

torch.save(model_ft,'resnet18_50.pth')
