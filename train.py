import os.path
from os.path import exists

import numpy as np
import torch
from skimage.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms as tfs
import torchvision.models as models

from modules.model.multiple import MultipleModel
from modules.model.resnet import ResnetModel
from modules.model.vgg import VggModel
from modules.util.data_utils import Fer2013
from modules.model.fer import FER2013

cel = []
#nn.CrossEntropyLoss(weight = torch.tensor([0.496, 0.989, 0.482, 0.05, 0.381, 0.61, 0.362]).to(device) )
cel.append( [0.496*25, 0.989, 0.482, 0.05, 0.381, 0.61, 0.362] )
cel.append([0.496, 0.989*25, 0.482, 0.05, 0.381, 0.61, 0.362])
cel.append([0.496, 0.989, 0.482*25, 0.05, 0.381, 0.61, 0.362])
cel.append([0.496, 0.989, 0.482, 0.05*25, 0.381, 0.61, 0.362])
cel.append([0.496, 0.989, 0.482, 0.05, 0.381*25, 0.61, 0.362])
cel.append([0.496, 0.989, 0.482, 0.05, 0.381, 0.61*25, 0.362])
cel.append([0.496, 0.989, 0.482, 0.05, 0.381, 0.61, 0.362*25])

print("Version:", torch.__version__)

def train_model(model, train_dl, val_dl, ii, epochs=10, lr=0.001, device="cpu", cel=[]):
    best = 0.0
    parameters = filter(lambda p: p.requires_grad, model.parameters())
   
    if  (model.__class__.__name__ == "ResnetModel" or True):
        print("Resnet")
        optimizer = torch.optim.Adam(parameters, lr=lr, betas=(0.9, 0.999), eps=0.001)
    else:
        print("VGG")
        #optimizer = torch.optim.Adam(parameters, lr=lr, betas=(0.9, 0.999))
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    if ii == -1:    
        loss_func = nn.CrossEntropyLoss()
    else:
        loss_func = nn.CrossEntropyLoss(weight = torch.tensor(cel).to(device) )

    for i in range(epochs):                     

        model.train()
        sum_loss = 0.0
        total = 0
        for x, y in train_dl:
            x = x.float().to(device)
            y = y.long().to(device)
            y_pred = model(x)
            loss = loss_func(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.item() * y.shape[0]
            total += y.shape[0]

        val_loss, val_acc, val_rmse = validation_train(model, val_dl, device)
        if val_acc > best:
            if ii == -1:
                torch.save(multiple.state_dict(), "halfface_multiple.pth") 
            else:    
                torch.save(model.state_dict(), "halfface_vgg" + str(ii) + ".pth")
        
        print("epoch %d train loss %.3f, val loss %.3f, val accuracy %.3f, and val rmse %.3f" % (
                    i, sum_loss / total, val_loss, val_acc, val_rmse))

            

def validation_train(model, valid_dl, device="cpu"):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    sum_rmse = 0.0
    loss_func = nn.CrossEntropyLoss()
    for x, y in valid_dl:
        x = x.float().to(device)
        y = y.long().to(device)
        y_hat = model(x)
        loss = loss_func(y_hat, y)
        pred = torch.max(y_hat, 1)[1]
        correct += (pred == y).float().sum()
        total += y.shape[0]
        sum_loss += loss.item() * y.shape[0]
        sum_rmse += np.sqrt(mean_squared_error(pred.cpu().numpy(), y.cpu().numpy())) * y.shape[0]
    return sum_loss / total, correct / total, sum_rmse / total


def validate_val(model, valid_dl, device="cpu"):
    model.eval()
    # Initialize the prediction and label lists(tensors)
    predlist = torch.zeros(0, dtype=torch.long, device=device)
    lbllist = torch.zeros(0, dtype=torch.long, device=device)

    for x, y in valid_dl:
        x = x.float().to(device)
        y = y.long().to(device)
        y_hat = model(x)
        y_hat = torch.argmax(y_hat, dim=1)
        # print("y_hat: {0}; y: {1}".format(y_hat.shape, y.shape))
        # Append batch prediction results
        predlist = torch.cat([predlist, y_hat.view(-1)])
        lbllist = torch.cat([lbllist, y.view(-1)])
        # print(lbllist.shape)
        # print(predlist.shape)

    # Confusion matrix
    conf_mat = confusion_matrix(lbllist.cpu().detach().numpy(), predlist.cpu().detach().numpy())
    print(conf_mat)

    # Per-class accuracy
    class_accuracy = 100 * conf_mat.diagonal() / conf_mat.sum(1)
    print(class_accuracy)


if __name__ == "__main__":
    # setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using {0}".format(device))

    if not exists("datasets/train_all.csv"):
        print("Dataset does not exist... loading")
        fer2013 = Fer2013()
        fer2013.download()

    train_preprocess = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_ds = FER2013("datasets", phase="train", transform=train_preprocess)
    print(train_ds.__len__())

    val_preprocess = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    val_ds = FER2013("datasets", phase="val", transform=train_preprocess)
    print(val_ds.__len__())
    test_ds = FER2013("datasets", phase="test", transform=train_preprocess)
    print(test_ds.__len__())

    batch_size = 1
    learning_rate = 0.001
    epochs = 30

    train_loader = DataLoader(train_ds, batch_size)
    validation_loader = DataLoader(val_ds, batch_size)
    test_loader = DataLoader(test_ds, batch_size)

    
    
    #resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)
    #resnet = ResnetModel().to(device)
    
    '''
    if not os.path.exists("halfface_resnet.pth"):
        print("Resnet2")
        train_model(resnet, train_loader, validation_loader, epochs=epochs, device=device)

        validate_val(resnet, test_loader, device)
        # val_metrics(model, validation_loader)

        torch.save(resnet.state_dict(), "halfface_resnet.pth")
    else:
        resnet.load_state_dict(torch.load("halfface_resnet.pth"))
    
    '''
    vgg = []
    for i in range(0, 7):
        vgg.append( VggModel().to(device) )
        if not os.path.exists("halfface_vgg" + str(i) + ".pth"):
            print("Vgg" + str(i) )
            train_model(vgg[i], train_loader, validation_loader, i, epochs=epochs, device=device, cel=cel[i])

            validate_val(vgg[i], test_loader, device)
            # val_metrics(model, validation_loader)

            #torch.save(vgg[i].state_dict(), "halfface_vgg" + str(i) + ".pth")
        else:
            vgg[i].load_state_dict(torch.load("halfface_vgg" + str(i) + ".pth"))
    
    print("MultipleModel")
    if not os.path.exists("halfface_multiple.pth"):
        multiple = MultipleModel(vgg=vgg).to(device)
        train_model(multiple, train_loader, validation_loader, -1, epochs=epochs, device=device)

        validate_val(multiple, test_loader, device)
        # val_metrics(model, validation_loader)

        
