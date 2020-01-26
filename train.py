import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from progressbar import ProgressBar

from data_module import DIV2K_x2, RandomHorizontalFlip, RandomVerticalFlip, Normalize, ToTensor, Compose
from EDSR import edsr

train_dir = 'data/train'
val_dir = 'data/validation'

train_transforms = Compose([RandomHorizontalFlip(p=0.5),
                            RandomVerticalFlip(p=0.5),
                            ToTensor(),
                            Normalize([0.449, 0.438, 0.404],
                                      [1.0, 1.0, 1.0])])

valid_transforms = Compose([RandomHorizontalFlip(p=0.5),
                            RandomVerticalFlip(p=0.5),
                            ToTensor(),
                            Normalize([0.440, 0.435, 0.403],
                                      [1.0, 1.0, 1.0])])

trainset = DIV2K_x2(root_dir=train_dir, im_size=40, scale=2, transform=train_transforms)
validset = DIV2K_x2(root_dir=val_dir, im_size=40, scale=2, transform=valid_transforms)

trainloader = DataLoader(trainset, batch_size=4, shuffle=True)
validloader = DataLoader(validset, batch_size=4, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = edsr().to(device)

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))

epochs = 100
print_every = 25
train_loss = 0
batch_num = 0

for epoch_num in range(epochs):
    for img, label in trainloader:

        optimizer.zero_grad()
        pred = model(img)
        # print(pred.shape, label.shape)
        batch_num += 1
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if batch_num % print_every == 0:
            print('Training Loss: {:.4f}'.format(train_loss / print_every))

    with torch.no_grad():
        val_loss = 0
        model.eval()
        for val_ims, val_lbs in validloader:
            test_pred = model(val_ims)
            vloss = criterion(test_pred, val_lbs)
            val_loss += vloss.item()

        print('Epoch : {}/{}'.format(epoch_num, epochs))
        print('Training Loss : {:.4f}'.format(train_loss / print_every))
        print('Validation Loss: {:.4f}'.format(val_loss / len(validloader)))
        train_loss = 0
        model.train()
