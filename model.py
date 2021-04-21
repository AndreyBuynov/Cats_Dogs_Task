import torch.nn as nn

class My_Cnn(nn.Module):
  
    def __init__(self, n_classes = 2):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)            
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(2, 2), bias=False),
                nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(2, 2), bias=False),
                nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(2, 2), bias=False),
                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
        )

        self.fc_cl = nn.Sequential(
            nn.Dropout(p=0.3, inplace=False),
            nn.Linear(in_features=512*4*4, out_features=128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=1, bias=True)
        )
        
        self.fc_xmin = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=512*4*4, out_features=256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=1, bias=True)
        )
        
        self.fc_ymin = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=512*4*4, out_features=256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=1, bias=True)
        )
        
        self.fc_xmax = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=512*4*4, out_features=256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=1, bias=True)
        )
        
        self.fc_ymax = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=512*4*4, out_features=256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=1, bias=True)
        )
  
  
    def forward(self, x):
        print('x in: ', x.shape)
        x = self.conv1(x)
        print('x from 1 layer : ', x.shape)
        x = self.conv2(x)
        print('x from 2 layer : ', x.shape)
        x = self.conv3(x)
        print('x from 3 layer : ', x.shape)
        x = self.conv4(x)
        print('x from 4 layer : ', x.shape)
        x = x.view(x.size(0), -1)
        print('x after view : ', x.shape)
        cl = self.fc_cl(x)
        xmin = self.fc_xmin(x)
        ymin = self.fc_ymin(x)
        xmax = self.fc_xmax(x)
        ymax = self.fc_ymax(x)
        
        return xmin, ymin, xmax, ymax, cl