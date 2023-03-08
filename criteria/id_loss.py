import torch
from torch import nn
from models.e4e.encoders.model_irse import Backbone
from decalib.datasets.detectors import MTCNN

class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load('/data/pzh/GAN-Geometry/pretrained_models/model_ir_se50.pth'))
        self.facenet.cuda()
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        for module in [self.facenet, self.face_pool]:
            for param in module.parameters():
                param.requires_grad = False
        self.face_detector = MTCNN(device='cuda')
                
    def align(self,img_tensor):
        aligned_face = self.face_detector.run(img_tensor)
        return aligned_face           
                

    def extract_feats(self, x):
        x = torch.nn.functional.interpolate(x, size=(256, 256))
        try:
            x = self.align(x)
        except :
            x = x[:, :, 35:223, 32:220]  # Crop interesting region
            x = self.face_pool(x)
            print('*'*59)
        # x = x[:, :, 35:223, 32:220]  # Crop interesting region
        # x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y, x):
        n_samples = x.shape[0]
        x = torch.nn.functional.interpolate(x, size=(256, 256))
        y = torch.nn.functional.interpolate(y, size=(256, 256))
        x_feats = self.extract_feats(x)
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there

        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        sim_improvement = 0
        id_logs = []
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            diff_input = y_hat_feats[i].dot(x_feats[i])
            diff_views = y_feats[i].dot(x_feats[i])
            id_logs.append({'diff_target': float(diff_target),
                            'diff_input': float(diff_input),
                            'diff_views': float(diff_views)})
            loss += 1 - diff_target
            id_diff = float(diff_target) - float(diff_views)
            sim_improvement += id_diff
            count += 1

        return loss / count, sim_improvement / count, id_logs
    
    
    
    # class IDLoss(nn.Module):
    #     def __init__(self):
    #     super(IDLoss, self).__init__()
    #     print('Loading ResNet ArcFace')
    #     self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
    #     self.facenet.load_state_dict(torch.load('../pretrained_models/model_ir_se50.pth'))
    #     self.facenet.cuda()
    #     self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
    #     self.facenet.eval()
    #     for module in [self.facenet, self.face_pool]:
    #         for param in module.parameters():
    #             param.requires_grad = False
    #     self.face_detector = MTCNN(device='cuda')
                
    
    # def align(self,img_tensor):
        
    #     aligned_face = self.face_detector.run(img_tensor)
        
        
    #     return aligned_face

    # def forward(self, x):
    #     x = torch.nn.functional.interpolate(x, size=(256, 256))
    #     try:
    #         x = self.align(x)
    #     except :
    #         x = x[:, :, 35:223, 32:220]  # Crop interesting region
    #         x = self.face_pool(x)
    #     x_feats = self.facenet(x)
    #     return x_feats