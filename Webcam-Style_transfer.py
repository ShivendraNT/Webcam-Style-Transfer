#Imports
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as pltx

device=torch.device("mps" if torch.mps.is_available() else "cpu")



#VGG-style normalization (ImageNet)
IMAGENET_MEAN=torch.tensor([0.485,0.456,0.406],device=device)
IMAGENET_STD=torch.tensor([0.229, 0.224, 0.225],device=device)

def bgr_to_torch(img_bgr,size=512,normalize=True,device=device):
    '''
    img_bgr: numpyy array (H,W,3) in BGR order, uint8 [0..255]
    returns: torch tensor(1,3,H,W), float32,RGB, optionally normalized
    '''

    #Resize (keep aspect by shortest side or force square - for demo, force square)
    img_bgr=cv2.resize(img_bgr,(size,size))

    #BGR -> RGB
    img_rgb=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)

    # to float (0..1)
    img=img_rgb.astype(np.float32)/255.0

    #HWC -> CHW
    img=np.transpose(img,(2,0,1))     #(3,H,W)

    #to torch
    img_t=torch.from_numpy(img).to(device)

    if normalize:
        #(x-mean)/std per channel
        mean=IMAGENET_MEAN.view(3,1,1)
        std=IMAGENET_STD.view(3,1,1)
        img_t=(img_t-mean)/std

    # add batch dim: (1,3,H,W)
    return img_t.unsqueeze(0)

def torch_to_bgr(img_t,denormalize=True):
    '''
    img_t: (1,3,H,W) RGB float tensor,maybe normalized
    returns: numpy BGR uint8 (H,W,3) for OpenCV imshow/imwrite
    '''
    img =img_t.detach().clone().squeeze(0)      #(3,H,W)

    if denormalize:
        mean=IMAGENET_MEAN.view(3,1,1)
        std=IMAGENET_STD.view(3,1,1)
        img=img*std + mean
    
    #Clamp to [0,1]
    img=img.clamp(0,1)

    #CHW -> HWC
    img=img.permute(1,2,0).cpu().numpy()       #(H,W,3)  RGB

    #[0..255] and to uint8
    img=(img*255.0).astype(np.uint8)

    #RGB -> BGR for OpenCV
    img_bgr= cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    return img_bgr

        
    

#=========================================================================
#                   Style Transfer Tools
#=========================================================================



# Helper functions
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        return out + residual

class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = torch.nn.functional.interpolate(x, scale_factor=self.upsample)
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

    
def clean_state_dict(state_dict):
    # Remove old InstanceNorm running stats that are no longer used
    for key in list(state_dict.keys()):
        if "running_mean" in key or "running_var" in key:
            del state_dict[key]
    return state_dict



class TransformerNet(torch.nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)

        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)

        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        y = self.relu(self.in1(self.conv1(x)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y


    

class Normalization(nn.Module):
    def __init__(self,mean,std):
        super().__init__()
        self.mean=mean.view(-1,1,1)
        self.std=std.view(-1,1,1)
    def forward(self,x):
        return (x-self.mean)/self.std

def gram_matrix(x):
    # Handle both 3D and 4D tensors
    if x.dim() == 4:
        # 4D tensor: [batch, channels, height, width]
        B, C, H, W = x.size()
        features = x.view(B, C, H * W)
        G = torch.bmm(features, features.transpose(1, 2))  # [B, C, C]
        return G / (C * H * W)
    elif x.dim() == 3:
        # 3D tensor: [channels, height, width] - add batch dimension
        C, H, W = x.size()
        features = x.view(C, H * W)  # [C, H*W]
        G = torch.mm(features, features.t())  # [C, C]
        return G / (C * H * W)
    else:
        raise ValueError(f"Expected 3 or 4 dimensional tensor, got {x.dim()}D")
    


def total_variation_loss(img, tv_weight=1e-4):
    # img: (1,3,H,W)
    dx = img[:, :, :, 1:] - img[:, :, :, :-1]
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    return tv_weight * (dx.abs().mean() + dy.abs().mean())





def optimize_step(model,generated,optimizer,content_loss,style_loss,
                  content_weight=1.0,style_weight=1e3,tv_weight=1e-4):
    optimizer.zero_grad(set_to_none=True)
    model(generated)               #Forward populated the loss modules

    c_loss=sum(cl.loss for cl in content_loss)
    s_loss=sum(sl.loss for sl in style_loss)
    tv=total_variation_loss(generated,tv_weight)

    loss=content_weight*c_loss + style_weight*s_loss + tv

    loss.backward()
    optimizer.step()
    return loss.item(), c_loss.item(), s_loss.item()


def prepare_style(style_path,device,size=512):
    style_bgr=cv2.imread(style_path)
    assert style_bgr is not None,f'Style not found: {style_path}'
    return bgr_to_torch(style_bgr,size=512,normalize=True,device=device)




#===========================================================================
#                  WEBCAM REALTIME LOOP
#===========================================================================

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))   # scale [0,1] → [0,255]
])

def deprocess(tensor):
    img = tensor.clone().detach().cpu().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")  # CxHxW → HxWxC
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)    # convert to OpenCV’s BGR
    return img

def run_webcam_style(models, device):
    cap = cv2.VideoCapture(0)  # open webcam
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    style_names=list(models.keys())
    current_style=style_names[0]
    model=models[current_style].to(device).eval()

    print("✅ Press 1/2/3/4 to switch styles | 'q' to quit")
    print(f"🎨 Current style: {current_style}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- Preprocess frame ---
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # OpenCV → RGB
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        # --- Apply style ---
        with torch.no_grad():
            output = model(img_tensor).cpu()[0]

        # --- Deprocess back to image ---
        styled_frame = deprocess(output)
        styled_frame=cv2.flip(styled_frame,1)
        frame=cv2.flip(frame,1)

        # --- Show side by side ---
        combined = cv2.hconcat([frame, styled_frame])
        cv2.imshow("Webcam | Original (left) vs Stylized (right)", combined)

        key=cv2.waitKey(1) & 0xFF
        if key==ord('q'):
            break
        elif key in [ord('1'),ord('2'),ord('3'),ord('4')]:
            idx=int(chr(key))-1
            current_style = style_names[idx]
            model = models[current_style].to(device).eval()
            print(f"🎨 Switched to: {current_style}")


    cap.release()
    cv2.destroyAllWindows()



models={
    "mosaic": TransformerNet(),
    "candy": TransformerNet(),
    "rain_princess": TransformerNet(),
    "udnie": TransformerNet()
}

state_dict=torch.load("saved_models/mosaic.pth", map_location=device)
state_dict = clean_state_dict(state_dict)
models["mosaic"].load_state_dict(state_dict,strict=False)
state_dict=torch.load("saved_models/candy.pth", map_location=device)
state_dict = clean_state_dict(state_dict)
models["candy"].load_state_dict(state_dict,strict=False)
state_dict=torch.load("saved_models/rain_princess.pth", map_location=device)
state_dict = clean_state_dict(state_dict)
models["rain_princess"].load_state_dict(state_dict,strict=False)
state_dict=torch.load("saved_models/udnie.pth", map_location=device)
state_dict = clean_state_dict(state_dict)
models["udnie"].load_state_dict(state_dict,strict=False)



if __name__=='__main__':
    run_webcam_style(models,device)