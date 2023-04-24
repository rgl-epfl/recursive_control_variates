import torch
import torchvision
from torchvision.models import vgg19
import drjit as dr

class VGGLoss():
    def __init__(self):

        # Load VGG19
        vgg = vgg19(pretrained=True).cuda()

        # Feature extraction layers, as in [Hu et al. 2022]
        self.features = [
            torch.nn.Sequential(vgg.features[:4]), # relu1_2
            torch.nn.Sequential(vgg.features[4:9]), # relu2_2
            torch.nn.Sequential(vgg.features[9:14]), # relu3_2
            torch.nn.Sequential(vgg.features[16:23]) # relu4_2
        ]

        # Disable gradients of the model weights
        for f in self.features:
            for p in f.parameters():
                p.requires_grad = False

        # Standardization factors for VGG19
        self.preprocess = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Last reference used for loss computation (to avoid recomputing features)
        self.target_features = []
        self.target_index = -1 # JIT index of the target image

    @dr.wrap_ad(source="drjit", target="torch")
    def compute_target_features(self, target):
        with torch.no_grad():
            p_img = self.preprocess(target.T[None, ...])
            for f in self.features:
                p_img = f(p_img)
                self.target_features.append(p_img)

    @dr.wrap_ad(source="drjit", target="torch")
    def loss(self, source, enable_grad=True):
        # Temporary fix to avoid GPU memory leaks due to wrap_ad in detached mode
        source.requires_grad = enable_grad

        # Standardize the image
        p_img = self.preprocess(source.T[None, ...])
        loss = 0
        for i, f in enumerate(self.features):
            p_img = f(p_img)
            loss += torch.mean((p_img - self.target_features[i])**2)

        if enable_grad:
            return loss
        # Temporary fix to avoid GPU memory leaks due to wrap_ad
        return loss.cpu()

    def __call__(self, source, target):
        if target.index != self.target_index:
            self.target_index = target.index
            self.compute_target_features(target)
        return self.loss(source, enable_grad=dr.grad_enabled(source))
