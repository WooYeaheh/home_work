import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from WACV.AFS_dataset import AFS_dataset_test
import os
device = 'cuda:0'
from model_load import model_loader

model_name = 'Diff_attention'
weight_name = 'diff2 cos'
best_epoch = 43
feats_name = 'feat_2'
def reshape(x, h=14, w=14):
    x = x[:, 1:, :]                       # [B,196,768]
    return x.transpose(1,2).reshape(x.size(0), 768, h, w)

class Custom_GradCam(GradCAM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(
            self, input_tensor: torch.Tensor, targets, eigen_smooth: bool = False
    ) -> np.ndarray:

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        self.outputs = outputs = self.activations_and_grads(input_tensor)

        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(category) for category in target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output) for target, output in zip(targets, outputs[feats_name])])
            if self.detach:
                loss.backward(retain_graph=True)
            else:
                torch.autograd.grad(loss, input_tensor, retain_graph=True, create_graph=True)
            if 'hpu' in str(self.device):
                self.__htcore.mark_step()

        cam_per_layer = self.compute_cam_per_layer(input_tensor, targets, eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    model = model_loader(model_name)
    model = model.to(device)
    model.load_state_dict(
        torch.load(f'D:/antispoof/result/{weight_name}/model_ckpt_{best_epoch}.pth', weights_only=True)['net'],strict=False)
    testDataloader = torch.utils.data.DataLoader(AFS_dataset_test, batch_size=1, shuffle=False, num_workers=0)
    model.eval()
    for p in model.parameters():p.requires_grad = True
    for batch_idx, (rgb, ir, depth, labels) in enumerate(testDataloader):
        inputs1, inputs2, inputs3, targets = rgb.to(device), ir.to(device), depth.to(device), labels.to(device)

        if not os.path.exists(f'D:/antispoof/grad_cam/{weight_name}/{feats_name}/{batch_idx + 1}/'): os.makedirs(
            f'D:/antispoof/grad_cam/{weight_name}/{feats_name}/{batch_idx + 1}/')

        fig, axes = plt.subplots(1, len(model.ViT_Encoder)-2)
        for idx in range(1,len(model.ViT_Encoder)-1):
            target_layer = model.ViT_Encoder[idx].ln_1

            cam = Custom_GradCam(model=model, target_layers=[target_layer], reshape_transform=reshape)

            targets = [ClassifierOutputTarget(labels.item())]  # class index 0 or use prediction
            grayscale_cam = cam(input_tensor=(inputs1,inputs2), targets=targets)
            grayscale_cam = grayscale_cam[0]

            visualization = show_cam_on_image(np.clip(np.float32(rgb.squeeze().permute(1, 2, 0)),0,1), grayscale_cam, use_rgb=True)

            axes[idx-1].imshow(visualization)
            axes[idx-1].axis("off")
        fig.savefig(f'D:/antispoof/grad_cam/{weight_name}/{feats_name}/{batch_idx + 1}/grad_cam.jpg')
        plt.close(fig)