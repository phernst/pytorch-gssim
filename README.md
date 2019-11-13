# pytorch-gssim

### Differentiable gradient structural similarity (GSSIM) index.
![einstein](https://raw.githubusercontent.com/Po-Hsun-Su/pytorch-ssim/master/einstein.png) ![Max_ssim](https://raw.githubusercontent.com/Po-Hsun-Su/pytorch-ssim/master/max_ssim.gif)

## Installation
1. Clone this repo.
2. Copy "pytorch_gssim" folder in your project.

## Example
### basic usage
```python
import pytorch_gssim
import torch
from torch.autograd import Variable

img1 = Variable(torch.rand(1, 1, 256, 256))
img2 = Variable(torch.rand(1, 1, 256, 256))

if torch.cuda.is_available():
    img1 = img1.cuda()
    img2 = img2.cuda()

print(pytorch_gssim.ssim(img1, img2))

gssim_loss = pytorch_gssim.GSSIM(window_size = 11)

print(gssim_loss(img1, img2))

```
### maximize gssim
```python
import pytorch_gssim
import torch
from torch.autograd import Variable
from torch import optim
import cv2
import numpy as np

npImg1 = cv2.imread("einstein.png")

img1 = torch.from_numpy(np.rollaxis(npImg1, 2)).float().unsqueeze(0)/255.0
img2 = torch.rand(img1.size())

if torch.cuda.is_available():
    img1 = img1.cuda()
    img2 = img2.cuda()


img1 = Variable( img1,  requires_grad=False)
img2 = Variable( img2, requires_grad = True)


# Functional: pytorch_gssim.gssim(img1, img2, window_size = 11, size_average = True)
gssim_value = pytorch_gssim.gssim(img1, img2).data[0]
print("Initial gssim:", gssim_value)

# Module: pytorch_gssim.GSSIM(window_size = 11, size_average = True)
gssim_loss = pytorch_gssim.GSSIM()

optimizer = optim.Adam([img2], lr=0.01)

while gssim_value < 0.95:
    optimizer.zero_grad()
    gssim_out = -gssim_loss(img1, img2)
    gssim_value = -gssim_out.data[0]
    print(gssim_value)
    gssim_out.backward()
    optimizer.step()

```

## Reference
https://ieeexplore.ieee.org/abstract/document/4107183
