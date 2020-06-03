# CenterNet

For our experiments, we took detector CenterNet and trained it on MPII.

<img src="readme/pose.png" style="zoom: 20%;" >

> [**Objects as Points**](http://arxiv.org/abs/1904.07850),            
> Xingyi Zhou, Dequan Wang, Philipp Kr&auml;henb&uuml;hl,        
> *arXiv technical report ([arXiv 1904.07850](http://arxiv.org/abs/1904.07850))*         

### Keypoint detection on MPII validation

Our results:

| Backbone                  | PCKh  | AP   | FPS  |
| ------------------------- | ----- | ---- | ---- |
| Resnet-18 (256 × 256)     | 73.12 | 45.2 | 31.3 |
| Resnet-18                 | 80.2  | 57.0 | 15.6 |
| DLA-34 (256 × 256)        | 82.4  | 59.8 | 15.3 |
| Hourglass-104 (256 × 256) | 86.3  | 64.8 | 3.8  |
| DLA-34                    | 88.16 | 69.4 | 5.2  |

## Installation

Please refer to [INSTALL.md](readme/INSTALL.md) for installation instructions.
