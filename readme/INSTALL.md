# Installation


0. [Optional but recommended] create a new conda environment. 

    ~~~bash
    conda create --name CenterNet python=3.6
    ~~~
    And activate the environment.
    
    ~~~bash
    conda activate CenterNet
    ~~~

1. Install pytorch 0.4.1:

    ~~~bash
    conda install pytorch=0.4.1 torchvision -c pytorch
    ~~~
    
    And disable cudnn batch normalization(Due to [this issue](https://github.com/xingyizhou/pytorch-pose-hg-3d/issues/16)).
    
     ~~~bash
    # PYTORCH=/path/to/pytorch # usually ~/anaconda3/envs/CenterNet/lib/python3.6/site-packages/
    # for pytorch v0.4.0
    sed -i "1194s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
    # for pytorch v0.4.1
    sed -i "1254s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
     ~~~
    
     For other pytorch version, you can manually open `torch/nn/functional.py` and find the line with `torch.batch_norm` and replace the `torch.backends.cudnn.enabled` with `False`. We observed slight worse training results without doing so. 

    check your version of CUDA and install correct version of pytorch

#### For CUDA >= 10.0 use pytorch==1.0

#### For CUDA < 10.0 use pytorch==0.4.1

```bash
nvcc --version
conda install pytorch==1.0 torchvision -c pytorch
```

2. Install [COCOAPI](https://github.com/cocodataset/cocoapi):

    ~~~bash
    # COCOAPI=/path/to/clone/cocoapi
    git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
    cd $COCOAPI/PythonAPI
    make
    python setup.py install --user
    ~~~

3. Clone this repo:

    ~~~bash
    CenterNet_ROOT=/path/to/clone/CenterNet
    git clone https://github.com/xingyizhou/CenterNet $CenterNet_ROOT
    ~~~


4. Install the requirements

    ~~~bash
    pip install -r requirements.txt
    ~~~
    
5. Compile deformable convolutional (from [DCNv2](https://github.com/CharlesShang/DCNv2/tree/pytorch_0.4)).

    ~~~bash
    cd $CenterNet_ROOT/src/lib/models/networks/DCNv2
    ./make.sh
    ~~~

6. [Optional] Download pertained models for [detection]() or [pose estimation]() and move them to `$CenterNet_ROOT/models/`. More models can be found in [Model zoo](MODEL_ZOO.md).

7. Clone submodules:


    ~~~bash
    git clone https://github.com/ovorlova/eval/ eval
    git clone https://github.com/ovorlova/convert/ convert
    ~~~

8. Download data from the website: http://human-pose.mpi-inf.mpg.de/ (in 'Downloads')

9. Place it in $CenterNet_ROOT/data/coco

10. Place annotations from $CenterNet_ROOT/data to $CenterNet_ROOT/data/coco/annotations

11. For train run (example for human pose estimation)

     ~~~bash
     python main.py multi_pose --batch_size 64 --lr 1.25e-4 --gpus 0 --input_res 128 --dataset mpii --num_epochs 100 
     ~~~

     For test run

     ~~~bash
     python test.py multi_pose --load_model ../models/multi_pose_dla_3x.pth --trainval --dataset mpii
     ~~~

     For the whole list of arguments watch **src/lib/opts.py**

     

12. How to solve some of errors

#### ImportError: cannot import name 'PILLOW_VERSION'

pillow 7.0.0 has removed `PILLOW_VERSION`, you should install another version

```bash
pip install Pillow==6.1
```

####  DCNv2: undefined symbol: __cudaRegisterFatBinaryEnd or error: command 'g++' failed with exit status 1

```bash
cd ~/Code/CenterNet/src/lib/models/networks
rm -r DCNv2
git clone https://github.com/CharlesShang/DCNv2.git
cd DCNv2
sh make.sh
```

