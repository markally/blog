---
title: "Deep Learning on Windows Subsystem for Linux"
date: 2020-08-16T18:30:52-07:00
draft: false
---

I wanted to do some side projects in deep learning (DL).

Some requirements:

1. I wanted to do it on my personal computer.
2. I had to have local development using a GPU.
3. I wanted to be able to pop into it whenever I wanted, with minimal friction. For me this meant I needed to have access to the development tools and environments I was used to at work (OSX and linux, Tensorflow), and I wanted it all without needing to dual-boot linux.

This led me to the wonderful world of [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/about) (WSL 2) which allows you to run a full linux environment without needing to dualboot, and with better performance than a typical VM. VS code gives you full text editor support that remotes into your linux environment with a simple extension, and is easy to customize with your favorite keybindings (e.g. Sublime for me). It's also possible to use a full IDE like PyCharm if that's your thing. I used this to create this blog and so I felt pretty good about reusing this for my DL side projects.

Great! There's a path for a clean and hassle-free linux development environment on windows. What about deep learning on my local GPU? As of two months ago, WSL 2 [supports GPU acceleration](https://blogs.windows.com/windowsdeveloper/2020/06/17/gpu-accelerated-ml-training-inside-the-windows-subsystem-for-linux/) in their preview builds. While Microsoft suggests using their DirectML library, they also support Nvidia's CUDA.

Let's give it a go. Below are the steps and resources to get fully set up from scratch, though not the exact commands (apologies in advance for the lack of screenshots).

## Installing WSL2 with GPU support

WSL2 with GPU support currently requires a preview version of windows. In addition, there's a particular flag you have to set to get an updated windows kernel. We're going to follow the [Enable NVIDIA CUDA in WSL 2](https://docs.microsoft.com/en-gb/windows/win32/direct3d12/gpu-cuda-in-wsl) guide, but read the clarifications below first.

1. Register for the Windows Insider program as per the step.
2. Select "Dev Channel" (I don't see "Fast Ring") for your Insider settings.
3. Go to "Windows Update" in settings, click "Advanced options" and turn on the flag for "Receive updates for other Microsoft products when you update Windows". This will enable updates for the WSL 2 linux kernel, which is necessary for the NVIDIA drivers.
4. Do your updates. Check that you end up with Build version 20145 or higher.
5. Install WSL 2 as per [Windows Subsystem for Linux Installation Guide for Windows 10](https://docs.microsoft.com/en-us/windows/wsl/install-win10) and then check for updates again BUT read the next section before installing a linux distribution.

At the time of writing, Tensorflow requires [Ubuntu 16.04 or later](https://www.tensorflow.org/install), cuDNN has [debian packages(https://developer.nvidia.com/rdp/cudnn-download)] for Ubuntu 16.x and 18.x. That being said, you can install cuDNN for Ubuntu 20.04 using the base linux tarball (which is what I did).

6. Do the rest of the steps in the WSL 2 installation guide to get a version of Ubuntu installed. 

Installing WSL 2 is fairly easy, with good troubleshooting tips on the website. Note that there's a troubleshooting section on the installation guide page as well as a separate general [troubleshooting section](https://docs.microsoft.com/en-us/windows/wsl/troubleshooting). The latter had the fix for the one issue I had `Error: 0x1bc` which required that I manually install the WSL 2 kernel.

## Installing CUDA and cuDNN

Installing CUDA+cuDNN has never been easy, and I didn't expact anything less here. We'll be following NVIDIA's installing guides for [CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) and [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html) to get libraries installed and then we'll be using miniconda to create an environment and installing tensorflow with pip.

Before starting let's check current Tensorflow CUDA and cuDNN [support](https://www.tensorflow.org/install/source#gpu) to see which versions we need. As of writing, TF 2.3 requires cuDNN 7.6 and CUDA 10.1 however the nightly build supports [CUDA 11 and cuDNN 8](https://github.com/tensorflow/tensorflow/issues/42047). I'll be using CUDA 11, cuDNN 8 and the Tensorflow nightly build.

1. Download and install the [NVIDIA Drivers for WSL](https://developer.nvidia.com/cuda/wsl). If you aren't sure which type of GPU you have, run `dxdiag` from the windows search and look at the display tab.
2. Open up ubuntu and install developer tools, which will include gcc.

`sudo apt-get install update && sudo apt-get install build-essential`

3. Run the commands for setting up CUDA Toolkit. To get the URLs for ubuntu 20.04, visit [CUDA Toolkit Downloads](https://developer.nvidia.com/cuda-downloads) and select the correct settings and then "deb (network)".
4. Follow the steps starting from 2.2 from the [cuDNN documentation](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html) to download and install the correct cuDNN for *linux*. Note that for Ubutun16 and 18 you can use a debian package (select Runtime Library unless you're planning to build TF from scratch), but for Ubuntu20 you should select "cuDNN Library for Linux". If you need to install an older version of cuDNN select "Archived cuDNN Releases". 
5. Add the libraries to your path by creating an entry in your `.bashrc` file. The installation instructions for the Tar File places these files at `/usr/local/cuda/include` and `/usr/local/cuda/lib64`.

```bash
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/include:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
```

## Installing Tensorflow

I use conda as a package manager, but use whichever works best for you. Do all these steps in Ubuntu.

1. Download miniconda (find other links [here](https://docs.conda.io/en/latest/miniconda.html))

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

2. Create an environment `conda create env -n <my_env_name> python=3.8` and activate it `conda activate <my_env_name>`
3. Install tensorflow (I'm using nightly) `pip install tf-nightly-gpu`.
4. Test it! If you get errors around opening dynamic library .so files, make sure you have the correct version installed, that it's in the path, and that you have the correct permissions on them.

You can use this python script to test, it requires both CUDA and cuDNN (taken from [nvidia docs](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)).

```python
import sys
import numpy as np
import tensorflow as tf
from datetime import datetime

device_name = sys.argv[1]  # Choose device from cmd line. Options: gpu or cpu
shape = (int(sys.argv[2]), int(sys.argv[2]))
if device_name == "gpu":
    device_name = "/gpu:0"
else:
    device_name = "/cpu:0"

tf.compat.v1.disable_eager_execution()
with tf.device(device_name):
    random_matrix = tf.random.uniform(shape=shape, minval=0, maxval=1)
    dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
    sum_operation = tf.reduce_sum(dot_operation)


startTime = datetime.now()
with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True)) as session:
        result = session.run(sum_operation)
        print(result)

# Print the results
print("Shape:", shape, "Device:", device_name)
print("Time taken:", datetime.now() - startTime)
```

Create a file like "test_gpu.py" and then `python test_gpu.py gpu 20000` and `python test_gpu.py cpu 20000`. Make sure your GPU is faster!

## Customizing

1. Install VS Code, the Remote WSL extension, and git ([link](https://docs.microsoft.com/en-us/windows/wsl/tutorials/wsl-vscode)). Git will be installed in Ubuntu.
2. Install good packages and themes for VS code! [Official MS Python package](https://code.visualstudio.com/docs/editor/intellisense) which includes linting, Intellisense and more; [Python Extension Pack](https://marketplace.visualstudio.com/items?itemName=donjayamanne.python-extension-pack) which also includes extra goodies such as IntelliCode and better syntax highlighting; [Atom One Dark Theme](https://marketplace.visualstudio.com/items?itemName=akamud.vscode-theme-onedark) for prettier development.

And there we have it! Now with one click we get an Ubuntu environment spun up that can run Tensorflow using my local GPU, and a text editor that interfaces with it with some nice tooling.
