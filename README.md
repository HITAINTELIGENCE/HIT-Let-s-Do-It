# Try on Clothes

## 🖥️ Demo

![Person](Database/val/person/000274_0.jpg)
![Human parsing](Database/val/person-parse/000274_0.png)

![Output](output/first/GMM/val/002385_1.jpg)

![Output](output/second/TOM/val/002385_1.jpg)

![Output](Database/val/tryon-person/002385_1.jpg)


## 🔎 Danh Mục

1. [Introduction](#💡-introduction)
2. [Description](#📝-description)
3. [Implementation](#🛠️-implementation)
4. [Dataset](#🗃️-dataset)
5. [Installation](#⚙️-installation)
    - [Download repository](#download-repository)
    - [Requirements](#requirements)
    - [Pre-trained](#pre-trained)
    - [Using app](#using-app)
6. [Documents](#📃-documents)
7. [Related](#🔗-related)

## 💡 Introduction
+ The project is inspired by the needs of the dropshipping T-shirt model. It aims to enable users to upload photos of themselves to virtually try on clothes, making it easier to finalize their purchases when they find a suitable match. 

+ Try-on-clothes is an interactive web app that benefits users by allowing them to see how a particular item fits them. This virtual try-on feature not only enhances the shopping experience and revolutionizes the way people shop for clothes but also reduces costs for retailers.
## 📝 Description

+ The following project is an implementation of paper "VITON: An Image-based Virtual Try-on Network" from University of Maryland, College Park, China. https://arxiv.org/abs/1711.08447

## 🛠️ Implementation
+ Pose generation using openpose
+ Human parsing using psp-net
+ Used Generative Adversarial Networks with Adversarial loss, perceptual loss and L1 loss for smoothening.
+ Used U-Net for generator and a downsampler for discriminator.

## 📈 Training process
+ The pose generated and parser and person image are concatenated along with and cloth imgae are fedded to GMM.
+ Output of above is a warped cloth.
+ Now that concatenated image along with warped cloth is feeded to Gans.
+ Final output is image of person wearing desired cloth.
+ Final loss of generator on validation : 5.016
+ Final loss of discriminator on validation :0.03
+ Epochs Trained : 100
+ Optimizer : Ranger


Here is the link to the training process.

+ [Final_Training_TOM](https://www.kaggle.com/code/hakorushiroki/try-on-gan)

+ [Training_GMM](https://www.kaggle.com/dekiru146/try-on)

+ [Human_Parsing](https://www.kaggle.com/dekiru146/lip-training-real)

+ [Cloth_mask](https://www.kaggle.com/dekiru146/viton-dekiru)


## 🗃️ Dataset
+ [LIP dataset to training human parsing](https://sysu-hcp.net/lip/)

+ [VITON_HD dataset to training cloth mask and model](https://www.dropbox.com/scl/fi/xu08cx3fxmiwpg32yotd7/zalando-hd-resized.zip?rlkey=ks83mdv2pvmrdl2oo2bmmn69w&e=1&dl=0)

+ [Additional Dataset Train](https://drive.google.com/drive/folders/1UfuLsd5pyYOr_TiSwbuK1fphNEF4a7GL?usp=sharing)

## ⚙️ Installation

### Download repository
```
git clone https://github.com/HITAINTELIGENCE/Try-on-clothes.git
```

### Requirements
```
pip install -r requirements.txt
```

### Pre-trained

Download the Model files from the link provided below in the appropriate locations.

[pre_trained](https://drive.google.com/drive/folders/1eauMp5Rtf7yWVV9yKuvLeV-zpmn1-V-c?usp=sharing)

### Using app
```
streamlit run app.py
```

### Using fastapi
```
python main.py
```

## 📃 Documents
[Specifically detail the activities in the article ](https://piquant-pineapple-c0d.notion.site/VITON-728018c0484d4d78b2c79f789541588a?pvs=4)

## 🔗 Related
+ [Human_parsing use pspnet](https://github.com/geekswaroop/Human-Parsing)
+ [Human_parsing use SCHP models](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing?tab=readme-ov-file)
+ [VITON_GAN](https://github.com/shionhonda/viton-gan)