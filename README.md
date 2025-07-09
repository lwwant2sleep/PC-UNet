# PC-UNet

In this study, a pure convolutional UNet with channel shuffle average, abbreviated as PC-UNet, has been proposed for medical image segmentation. Notably, the proposed PC-UNet is suitable for extracting context features, which is useful for model improvement. PC-UNet operates as an encoder-decoder network, where both the encoder and decoder are stacked with the proposed Pure Convolution (PC) modules. The PC module, containing a channel shuffle average (CSA) component, is efficient in capturing context feature withoug heavy computation. The proposed CSA component converts features in channel dimension into spatial dimension. Thus, cheap computation can be achieved. The effectiveness of the proposed PC-UNet has been rigorously validated on four widely used datasets, which are ISIC 2018, BUSI, GlaS, and Kvasir-SEG, respectively. Experimental results demonstrate that PC-UNet yields outstanding performance without imposing a significant computational load or increasing float operations (FLOPs). PC-UNet is compared with the other eight mainstream models on four datasets, and both Dice and IoU take the lead.


## Experiment
In the experimental section, five publicly available and widely utilized datasets are employed for testing purposes. These datasets are:<br> 
GlaS (gland, with 165 images)<br>
ISIC-2018 (dermoscopy, with 2,594 images)<br>
Kvasir-SEG (endoscopy, with 1,000 images)<br> 
BUSI (breast ultrasound, with 647 images)<br> 
CVC-ClinicDB (colonoscopy, with 612 images)<br>  


In GlaS dataset, we split the dataset into a training set of 85 images and a test set of 80 images. <br>
In ISIC 2018 dataset, we adopt the official split configuration, consisting of a training set with 2,594 images, a validation set with 100 images, and a test set with 1,000 images. <br>
For other dataset, the images are randomly split into training, validation, and test sets with a ratio of 6:2:2.<br>
The dataset path may look like:
```bash
/Your Dataset Path/
├── BUSI/
    ├── Train_Folder/
    │   ├── img
    │   ├── labelcol
    │
    ├── Val_Folder/
    │   ├── img
    │   ├── labelcol
    │
    ├── Test_Folder/
        ├── img
        ├── labelcol
```


## Usage

---

### **Installation**
```bash
conda create -n env_name python=3.7
conda activate env_name
conda install pytorch==1.9.0 torchvision==0.14.1 torchaudio==0.10.0 -c pytorch -c nvidia
``` 


### **Training**
```bash
python train_model.py
```
To run on different setting or different datasets, please modify Config.py .


### **Evaluation**
```bash
python test_model.py
``` 


## Citation

Our repo is useful for your research, please consider citing our article. <br>
This article has been submitted for peer-review in the journal called *Applied Intelligence*.<br>
```bibtex
@ARTICLE{ACHE-Net,
  author  = {Wei Liu, Qian Dong, et al},
  journal = {Applied Intelligence}
  title   = {PC-UNet: A Pure Convolutional UNet with Channel Shuffle Average for Medical Image Segmentation},
  year    = {2025}
}
```


## Contact
For technical questions, please contact lwwant2sleep@qq.com .
