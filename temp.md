## Methodology

As illustrated in Figure, our pipeline mainly consists of a image preocess module, Restormer<sup>[4]</sup> and a NIMA<sup>[5]</sup> module. In the inference stage, we randomly select a total of 20 batches of images from the 100 low-resolution images and each batch contains of 20 images. 

### Preprocessing
A sliding window processing<sup>[10]</sup> and a rotation processing are done on each batch images. The sliding window size is 128 x 128 and the stride is 32. The rotaion angle are set to 0<sup>o</sup>, 90<sup>o</sup>, 180<sup>o</sup> and 270<sup>o</sup>, respectively.

### Inference
Restormer takes the preocessed images as input and output a high resolution image. Since each batch of images is processed 4 times, a total of 4 high-resolution images are obtained. Therefore, a total of 80 high-resolution images are obtained after 20 rounds of inference. The average of these 80 images is used as the reconstructed image. Drawing on the residual design, we used NIMA<sup>[5]</sup>, a non-reference image quality assessment method, to score the input information and weight it with the reconstructed image and the weighted reconstructed image is used as the final output.


## The simulator configuration
- parameter configuration
```python
import random
kwargs = dict(
                L=random.randint(200, 400),
                D=random.uniform(0.06143, 0.091254),
                Cn2=random.uniform(5.7386e-14, 9.7386e-14),
                corr=random.choice(np.arange(-1, -.00, 0.01)).__round__(3),
            )   
```

## Ground Truth Data
- scene data
    + CTW<sup>[1]</sup>
    + DIV2K<sup>[2]</sup>
    + Total-text<sup>[3]</sup>

- text data
    + 


## External Data
We didn't use external data 



## Reference,
[1] [A Large Chinese Text Dataset in the Wild](https://ctwdataset.github.io/)

[2] [NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

[3] [Total-Text: Towards Orientation Robustness in Scene Text Detection](https://github.com/cs-chan/Total-Text-Dataset)

[4] [Restormer: Efficient Transformer for High-Resolution Image Restoration](https://arxiv.org/abs/2111.09881)

[5] [NIMA: Neural Image Assessment](https://arxiv.org/abs/1709.05424)

[6] [Turbulence Simulator v2: Phase-to-space transform](https://engineering.purdue.edu/ChanGroup/project_turbulence.html)

[7] [Turbulence Simulator v1: Multi-aperture simulator](https://engineering.purdue.edu/ChanGroup/project_turbulence.html)

[8] [Turbulence Reconstruction](https://engineering.purdue.edu/ChanGroup/project_turbulence.html)

[9] [CycleISP: Real Image Restoration via Improved Data Synthesis](https://arxiv.org/abs/2003.07761)

[10] [Revisiting Global Statistics Aggregation for Improving Image Restoration](https://arxiv.org/pdf/2112.04491.pdf)

[11] [Simple Baselines for Image Restoration](https://arxiv.org/abs/2204.04676)
