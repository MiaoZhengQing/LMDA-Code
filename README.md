
# Accepted by Neuroimage
[LMDA-Net](https://doi.org/10.1016/j.neuroimage.2023.120209) 

# Description
Thanks Can Han @ SJTU for pointing out the EEGDepthAttention parameter update issue, which has been fixed.  
Uploaded the model from the paper.  
I will compare the effect of DepthAttention module based on backpropagation and DepthAttention module without backpropagation on the results of different datasets, and the results will be uploaded later.  
Code for paper, [LMDA-Net: A lightweight multi-dimensional attention network for general EEG-based brain-computer interface paradigms and interpretability](https://arxiv.org/pdf/2303.16407.pdf)
- Provided the model required in the paper
- Code required for the interpretable algorithms used in the paper is provided
- Unoptimized code, under continuous updates.
- The code for preprocessingreferences [braindecode](https://github.com/braindecode/braindecode), but with some streamlining. In the future, I hope to release a scientific version of braindecode, focusing on preprocessing for different datasets. 

## Important Statement
This is my first work on designing a transformer specifically for EEG. The preliminary experiments for a related project have been conducted, but due to the pressing deadline of my doctoral thesis, I have had to temporarily pause this aspect of my work. After October 2023, I hope to continue researching this topic if given the opportunity (That's when I might visit the University of Vienna). 
My doctoral research focuses on the deep integration of artificial neural networks and neuroscience, advocating for the use of lightweight artificial neural network technologies to enhance EEG decoding. It is possible that my doctoral thesis and published papers may not have Chinese versions. However, there is a high-quality Chinese translation of the LMDA-Net paper available on WeChat official account (脑机接口社区), which interested readers can search for.

In the future, I hope to collaborate with internationally renowned research groups to further explore the applications of lightweight artificial neural networks in BCI. My research strengths in this field lie in the deep integration of digital signal processing, machine learning, deep learning, and neuroscience systems. I possess strong problem-solving abilities and have a solid foundation in mathematics and programming. I am comfortable working in an all-English office environment and capable of independently completing research tasks in this field. Additionally, I have previous experience in research on autonomous driving platforms, which has provided me with knowledge in areas such as computer vision and circuitry. I also possess strong teamwork skills.
If your research group is seeking to recruit a postdoctoral researcher in this field, I would greatly appreciate the opportunity for an interview. (mzq@tju.edu.cn)

## Results
- ![image](https://github.com/MiaoZhengQing/LMDA-Code/assets/116713490/b8327d97-6f55-4ac1-8add-912a3feceff8)
- ![8589280dcb2fceae77bbb632240d027](https://github.com/MiaoZhengQing/LMDA-Code/assets/116713490/cbfede5b-9a45-46a4-95a6-f67f9ba0f15e)
-![4bbf01791520f5e47b32826614a38e8](https://github.com/MiaoZhengQing/LMDA-Code/assets/116713490/fb0f4bb7-c7a1-4f4c-964b-cbd3bea326e3)
- ![b59ce60a13237ed9a2947155b7d9b3a](https://github.com/MiaoZhengQing/LMDA-Code/assets/116713490/a162dfc7-ded0-4629-822d-d4ff3acd8355)


# Requirements
- Python == 3.6 or higher
- Pytorch == 1.10 or higher
- GPU is required. 

# Models Implemented
- [LMDA-Net](https://doi.org/10.1016/j.neuroimage.2023.120209) 
- [EEGNet](https://github.com/vlawhern/arl-eegmodels)
- [ShallowConvNet](https://github.com/TNTLFreiburg/braindecode)

# Related works
- This paper is a follow-up version of [SDDA](https://arxiv.org/pdf/2202.09559.pdf), the preprocessing method is inherited from SDDA.
- This paper is followed by the work of  [Time-space-frequency feature Fusion for 3-channel motor imagery classification](https://arxiv.org/pdf/2304.01461.pdf), which investigates the application of time-space-frequency feature fusion methods to 3-lead motion imagery. [Code of TSFF](https://github.com/MiaoZhengQing/TSFF)
- [SDDA](https://arxiv.org/pdf/2202.09559.pdf), [LMDA-Net](https://doi.org/10.1016/j.neuroimage.2023.120209) and [Time-space-frequency feature Fusion for 3-channel motor imagery classification](https://arxiv.org/pdf/2304.01461.pdf) are all under review,  the arxiv  is an early version, the final manuscript will be different. 

# Paper Citation
If you use this code in a scientific publication, please cite us as:  

% for **LMDA-Net architecture and interpretability**, please cite:   
@article{miao2023lmda,
  title={LMDA-Net: A lightweight multi-dimensional attention network for general EEG-based brain-computer interfaces and interpretability},
  author={Miao, Zhengqing and Zhao, Meirong and Zhang, Xin and Ming, Dong},
  journal={NeuroImage},
  pages={120209},
  year={2023},
  publisher={Elsevier}
}

% for **preprocessing**, please cite SDDA:   
@article{miao2022priming,
  title={Priming Cross-Session Motor Imagery Classification with A Universal Deep Domain Adaptation Framework},
  author={Miao, Zhengqing and Zhang, Xin and Menon, Carlo and Zheng, Yelong and Zhao, Meirong and Ming, Dong},
  journal={arXiv preprint arXiv:2202.09559},
  year={2022}
}

# Contact
Email: mzq@tju.edu.cn
