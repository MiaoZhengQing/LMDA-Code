
# Description
Thanks Can Han @ SJTU for pointing out the EEGDepthAttention parameter update issue, which has been fixed.
Code for paper, [LMDA-Net: A lightweight multi-dimensional attention network for general EEG-based brain-computer interface paradigms and interpretability](https://arxiv.org/pdf/2303.16407.pdf)
- Provided the model required in the paper
- Code required for the interpretable algorithms used in the paper is provided
- Unoptimized code, under continuous updates.

# Requirements
- Python == 3.6 or higher
- Pytorch == 1.10 or higher
- GPU is required. 

# Models Implemented
- [LMDA-Net](https://arxiv.org/pdf/2303.16407.pdf)
- [EEGNet](https://github.com/vlawhern/arl-eegmodels)
- [ShallowConvNet](https://github.com/TNTLFreiburg/braindecode)

# Related works
- This paper is a follow-up version of [SDDA](https://arxiv.org/pdf/2202.09559.pdf), the preprocessing method is inherited from SDDA.
- This paper is followed by the work of  [Time-space-frequency feature Fusion for 3-channel motor imagery classification](https://arxiv.org/pdf/2304.01461.pdf), which investigates the application of time-space-frequency feature fusion methods to 3-lead motion imagery. [Code of TSFF](https://github.com/MiaoZhengQing/TSFF)
- [SDDA](https://arxiv.org/pdf/2202.09559.pdf), [LMDA-Net](https://arxiv.org/pdf/2303.16407.pdf) and [Time-space-frequency feature Fusion for 3-channel motor imagery classification](https://arxiv.org/pdf/2304.01461.pdf) are all under review,  the arxiv  is an early version, the final manuscript will be different. 

# Paper Citation
If you use this code in a scientific publication, please cite us as:  
% TSFF-Net  
Miao Z, Zhao M. Time-space-frequency feature Fusion for 3-channel motor imagery classification[J]. arXiv preprint arXiv:2304.01461, 2023.

% LMDA-Net  
Miao Z, Zhang X, Zhao M, et al. LMDA-Net: A lightweight multi-dimensional attention network for general EEG-based brain-computer interface paradigms and interpretability[J]. arXiv preprint arXiv:2303.16407, 2023.

% SDDA  
Miao Z, Zhang X, Menon C, et al. Priming Cross-Session Motor Imagery Classification with A Universal Deep Domain Adaptation Framework[J]. arXiv preprint arXiv:2202.09559, 2022.

```
% TSFF-Net
@article{miao2023time,
  title={Time-space-frequency feature Fusion for 3-channel motor imagery classification},
  author={Miao, Zhengqing and Zhao, Meirong},
  journal={arXiv preprint arXiv:2304.01461},
  year={2023}
}

% LMDA-Net
@article{miao2023lmda,
  title={LMDA-Net: A lightweight multi-dimensional attention network for general EEG-based brain-computer interface paradigms and interpretability},
  author={Miao, Zhengqing and Zhang, Xin and Zhao, Meirong and Ming, Dong},
  journal={arXiv preprint arXiv:2303.16407},
  year={2023}
}

% SDDA
@article{miao2022priming,
  title={Priming Cross-Session Motor Imagery Classification with A Universal Deep Domain Adaptation Framework},
  author={Miao, Zhengqing and Zhang, Xin and Menon, Carlo and Zheng, Yelong and Zhao, Meirong and Ming, Dong},
  journal={arXiv preprint arXiv:2202.09559},
  year={2022}
}
```

# Contact
Email: mzq@tju.edu.cn
