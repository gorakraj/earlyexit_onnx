# earlyexit_onnx

# A benchmark repository of Early Exit Neural Networks in .onnx format

This is a collection of benchmark papers for Early Exit neural networks from recent literature. The start point was the survey paper by researchers at SamsungAI, https://arxiv.org/abs/2106.05022. [see Table 1 on page 5]

Initially started with the following 10 papers relevant to Vision/Classification Tasks:

| No. | Title  | Task | Link to paper | Link to code | Code format | Works with FPGAconvnet |
| - | - | - | - | - | - | - |
| 1 | MSDNet | Vision / Classification | https://arxiv.org/abs/1703.09844 | https://github.com/kalviny/MSDNet-PyTorch | PyTorch | Yes |
| 2 | Not all pixels are equal | Vision / Segmentation | https://arxiv.org/abs/1704.01344 | https://github.com/liuziwei7/region-conv | Caffe | - |
| 3 | Phuong et al. | Vision / Classification | https://openaccess.thecvf.com/content_ICCV_2019/html/Phuong_Distillation-Based_Training_for_Multi-Exit_Architectures_ICCV_2019_paper.html | https://github.com/mary-phuong/multiexit-distillation | PyTorch | - |
| 4 | RBQE | Vision / Enhancement | https://arxiv.org/abs/2006.16581 | https://github.com/RyanXingQL/RBQE | MATLAB | - |
| 5 | MonoBERT | IR / Document Ranking | https://aclanthology.org/2020.sustainlp-1.11/ | https://github.com/castorini/earlyexiting-monobert | PyTorch | - |
| 6 | BranchyNet | Vision / Classification | https://arxiv.org/abs/1709.01686 | https://github.com/kunglab/branchynet | PyTorch | Yes |
| 7 | SDN | Vision / Classification | https://arxiv.org/abs/1810.07052 | https://github.com/gmum/Zero-Time-Waste | PyTorch | - |
| 8 | L2Stop | Vision / {Classification, Denoising} | https://arxiv.org/abs/2006.05082 | https://github.com/xinshi-chen/l2stop | PyTorch | Yes |
| 9 | Triple-wins |  Vision / Classification| https://arxiv.org/abs/2002.10025 | https://github.com/VITA-Group/triple-wins | PyTorch | Yes |
| 10 | DeepSloth | Vision / Classification | https://arxiv.org/abs/2010.02432 | https://github.com/sanghyun-hong/deepsloth | PyTorch | - |



-------------------
Only 4 papers remained (1,6,8,9) that had an available codebase and which layers were supported by fpgaConvNet (http://cas.ee.ic.ac.uk/people/sv1310/fpgaConvNet.html):


| No. | Category                                  | Title of paper  | Modality/Task                         | Paper link                       | Code? | Code link                                       | Format of Code | Comments                                       | Layer types                                                  | Exit calculation type               | Network run?       | .onnx export? | Netron view? |
|-----|-------------------------------------------|-----------------|---------------------------------------|----------------------------------|-------|-------------------------------------------------|----------------|------------------------------------------------|--------------------------------------------------------------|-------------------------------------|--------------------|---------------|--------------|
| 1   | Early-exit network-specific techniques    | MSDNet          | Vision / Classification               | https://arxiv.org/abs/1703.09844 | Yes   | https://github.com/kalviny/MSDNet-PyTorch       | PyTorch        | Possible issue with dotted line in schematic   | nn.Conv2d, nn.BatchNorm2d, nn.Linear, nn.ReLU, nn.MaxPool2d  | softmax with entropy                | Yes   |               |              |
| 6   | Early-exiting network-agnostic techniques | BranchyNet      | Vision / Classification               | https://arxiv.org/abs/1709.01686 | Yes   | https://github.com/kunglab/branchynet,  https://github.com/biggsbenjamin/earlyexitnet           | PyTorch        | -                                              | nn.Conv2d, nn.ReLU, nn.MaxPool2d                             | softmax with entropy                | Yes                | Yes           | Yes          |
| 8   | Learnable exit policies                   | L2Stop          | Vision / {Classification, Denoising}  | https://arxiv.org/abs/2006.05082 | Yes   | https://github.com/xinshi-chen/l2stop           | PyTorch        | section 5.2/5.3 not needed                     | nn.Linear                                                    | own model (see 3.1 Stopping Policy) | No - problem       |               |              |
| 9   | Adversarial robustness                    | Triple-wins     | Vision / Classification               | https://arxiv.org/abs/2002.10025 | Yes   | https://github.com/VITA-Group/triple-wins       | PyTorch        | -                                              | nn.Conv2d, nn.ReLU, nn.MaxPool2d, nn.Dropout, nn.BatchNorm2d | softmax with entropy                | Yes                | Yes           | Yes          |
  
-------------------
**Current Status (for future users):**

The project goal was to convert these codebases into .onnx formats. At the moment papers 6 and 9 have .onnx files and the network schematic can be viewed using the Netron viewer (https://github.com/lutzroeder/netron).
  
The onnx export for paper 1 is nearly complete - there are just some parts of msdnet that need to be adapted for it to work. Error messages will be included as well.

Paper 8 also has some issues sourcing TinyImageNet and running it (section 5.4 in the github repo). The LISTA-stop stage 2 training also does not seem to work (section 5.1). Redownload the files from https://github.com/xinshi-chen/l2stop (some files too big to upload to github) but save the onnx_2 file elsewhere and add it after.


**Useful Notes:**

Individual READMEs are present in the main folder of each network

In each folder there is an environment .yaml file with the packages required (env_1, env_6, env_8 and env_9). Install the environments and activate the relevant file with "conda activate env_x" before running.

When adapting the onnx export files from papers 6 and 9 to use on 1 and 8, use the file onnx_2.py as a base.

-------------------
**Information about the project:**

This project was undertaken as part of my 2 month UROP (Undergraduate Research Opportunity Project) at Imperial College London. I was supervised by Ben Biggs and Prof George Constantinides.

**Methodology:**

I started off collating a spreadsheet based on the SamsungAI survey paper. Then I briefly read the papers and filtered through to find the relevant ones for the criteria Ben required. Looking at the network schematic diagrams were very useful in understanding the structure of the networks. I ended up with 10 relevant papers of which 4 had codebases. I then ran training for those models - in the process creating standardised environment files based on modules required - and then developed an .onnx export file for each of them.

**Useful links (from Ben):**

*For PyTorch tutorials to get up to scratch:*
-  https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
-  https://pytorch.org/tutorials/
-  https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html

*Introductions to .onnx format:*
-  https://onnx.ai/about.html (background on .onnx)
-  https://github.com/onnx/onnx (onnx repo with examples of .onnx format models)
-  https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html (how to export a model from PyTorch to .onnx)
-  https://www.electronjs.org/apps/netron (useful tool for viewing onnx models and helpful when debugging)

  
-------------------
Credit:

1. MSDNet (https://github.com/gaohuang/MSDNet)

>

    @article{DBLP:journals/corr/HuangCLWMW17,
      author    = {Gao Huang and
                   Danlu Chen and
                   Tianhong Li and
                   Felix Wu and
                   Laurens van der Maaten and
                   Kilian Q. Weinberger},
      title     = {Multi-Scale Dense Convolutional Networks for Efficient Prediction},
      journal   = {CoRR},
      volume    = {abs/1703.09844},
      year      = {2017},
      url       = {http://arxiv.org/abs/1703.09844},
      eprinttype = {arXiv},
      eprint    = {1703.09844},
      timestamp = {Mon, 13 Aug 2018 16:49:03 +0200},
      biburl    = {https://dblp.org/rec/journals/corr/HuangCLWMW17.bib},
      bibsource = {dblp computer science bibliography, https://dblp.org}
    }
    
    
  
6. BranchyNet (https://github.com/kunglab/branchynet)

>

    @article{DBLP:journals/corr/abs-1709-01686,
      author    = {Surat Teerapittayanon and
                   Bradley McDanel and
                   H. T. Kung},
      title     = {BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks},
      journal   = {CoRR},
      volume    = {abs/1709.01686},
      year      = {2017},
      url       = {http://arxiv.org/abs/1709.01686},
      eprinttype = {arXiv},
      eprint    = {1709.01686},
      timestamp = {Tue, 28 Apr 2020 13:45:04 +0200},
      biburl    = {https://dblp.org/rec/journals/corr/abs-1709-01686.bib},
      bibsource = {dblp computer science bibliography, https://dblp.org}
    }
    
    
  
8. L2Stop (https://github.com/xinshi-chen/l2stop)

>

    @InProceedings{pmlr-v119-chen20c,
      title = 	 {Learning To Stop While Learning To Predict},
      author =       {Chen, Xinshi and Dai, Hanjun and Li, Yu and Gao, Xin and Song, Le},
      booktitle = 	 {Proceedings of the 37th International Conference on Machine Learning},
      pages = 	 {1520--1530},
      year = 	 {2020},
      editor = 	 {III, Hal Daumé and Singh, Aarti},
      volume = 	 {119},
      series = 	 {Proceedings of Machine Learning Research},
      month = 	 {13--18 Jul},
      publisher =    {PMLR},
      pdf = 	 {http://proceedings.mlr.press/v119/chen20c/chen20c.pdf},
      url = 	 {https://proceedings.mlr.press/v119/chen20c.html},
    }
    
    
  
9. Triple Wins (https://github.com/VITA-Group/triple-wins)

>

    @article{DBLP:journals/corr/abs-2002-10025,
      author    = {Ting{-}Kuei Hu and
                   Tianlong Chen and
                   Haotao Wang and
                   Zhangyang Wang},
      title     = {Triple Wins: Boosting Accuracy, Robustness and Efficiency Together
                   by Enabling Input-Adaptive Inference},
      journal   = {CoRR},
      volume    = {abs/2002.10025},
      year      = {2020},
      url       = {https://arxiv.org/abs/2002.10025},
      eprinttype = {arXiv},
      eprint    = {2002.10025},
      timestamp = {Tue, 03 Mar 2020 14:32:13 +0100},
      biburl    = {https://dblp.org/rec/journals/corr/abs-2002-10025.bib},
      bibsource = {dblp computer science bibliography, https://dblp.org}
    }



  
Thanks once again to Ben Biggs and George Constantinides for their support and guidance in this project. 
    



    
