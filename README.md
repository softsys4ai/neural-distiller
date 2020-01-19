# PaKD Model Compression
<b>Project:</b> Multi-stage Compression of Deep Neural Networks through Pruning and Knowledge Distillation<br>
<b>Lead Professor:</b> Dr. Pooyan Jamshidi<br>
<b>Project lead:</b> Blake Edwards<br>
<b>Contributers:</b> Blake Edwards, Shahriar Iqbal, Stephen Baione<br>


## Table of Contents
1. **Overview**
2. **Dependencies**
3. **Pruning**
4. **Knowledge Distillation**
5. **Project Organization**
6. **Getting Started**
7. **How to Contribute**
8. **References**

## 1. Overview
Exploration and analysis of deep neural network knowledge distillation techniques: teacher assisted knowledge distillation, mulitstage knowledge distillation, and early stopping knowledge distillation.

## 2. Dependencies
### Software
### Hardware
Minimum memory requirement: 3GB of RAM

## 3. Pruning

## 4. Knowledge Distillation
overview of implemented knowledge distillation (KD) methods

## 5. Project Organization

------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------


## 6. Getting Started

## 7. How to Contribute

## 8. References

<br>
<b>References</b>

1. Asif, U., Tang, J., & Harrer, S. (2019). Ensemble knowledge distillation for learning improved and efficient networks. ArXiv:1909.08097 [Cs]. Retrieved from http://arxiv.org/abs/1909.08097

2. Cheng, Y., Wang, D., Zhou, P., & Zhang, T. (2017). A Survey of Model Compression and Acceleration for Deep Neural Networks. ArXiv, abs/1710.09282.

3. Cole, Casey & Janos, Bethany & Anshari, Dien & Thrasher, James & Strayer, Scott & Valafar, Homayoun. (2016). Recognition of Smoking Gesture Using Smart Watch Technology.

4. Han, S., Mao, H., & Dally, W.J. (2015). Deep Compression: Compressing Deep Neural Network with Pruning, Trained Quantization and Huffman Coding. CoRR, abs/1510.00149.

5. He, Y., Kang, G., Dong, X., Fu, Y., & Yang, Y. (2018). Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks. IJCAI.

6. Hinton, G., Vinyals, O. & Dean, J. (2015). Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531.

7. Iandola, F. N., Han, S., Moskewicz, M. W., Ashraf, K., Dally, W. J. & Keutzer, K. (2017). SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size (cite arxiv:1602.07360Comment: In ICLR Format)

8. Iqbal, M. S., Kotthoff, L., & Jamshidi, P. (2019). Transfer learning for performance modeling of deep neural network systems. ArXiv:1904.02838 [Cs]. Retrieved from http://arxiv.org/abs/1904.02838

9. Lecun, Y., Denker, J. S., Solla, S. A., Howard, R. E., & Jackel, L. D. (1990). Optimal brain damage. In D. Touretzky (Ed.), Advances in Neural Information Processing Systems (NIPS 1989), Denver, CO (Vol. 2). Morgan Kaufmann.

10. Mirzadeh, S., Farajtabar, M., Li, A., & Ghasemzadeh, H. (2019). Improved Knowledge Distillation via Teacher Assistant: Bridging the Gap Between Student and Teacher. ArXiv, abs/1902.03393.

11. Jin, S., Di, S., Liang, X., Tian, J., Tao, D., & Cappello, F. (2019). Deepsz: A novel framework to compress deep neural networks by using error-bounded lossy compression. Proceedings of the 28th International Symposium on High-Performance Parallel and Distributed Computing - HPDC ’19, 159–170. https://doi.org/10.1145/3307681.3326608

12. Singh, P., Verma, V. K., Rai, P., & Namboodiri, V. P. (2019). Play and prune: Adaptive filter pruning for deep model compression. ArXiv:1905.04446 [Cs]. Retrieved from http://arxiv.org/abs/1905.04446

13. Szegedy, C., Zaremba, W., Sutskever, I., Bruna, J., Erhan, D., Goodfellow, I.J., & Fergus, R. (2013). Intriguing properties of neural networks. CoRR, abs/1312.6199.

14. You, Z., Yan, K., Ye, J., Ma, M., & Wang, P. (2019). Gate decorator: Global filter pruning method for accelerating deep convolutional neural networks. ArXiv:1909.08174 [Cs, Eess]. Retrieved from http://arxiv.org/abs/1909.08174
 
