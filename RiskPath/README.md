<div align="right">
  Last update: 2025 June 20, 11:51 (by Wayne Lam)
</div>
<hr>

# RiskPath - Explainable deep learning for multistep biomedical prediction in longitudinal data

<!-- Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0 -->
<!-- Applying Apache 2.0: https://www.apache.org/legal/apply-license.html -->
<!-- Badge generator: [https://shields.io/badges](https://shields.io/badges) -->
<!-- Badges: https://github.com/danmadeira/simple-icon-badges -->
<!-- Conda packaging: https://stackoverflow.com/a/49487721 -->
<!-- Emoji: https://gist.github.com/rxaviers/7360908 -->
<!-- Markdown: https://github.com/tchapi/markdown-cheatsheet -->
<!-- Repo DOI: https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content -->
<!-- Repo license: https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository -->
<!-- van der Schaar's autoprognosis: https://github.com/vanderschaarlab/autoprognosis -->

[![tutorial_badge](https://img.shields.io/badge/Tutorial-RiskPath-green)](https://colab.research.google.com/drive/1S8rkrQi39-OUuc0hVasiNjMCGeadKum0?usp=sharing)
[![Patterns_badge](https://img.shields.io/badge/Patterns-10.1016%2Fj.patter.2025.101240-red)](https://www.cell.com/patterns/fulltext/S2666-3899(25)00088-1)
[![medRxiv_badge](https://img.shields.io/badge/medRxiv-2024.09.19.24313909-blue)](https://www.medrxiv.org/content/10.1101/2024.09.19.24313909v2)
[![Zenodo DOI](https://img.shields.io/badge/Zenodo-15061547-yellow)](https://doi.org/10.5281/zenodo.15061547)
[![license_badge](https://img.shields.io/badge/License-Apache_2.0-8A2BE2)](https://www.apache.org/licenses/LICENSE-2.0)

<!-- [![PubMed_badge](https://img.shields.io/badge/PubMed-PMC11451668-blue)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11451668/) -->

<p align="center">
	<img width="800" height="600" src="https://github.com/user-attachments/assets/f6ac67e4-68ff-4a25-bd17-bb471e3e86b6" 	alt="RiskPath_Pipeline_Visualized_Optional")
</p>

# :crystal_ball: Abstract #

_RiskPath_ is a multistep predictive pipeline for temporally-sensitive biomedical risk stratification that achieves solid performance and is tailored to the constraints and demands of biomedical practice. The core algorithm is a Long-Short-Term-Memory network (LSTM, Hochreiter & Schmidhuber 1997), a Transformer network (Vaswani et al. 2017), or a Temporal Convolutional Network (Bai et al. 2018) adapted to data with the characteristics common in clinical practice (tabular; incomplete; collected annually; ≤10 timestamps) and rendered translationally explainable by extending the Shapley method of computing feature importances (Lundberg & Lee 2017) for time-series data and embedding this into the algorithm. RiskPath also provides data-driven approaches for streamlining features in time-series data before and during model training and analyzing performance-complexity trade-offs in model construction.

# :paperclip: Characteristics # 
* Multistep time-series prediction
* Strong performance on risk stratification
* Automated pipelines for classification and regression
* Tuning neural network's topological parameters to optimize performance
* Feature selection for improvements in time and predictive performance
* Explainable AI methods
* Empowered by Pytorch and scikit-learn
  
# :computer: Installation #

We provide a beta version of RiskPath for testing to ensure usability. A release candidate (gamma version) will be available (through conda installation). 

**Using gitclone and pip (AVAILABLE NOW)** 
```
git clone https://github.com/delacylab/PathLearn.git
pip install PathLearn/RiskPath
```

# :page_with_curl: Sample Script #
```python
import torch
from PathLearn.RiskPath.DL.algo import TCN_Classifier
from PathLearn.RiskPath.DL.riskpath import RP_Classifier
from PathLearn.RiskPath.Utils.plots import plot_shap_movie
from PathLearn.RiskPath.Utils.timeseries_simulators import sample_dataset_1

# Create a partitioned sample dataset
x_train, x_test, y_train, y_test, feature_names = sample_dataset_1()

# Create, fit, and evaluate a RiskPath model (embedded with Temporal Convolutional Networks)
RPC = RP_Classifier(base_model=TCN_Classifier, param_grid=[16, 32, 64])
RPC.fit(x_train, y_train, criterion=torch.nn.BCELoss(), optimizer=torch.optim.AdamW)
RPC.evaluate(x_test, y_test)

# Identify the parameter associated with the best AUROC in the test partition
best_param = RPC.get_best_performance(partition='Test', metric='AUROC')['param']

# Compute the SHAP values evaluated by the best-AUROC model at the test partition
shap = RPC.get_SHAP(param=best_param, X=x_test)

# Create a 3D animation illustrating how SHAP values vary across timestamps
plot_shap_movie(shap, feature_names=feature_names, top_n_features=10)
```

<p align="center">
	<img width="800" height="600" src="https://github.com/user-attachments/assets/73ed8833-c76c-483e-8c35-42c47236f572" alt="Animated_SHAP")
</p>

See [![RiskPath Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1S8rkrQi39-OUuc0hVasiNjMCGeadKum0?usp=sharing) for a detailed tutorial on our other visualization tools for performance statistics and SHAP values. 

![RiskPath_Visualization_Grid](https://github.com/user-attachments/assets/3e4f0c4e-f566-4f4c-aead-34dc4fd9331a)


# :speech_balloon: Script Description #

| Class | Description |
|---|---|
| DL/riskpath.py [![RiskPath Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1S8rkrQi39-OUuc0hVasiNjMCGeadKum0?usp=sharing)| <ins>RiskPath</ins> uses an embedded model (e.g. LSTM/Transformer/Temporal Convolutional Network) to effectively classify a target of interest by performing a grid-search over the embedded models' architecture. Leveraged by SHAP values for model interpretability, RiskPath achieves solid performance with an explainable mechanism. This script provides the RiskPath algorithm for both classification (RP_Classifer) and regression (RP_Regressor) tasks.|
| Preprocess/feature_select.py * [![FeatureSelection_Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BY8CqLi3Fj3MSMgkUJXjrFyHClVCWv4x?usp=sharing)| <ins>Feature selection</ins> aims to identify the feature subset relevant to the target variable. It tends to improve both computational efficiency and predictive performance in subsequent prediction tasks (Kohavi and John 1997). We provide a pipeline that integrates L1-regularized logistic regression (with cross-validation) and Boruta (Kursa and Rudnicki 2010) to effectively capture linearly and non-linearly relevant features for classification tasks.|
| Preprocess/padding.py * [![Concatenate Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19bQd80GcpZMv-4LKVrwAOwYEYP-zgOG0?usp=sharing)| Real-world time-series datasets are often irregular or non-square, with features lacking samples at certain timestamps. <ins>Feature imputation</ins> using surrogate data from other timestamps is required to ensure that predictive models can operate smoothly, avoiding common errors caused by missing data.|
| Utils/benchmarking.py *| To compare with the performance of RiskPath optimization, this script embeds commonly used machine learning algorithms (e.g., random forest, logistic regression, SVM) into a pipeline to fit and evaluate the embedded model with flattened 2-dimensional data.| 
| Utils/knee_identifier.py *| A geometrical method to determine the knee (or elbow) of a curve interpolated by a set of values. This function is used for feature ablation, where the features with a predictor importance ranked higher than the knee are used for re-modeling.|
| Utils/plots.py *| This script provides 9 different plotting functions used to visualize the performance statistics and feature explanability of the trained RiskPath model. See the images above for examples.|
| Utils/timeseries_simulators.py *| This script provides a simple method to simulate a synthetic time-series dataset for testing the functionality of RiskPath. See **Sample Script** for the simple one-line execution.|
| Utils/wavelet_transform.py *| Wavelet transformation is used to convert a 3-dimensional time-series dataset into a 2-dimensional one that represents the former's frequency information.|

<p align="right"> * Optional utilities </p>

A later release will provide more Colab Notebook tutorials on the utilities offered in the RiskPath package (e.g., feature ablation and wavelet transformation).

# :bar_chart: Tested Dataset #

| Dataset | Description | Targets studied |
|---|---|---|
| Adolescent Brain Cognitive Development <br> [![ABCD badge](https://img.shields.io/badge/Data-ABCD-blue)](https://abcdstudy.org/)| The largest longitudinal study of brain development and child health in the United States. It includes data on brain imaging, cognitive assessments, mental and physical health, substance use, and environmental factors from over 11,000 children aged 9-10.| ADHD, anxiety, depression, disruptive behaviors, total burden of mental illness|
| Cardiovascular Health Study <br> [![CHS badge](https://img.shields.io/badge/Data-CHS-blue)](https://biolincc.nhlbi.nih.gov/studies/chs/)| A longitudinal study focused on identifying risk factors for cardiovascular disease in older adults. It includes data from over 5,800 participants aged 65 and older, covering cardiovascular events, lifestyle factors, physical function, and extensive clinical and imaging measurements. | (Borderline) Hypertension |
| Multi-Ethnic Study of Atherosclerosis <br> [![MESA badge](https://img.shields.io/badge/Data-MESA-blue)](https://biolincc.nhlbi.nih.gov/studies/mesa/)| A longitudinal study aimed at understanding the development of cardiovascular disease across diverse populations. It includes data from over 6,800 participants aged 45-84, with no prior history of cardiovascular disease, representing multiple ethnic groups. The dataset features extensive imaging, genetic, and clinical data. | Metabolic syndrome |

<!-- | Atherosclerosis Risk in Communities <br> [![ARIC badge](https://img.shields.io/badge/Data-ARIC-blue)](https://biolincc.nhlbi.nih.gov/studies/aric-non/)| A large-scale, longitudinal study designed to investigate the causes and outcomes of atherosclerosis and cardiovascular diseases. It includes data from over 15,000 participants aged 45-64, encompassing medical history, clinical measurements, lifestyle factors, and genetic information. | (TBD) | -->

Users are welcome to contribute to this section by providing their test results using RiskPath. 

# :pencil: Citation #

Please cite the associated paper when using our code.

```
@article{deLacy2025,
  title = {RiskPath: Explainable deep learning for multistep biomedical prediction in longitudinal data},
  author = {de Lacy, Nina and Ramshaw, Michael and Lam, Wai Yin},
  journal = {Patterns},
  publisher = {Elsevier},
  year = {2025},
  doi = {10.1016/j.patter.2025.101240},
  url = {https://doi.org/10.1016/j.patter.2025.101240},
  issn = {2666-3899},
}
```

# :book: References #

Bai, S., Kolter, J. Z., & Koltun, V. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. arXiv preprint arXiv:1803.01271. https://arxiv.org/abs/1803.01271.

Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735–1780. doi: https://doi.org/10.1162/neco.1997.9.8.1735.

Homola, D. (2015). [BorutaPy package](https://danielhomola.com/boruta_py).

Kohavi, R. & John, G. (1997). Wrappers for Feature Subset Selection. Artificial Intelligence. 97. 273-324. doi: https://doi.org/10.1016/S0004-3702(97)00043-X. 

Kursa, M. & Rudnicki, W. (2010). Feature Selection with Boruta Package. Journal of Statistical Software. 36. 1-13. doi: https://doi.org/10.18637/jss.v036.i11. 

Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems, 30. https://arxiv.org/abs/1705.07874.

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., & Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. Advances in Neural Information Processing Systems, 32. https://arxiv.org/abs/1912.01703.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830. http://jmlr.org/papers/v12/pedregosa11a.html.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is All You Need. Advances in Neural Information Processing Systems, 30. https://arxiv.org/abs/1706.03762.



# :globe_with_meridians: License #
This project is licensed under the Apache License, Version 2.0. See the [LICENSE](./LICENSE) file for details.
