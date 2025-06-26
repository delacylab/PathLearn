<div align="right">
  Last update: 2025 June 26, 13:32 (by Wayne Lam)
</div>

<hr>

<div align="center">
  <img src="https://github.com/user-attachments/assets/a26e381d-8003-47f5-912d-6c5f675827d3" width="500"/>
</div>

This is a parent directory of the repositories for multiple interconnected explainable time-series AI algorithms, including *RiskPath*, published in *Patterns* in 2025.

While *RiskPath* is designed for time-series cohort data with regularly spaced timestamps, its new variant --- *SparsePath* --- relaxes this assumption, 
enabling predictions using irregularly spaced data. Leveraging sparse inputs such as electronic health records, recent experiments on the ABCD Study v5.1 dataset show that SparsePath achieves strong predictive performance across various mental health conditions. By incorporating an internal attention-masking mechanism, 
SparsePath enables the embedded transformer model to make accurate predictions without requiring the imputation of missing input data. 
Similar to RiskPath, SparsetPath integrates an optimization technique to efficiently identify the best-performing structural hyperparameter for prediction. The package built for SparsePath will include various utilities, including tools for feature selection and visualization. Please stay tuned for the updates on SparsePath. 

To cite RiskPath, 

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

