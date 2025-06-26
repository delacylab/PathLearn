<div align="right">
  Last update: 2025 June 26, 13:32 (by Wayne Lam)
</div>

<hr>

This is a parent directory of the repositories for multiple interconnected time-series AI algorithms, including *RiskPath*, published in *Patterns* in 2025.

While *RiskPath* is designed for time-series cohort data with regularly spaced timestamps, a forthcoming algorithm --- *SparsePath* --- relaxes this assumption, 
enabling predictions on irregularly spaced data. Leveraging sparse inputs such as electronic health records, 
recent experiments on the ABCD v5.1 dataset show that SparsePath achieves strong predictive performance across 
various mental health conditions. SparsePath incorporates an internal attention-masking mechanism, 
enabling the embedded transformer model to make accurate predictions without requiring imputation of missing input data. 
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

