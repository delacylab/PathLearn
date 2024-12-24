from setuptools import setup, find_packages

setup(
    name='riskpath',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.24.4',
        'matplotlib>=3.8.0',
        'pandas>=2.2.2',
        'plotly>=5.19.0',
        "scikit-learn>=1.5.0",
        'seaborn>=0.12.2',
        "torch==2.1.0",
        'torchvision>=0.16.0',
        'torchaudio>=2.1.0',
        'imbalanced-learn>=0.12.3',
        'captum>=0.7.0',
        'pillow>=10.0.1',
        'statsmodels>=0.14.1',
        'tslearn>=0.6.3',
    ],
    author='Wai-yin Lam',
    author_email='u6054998@utah.edu',
    description='RiskPath is a multistep predictive pipeline for temporally-sensitive biomedical risk stratification '
                'that achieves solid performance and is tailored to the constraints and demands of biomedical practice.'
                ' The core algorithm is a Long-Short-Term-Memory network, a Transformer network, or a Temporal '
                'Convolutional Network adapted to data with the characteristics common in clinical practice '
                '(tabular; incomplete; collected annually; ≤10 timestamps) and rendered translationally explainable by '
                'extending the Shapley method of computing feature importances for time-series data and embedding this '
                'into the algorithm. RiskPath also provides data-driven approaches for streamlining features in '
                'time-series data before and during model training and analyzing performance-complexity trade-offs '
                'in model construction.',
    url='https://github.com/delacylab/PathLearn/tree/main/RiskPath',
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
    ],
    python_requires='>=3.8'
)
