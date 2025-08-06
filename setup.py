from setuptools import setup, find_packages

setup(
    name='G2PT',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'pandas',
        'torch',
        'scikit-learn',
        'scipy',
        'statsmodels',
        'networkx',
        'matplotlib',
        'seaborn',
        'tqdm',
        'mlflow',
        'dvc',
        'obonet',
    ],
    extras_require={
        'docs': [
            'sphinx',
            'sphinx-rtd-theme',
            'myst-parser',
        ]
    },
)
