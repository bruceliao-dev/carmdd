
---

## setup.py

```python
from setuptools import setup, find_packages

setup(
    name="carmdd",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "numpy>=1.24",
        "scipy>=1.10",
        "scikit-learn>=1.3",
        "fastdtw>=0.3.4",
        "pyyaml>=6.0",
        "tqdm>=4.65",
        "einops>=0.7",
    ],
    python_requires=">=3.9",
)