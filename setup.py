# setup.py
import setuptools

setuptools.setup(
    name="ML-toolkit",                
    version="0.1.0",                 
    author="AnSyu Li",
    author_email="yessir0621@gmail.com",
    packages=setuptools.find_packages(),  
    install_requires=[
        "tqdm==4.66.5",
        "torch==2.4.0", # mine torch==2.4.0+cu118
        "numpy==1.26.4",
        "pandas==2.2.3",
        "matplotlib==3.10.0"
    ],
    python_requires=">=3.7",          
)
