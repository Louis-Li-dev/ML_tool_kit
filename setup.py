# setup.py
import setuptools

setuptools.setup(
    name="ML-toolkit",                
    version="0.4.9",                 
    author="An-Syu Li",
    author_email="yessir0621@gmail.com",
    url="https://github.com/Louis-Li-dev/ML_tool_kit",
    packages=setuptools.find_packages(include=['mkit', 'mkit.*']),  
    install_requires=[
        "tqdm==4.66.5",
        "torch==2.4.0", # mine torch==2.4.0+cu118
        "numpy==1.26.4",
        "pandas==2.2.3",
        "matplotlib==3.10.0",
        "torch-geometric==2.5.3",
        "scikit-learn==1.4.2",
        "seaborn==0.13.2"
    ],
    python_requires=">=3.7",          
)
