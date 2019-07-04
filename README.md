# flygenvectors: sandbox for analysis of whole-brain imaging data in flies

## Environment Set-Up

Create a conda environment:

```
conda create --name=flygenvectors python=3.6
source activate flygenvectors
```

To make the package modules visible to the python interpreter, locally run pip 
install from inside the main `flygenvectors` directory:

```
pip install -e .
```

To be able to use this environment for jupyter notebooks:

```
python -m ipykernel install --user --name flygenvectors --display-name "flygenvectors"
``` 

Install ssm:

```
TODO
```