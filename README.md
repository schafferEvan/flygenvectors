# flygenvectors: sandbox for analysis of whole-brain imaging data in flies

## Environment Set-Up

Create a conda environment:

```
$: conda create --name=flygenvectors python=3.6
$: source activate flygenvectors
(flygenvectors) $: pip install -r requirements.txt 
```

To make the package modules visible to the python interpreter, locally run pip 
install from inside the main `flygenvectors` directory:

```
(flygenvectors) $: pip install -e .
```

To be able to use this environment for jupyter notebooks:

```
(flygenvectors) $: python -m ipykernel install --user --name flygenvectors
``` 

To install ssm, `cd` to any directory where you would like to keep the ssm code and run the following:

```
(flygenvectors) $: git clone git@github.com:slinderman/ssm.git
(flygenvectors) $: cd ssm
(flygenvectors) $: pip install cython
(flygenvectors) $: pip install -e .
```