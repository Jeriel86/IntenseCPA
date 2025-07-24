#  IntenseCPA - Interpretable Tensor Fusion Compositional Perturbation Autoencoder 
[![PyPI version](https://badge.fury.io/py/cpa-tools.svg)](https://badge.fury.io/py/cpa-tools) [![Documentation Status](https://readthedocs.org/projects/cpa-tools/badge/?version=latest)](https://cpa-tools.readthedocs.io/en/latest/?badge=latest) [![Downloads](https://static.pepy.tech/badge/cpa-tools)](https://pepy.tech/project/cpa-tools)

## What is CPA?

![Alt text](https://user-images.githubusercontent.com/33202701/156530222-c61e5982-d063-461c-b66e-c4591d2d0de4.png?raw=true "Title")

`CPA` is a framework to learn the effects of perturbations at the single-cell level. CPA encodes and learns phenotypic drug responses across different cell types, doses, and combinations. CPA allows:

* Out-of-distribution predictions of unseen drug and gene combinations at various doses and among different cell types.
* Learn interpretable drug and cell-type latent spaces.
* Estimate the dose-response curve for each perturbation and their combinations.
* Transfer pertubration effects from on cell-type to an unseen cell-type.
* Enable batch effect removal on a latent space and also gene expression space.

`IntenseCPA` is a CPA based model which combine the modality embeddings non-linearly using interpretable tensor fusion (InTense). Thus IntenseCPA is thus able to model explicitly the interaction between the model embeddings and produce a relevance scores for both (modality and interactions).
## Installation

### Installing CPA
You can install the CPA models using pip install -e .

## How to use CPA
Several tutorials are available [here](https://cpa-tools.readthedocs.io/en/latest/tutorials/index.html) to get you started with CPA.
The following table contains the list of tutorials:

|Description | Link |
| --- | --- |
| Predicting combinatorial drug perturbations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theislab/cpa/blob/master/docs/tutorials/combosciplex.ipynb) - [![Open In Documentation](https://img.shields.io/badge/docs-blue)](https://cpa-tools.readthedocs.io/en/latest/tutorials/combosciplex.html) |
| Predicting unseen perturbations uisng external embeddings enabling the model to predict unseen reponses to unseen drugs| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theislab/cpa/blob/master/docs/tutorials/combosciplex_Rdkit_embeddings.ipynb) - [![Open In Documentation](https://img.shields.io/badge/docs-blue)](https://cpa-tools.readthedocs.io/en/latest/tutorials/combosciplex_Rdkit_embeddings.html) |
|Predicting combinatorial CRISPR perturbations| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theislab/cpa/blob/master/docs/tutorials/Norman.ipynb) - [![Open In Documentation](https://img.shields.io/badge/docs-blue)](https://cpa-tools.readthedocs.io/en/latest/tutorials/Norman.html) |
|Context transfer (i.e. predict the effect of a perturbation (e.g. disease) on unseen cell types or transfer perturbation effects from one context to another) demo on IFN-β scRNA perturbation dataset | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theislab/cpa/blob/master/docs/tutorials/Kang.ipynb) - [![Open In Documentation](https://img.shields.io/badge/docs-blue)](https://cpa-tools.readthedocs.io/en/latest/tutorials/Kang.html) |
|Batch effect removal in gene expression and latent space| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theislab/cpa/blob/master/docs/tutorials/Batch_correction_in_expression_space.ipynb) - [![Open In Documentation](https://img.shields.io/badge/docs-blue)](https://cpa-tools.readthedocs.io/en/latest/tutorials/Batch_correction_in_expression_space.html) |


### How to use IntenseCPA
To train IntenseCPA, follow the same steps as shown in the tutorials above, with the addition that you need to include specific model hyperparameters unique to IntenseCPA. These include:

- interaction_order: Represents the order of interactions to consider. The default is 2, but it can be set to 3 or 1 (no interaction).
- intense_p: The p-norm value to use.
- intense_reg_rate: The regularization rate applied by the IntenseCPA module.
- tf_latent_dim: The dimension to which modality embeddings are projected before tensor products are computed.

Below is an example of how the model parameters for IntenseCPA are structured:

```python
model_hparams = {
"n_latent": 128,
"recon_loss": "nb",
"doser_type": "logsigm",
"n_hidden_encoder": 512,
"n_layers_encoder": 3,
"n_hidden_decoder": 512,
"n_layers_decoder": 3,
"use_batch_norm_encoder": True,
"use_layer_norm_encoder": False,
"use_batch_norm_decoder": True,
"use_layer_norm_decoder": False,
"dropout_rate_encoder": 0.1,
"dropout_rate_decoder": 0.1,
"variational": False,
"seed": 434,
"use_intense": True,
"interaction_order": 2,
"intense_reg_rate": 0.05073135389055399,
"intense_p": 2,
"tf_latent_dim": 8
}

trainer_params = {
"n_epochs_kl_warmup": None,
"n_epochs_pretrain_ae": 3,
"n_epochs_adv_warmup": 3,
"n_epochs_mixup_warmup": 10,
"mixup_alpha": 0.1,
"adv_steps": 2,
"n_hidden_adv": 256,
"n_layers_adv": 2,
"use_batch_norm_adv": False,
"use_layer_norm_adv": True,
"dropout_rate_adv": 0,
"reg_adv": 1.419091687459432,
"pen_adv": 12.775412073171998,
"lr": 0.003273373979034034,
"wd": 4e-07,
"adv_lr": 0.00015304936848310163,
"adv_wd": 0.00000011309928874122,
"adv_loss": "cce",
"doser_lr": 0.0007629540879596654,
"doser_wd": 0.00000043589345787571,
"do_clip_grad": False,
"gradient_clip_value": 1.0,
"step_size_lr": 25,
"momentum": 0.5126039493891473
}
```
After the training the relevance score dictionnary can be accessed using the `get_relevance_score` method of the Intense module.

How to optimize the intenseCPA hyperparamters for your data
-----------------------------------------------
We provide an example script to use the built-in hyperparameter optimization function in CPA (based on scvi-tools hyperparam optimizer). You can find the script at `examples/tune_script.py`.

After the hyperparameter optimization using tune_script.py is done, `result_grid.pkl` is saved in your current directory using the `pickle` library. You can load the results using the following code:

```python
import pickle
with open('result_grid.pkl', 'rb') as f:
    result_grid = pickle.load(f)
```
From here, you can follow the instructions in the [Ray Documentations](https://docs.ray.io/en/latest/tune/examples/tune_analyze_results.html#experiment-level-analysis-working-with-resultgrid) to analyze the run, and choose the best hyperparameters for your data.

You can also use the integration with wandb to log the hyperparameter optimization results. You can find the script at `examples/tune_script_wandb.py`. --> `use_wandb=True`

Everything is based on [Ray Tune](https://ray.io/). You can find more information about the hyperparameter optimization in the [Ray Tune Documentations](https://docs.ray.io/en/latest/tune/index.html).

The tuner is adapted and adjusted from scvi-tools v1.2.0 (unreleased) [release notes](https://docs.scvi-tools.org/en/stable/release_notes/index.html)


Datasets and Pre-trained models
-------------------------------
Datasets and pre-trained models are available [here](https://drive.google.com/drive/folders/1yFB0gBr72_KLLp1asojxTgTqgz6cwpju?usp=drive_link).


Recepie for Pre-processing a custom scRNAseq perturbation dataset
-----------------------------------------------------------------
If you have access to you raw data, you can do the following steps to pre-process your dataset. A raw dataset should be a [scanpy](https://scanpy.readthedocs.io/en/stable/) object containing raw counts and available required metadata (i.e. perturbation, dosage, etc.).

Pre-processing steps
--------------------
0. Check for required information in cell metadata:
    a) Perturbation information should be in `adata.obs`.
    b) Dosage information should be in `adata.obs`. In cases like CRISPR gene knockouts, disease states, time perturbations, etc, you can create & add a dummy dosage in your `adata.obs`. For example:
    ```python
        adata.obs['dosage'] = adata.obs['perturbation'].astype(str).apply(lambda x: '+'.join(['1.0' for _ in x.split('+')])).values
    ```
    c) [If available] Cell type information should be in `adata.obs`.
    d) [**Multi-batch** integration] Batch information should be in `adata.obs`.

1. Filter out cells with low number of counts (`sc.pp.filter_cells`). For example:
    ```python
    sc.pp.filter_cells(adata, min_counts=100)
    ```

    [optional]
    ```python
    sc.pp.filter_genes(adata, min_counts=5)
    ```
    
2. Save the raw counts in `adata.layers['counts']`.
    ```python
    adata.layers['counts'] = adata.X.copy()
    ```
3. Normalize the counts (`sc.pp.normalize_total`).
    ```python
    sc.pp.normalize_total(adata, target_sum=1e4, exclude_highly_expressed=True)
    ```
4. Log transform the normalized counts (`sc.pp.log1p`).
    ```python
    sc.pp.log1p(adata)
    ```
5. Highly variable genes selection:
    There are two options:
        1. Use the `sc.pp.highly_variable_genes` function to select highly variable genes.
        ```python
            sc.pp.highly_variable_genes(adata, n_top_genes=5000, subset=True)
        ```
        2. (**Highly Recommended** specially for **Multi-batch** integration scenarios) Use scIB's [highly variable genes selection](https://scib.readthedocs.io/en/latest/api/scib.preprocessing.hvg_batch.html#scib.preprocessing.hvg_batch) function to select highly variable genes. This function is more robust to batch effects and can be used to select highly variable genes across multiple datasets.
        ```python
            import scIB
            adata_hvg = scIB.pp.hvg_batch(adata, batch_key='batch', n_top_genes=5000, copy=True)
        ```


Congrats! Now you're dataset is ready to be used with CPA. Don't forget to save your pre-processed dataset using `adata.write_h5ad` function.


Support and contribute
-------------------------------
If you have a question or new architecture or a model that could be integrated into our pipeline, you can
post an [issue](https://github.com/theislab/cpa/issues/new)

Reference
-------------------------------
If CPA is helpful in your research, please consider citing the  [Lotfollahi et al. 2023](https://www.embopress.org/doi/full/10.15252/msb.202211517)


    @article{lotfollahi2023predicting,
        title={Predicting cellular responses to complex perturbations in high-throughput screens},
        author={Lotfollahi, Mohammad and Klimovskaia Susmelj, Anna and De Donno, Carlo and Hetzel, Leon and Ji, Yuge and Ibarra, Ignacio L and Srivatsan, Sanjay R and Naghipourfar, Mohsen and Daza, Riza M and 
        Martin, Beth and others},
        journal={Molecular Systems Biology},
        pages={e11517},
        year={2023}
    }

