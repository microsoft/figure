# FiGURe: Simple and Efficient Unsupervised Node Representations with Filter Augmentations

This repo contains the code for our paper accepted at NeurIPS'23 and MLG, KDD'23. Our preprint can be found here: https://arxiv.org/abs/2310.01892.

## Citation

Please consider citing the following paper when using our code.

```bibtex
@inproceedings{
ekbote2023figure,
title={FiGURe: Simple and Efficient Unsupervised Node Representations with Filter Augmentations},
author={Chanakya Ekbote and Ajinkya Deshpande and Arun Iyer and Sundararajan Sellamanickam and Ramakrishna B Bairi},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=yh0OkiUk5h}
}
```


## Abstract

 Unsupervised node representations learnt using contrastive learning-based methods have shown good performance on downstream tasks. However, these methods rely on augmentations that mimic low-pass filters, limiting their performance on tasks requiring different eigen-spectrum parts. This paper presents a simple filter-based augmentation method to capture different parts of the eigen-spectrum. We show significant improvements using these augmentations. Further, we show that sharing the same weights across these different filter augmentations is possible, reducing the computational load. In addition, previous works have shown that good performance on downstream tasks requires high dimensional representations. Working with high dimensions increases the computations, especially when multiple augmentations are involved. We mitigate this problem and recover good performance through lower dimensional embeddings using simple random Fourier feature projections. Our method, FiGURe, achieves an average gain of up to 4.4%, compared to the state-of-the-art unsupervised models, across all datasets in consideration, both homophilic and heterophilic.

## Reproducing Results

To reproduce the results presented in the paper, you can utilize the bash scripts available in the `bash_scripts` directory. The bash scripts are further organized into three subdirectories: `32`, `128`, and `512`, representing the results of FiGURe with 32, 128, and 512 dimensions, respectively. To run the scripts, simply cd into the concerned directory and use "bash <filename>".

Please note that the bash scripts provided are specifically for the first data split. For most datasets, the results in the paper are averaged over multiple data splits, so there might be slight differences in the reproduced results.

## Steps To Run

1. Install Anaconda basis this [link](https://www.anaconda.com).

2. Create and activate a conda environment (Note that we use Python 3.10.11):

```bash
conda create -n figure python=3.10.11

conda activate figure
```

3. Install the requirements basis the OS you are using:

```bash
pip install -r <os>_requirements.txt

#For example:

pip install -r linux_requirements.txt
```

4. Download the public datasets present in a drive link:

```python
python download_datasets.py
```

5. Run the commands for a particular dataset given in scripts/all_commands.txt (for a particular embedding dimension):

```bash
python main.py --dataset arxiv-year --algorithm figure --lr 0.001 --hid_units 32 --batch_size 1 --sample_size 5000 --logreg_weight_decay 0 --logreg_epochs 10000 --logreg_lr 2 --alpha_masks 1111 --lr_alphas 0.01 --alpha_activation none --data_dir graph_datasets --dataset_split 0 --seed 5 --gamma 0.7 --sparse True

```

## People Involved

- Chanakya Ekbote [📧](mailto:chanakyekbote@gmail.com)
- Ajinkya Deshpande [📧](mailto:ajinkya.deshpande56@gmail.com)
- Arun Iyer [📧](mailto:ariy@microsoft.com)
- Ram Bairi [📧](mailto:rkbairi@gmail.com)
- Sundararajan Sellamanickam [📧](mailto:ssrajan@microsoft.com)
- B. Ashok (BASH) [📧](mailto:bash@microsoft.com)

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit <https://cla.opensource.microsoft.com>.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
