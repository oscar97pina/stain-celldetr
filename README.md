# Unsupervised Domain Adaptation for Multi-Stain Cell Detection in Breast Cancer with Transformers

## Abstract
The complexity of digital pathology image analysis arises from histopathological slide variability, including tissue specimen differences and stain variations. While publicly available datasets primarily focus on hematoxylin and eosin (H\&E) staining, pathologists often require analysis across multiple stains for comprehensive diagnosis. Deep learning pipelines' implementation in clinical settings is hindered by poor cross-stain generalization, necessitating exhaustive annotations for each stain, which are time-consuming to obtain. In this work, we address these challenges by focusing on breast cancer analysis across four crucial stains: ER, PR, HER2, and Ki-67. Given the necessity of cell-level information for diagnosis, we concentrate on cell detection tasks with detection transformers. Leveraging unsupervised domain adaptation techniques, we bridge the gap between publicly available, annotated H\&E datasets and unlabeled data in other stains. We demonstrate the superiority of adversarial feature learning over source-only and image-level generative methods. Our work contributes to improving digital pathology image analysis by enabling robust and efficient computer-aided diagnosis pipelines across multiple stains, thereby improving diagnostic accuracy in practical settings. The code can be found at \url{https://github.com/oscar97pina/stain-celldetr}.

## Code

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)

## Installation

1. Create a Virtual environment
```bash
python3 -m venv myenv
```

2. Activate the environment
```bash
source myenv/bin/activate
```

3. Install torch>=2.1.1 and torchvision>=0.16.1. We've used cuda 11.8. You can find more information in the [official website](https://pytorch.org/get-started/locally/).
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

4. Install the other requirements
```bash
pip3 install -r requirements.txt
```

5. Build the MultiScaleDeformAttn module (you will need GPU, follow the authors' [instructions](https://github.com/fundamentalvision/Deformable-DETR))
```bash
cd celldetr/models/deformable_detr/ops
./make.sh
``` 

## Usage

### Project structure
```bash
celldetr/
|-- tools/                   # Contains scripts for training, evaluation, and inference
|   |-- train.py             # Training on COCO format dataset
|   |-- eval.py              # Evaluation on COCO format dataset
|   |-- infer.py             # Inference on WSIs
|   |-- adapt.py             # Adapt from source to target datasets   
|-- eval/                    # Evaluation module for evaluating model performance (COCO and Cell detection)
|-- util/                    # Utility functions and modules used throughout the project
|-- data/                    # Datasets, transforms and augmentations for cell detection
|-- models/                  # Deep learning models used in CellDetr
|   |-- deformable_detr/     # Implementation of Deformable DETR model
|   |-- window/              # Implementation of window-based DETR model
|   |-- backbone/            # Backbone networks used in the models
|-- engine.py                # Main engine script for coordinating the training and evaluation process
````

### Adapting form HE to IHC

This repository is based on our original Cell-DETR [paper](https://github.com/oscar97pina/celldetr), so please refer there.

For adapting from HE to IHC, you can firstly pre-train a model for cell detection in IHC:

```bash
python3 -m torch.distributed.launch --use-env --nproc-per-node=NUM_GPUs tools/train.py \
                        --config-file /path/to/pretraining/config/file
```

You have an example of the pre-training config file in the [configs/experiments/adapt/pretrain_he_detection.yaml](configs/experiments/adapt/pretrain_he_detection.yaml). Then, you can adapt from HE to the others stains by calling the ```adapt.py``` script:

```bash
python3 -m torch.distributed.launch --use-env --nproc-per-node=NUM_GPUs tools/adapt.py \
                        --config-file /path/to/pretraining/config/file
```

You can find the config files in [configs/experiments/adapt](configs/experiments/adapt).

## Citation
If you find this work helpful in your research, please consider citing us:

@inproceedings{pina2024unsupervised,
  title={Unsupervised Domain Adaptation for Multi-Stain Cell Detection in Breast Cancer with Transformers},
  author={Pina, Oscar and Vilaplana, Ver{\'o}nica},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5066--5074},
  year={2024}
}

## License
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This project is licensed under the [MIT License](LICENSE).
