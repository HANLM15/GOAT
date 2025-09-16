# Gradient-Optimized Alternating Training for Multimodal Emotion Recognition(GOAT)

This project is the official open-source implementation of the paper *“Gradient-Optimized Alternating Training for Multimodal Emotion Recognition”*, primarily intended to reproduce and validate the methods and experimental results presented in the paper.

**Authors**: Xianhong Chen, Lingmin Han, and Yichuang Liu

------

### 🛠 Requirements

- `pytorch==2.2.1`
- `python==3.8.18`
- `yaml==0.2.5`
- `numpy==1.24.4`
- `scikit-learn==1.3.2`

------

### 📂 Datasets

- [IEMOCAP](https://sail.usc.edu/iemocap/)

------

### 📄 Code Structure

#### Experiments

The `GOAT_exp` directory contains experiments on the IEMOCAP dataset, where 5,531 utterances are used for 4-class emotion classification (anger, happiness + excited, neutral, sadness). A 5-fold cross-validation strategy is used for training.

- `bert_iemocap/`: Pre-extracted BERT text features from the IEMOCAP dataset.
- `emotion2vec_iemocap/`: Pre-extracted emotion2vec audio features from the IEMOCAP dataset.
- `iemocap_csv/`: CSV files corresponding to each session in the IEMOCAP dataset.
- `config.yml`: Configuration file for the experiments.
- **`ATSmain.py`**: Main program of GOAT for running experiments.
- `ATSutils.py`: Various utility methods used throughout.
- `dataloader.py`: Construction process for training, validation, and testing data loaders.
- `ATSmodel.py`: File containing the model definition.
- `baseline/`: The baseline experiments of the joint training mentioned in the paper.
- `baseline/Amodel.py`: File containing the speech-modal model definition for baseline.
- `baseline/Tmodel.py`: File containing the text-modal model definition for baseline.
- `baseline/Mmodel.py`: File containing the multimodal model definition for baseline.
- `baseline/Mutils.py`: Various utility methods used in baseline.
- **`baseline/main.py`**: Main program of baseline for running experiments.

