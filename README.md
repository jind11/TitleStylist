# TitleStylist
Source code for our "TitleStylist" paper at ACL 2020: [Jin, Di, Zhijing Jin, Joey Tianyi Zhou, Lisa Orii, and Peter Szolovits. "Hooks in the Headline: Learning to Generate Headlines with Controlled Styles." ACL (2020).](https://arxiv.org/abs/2004.01980). If you use the code, please cite the paper:

```
@article{jin2020hooks,
  title={Hooks in the Headline: Learning to Generate Headlines with Controlled Styles},
  author={Jin, Di and Jin, Zhijing and Zhou, Joey Tianyi and Orii, Lisa and Szolovits, Peter},
  journal={arXiv preprint arXiv:2004.01980},
  year={2020}
}
```

## Requirements
### Python packages
- Pytorch
- fairseq
- blingfire

In order to install them, you can run this command:

```
pip install -r requirements.txt
```

### Bash commands
In order to evaluate the generated headlines by ROUGE scores, you need to install the "files2rouge" package. To do so, run the following commands (provided by [this repository](https://github.com/pltrdy/files2rouge)):

```
pip install -U git+https://github.com/pltrdy/pyrouge
git clone https://github.com/pltrdy/files2rouge.git     
cd files2rouge
python setup_rouge.py
python setup.py install
```

## Usage
1. All data including the combination of CNN and NYT article and headline pairs, and the three style-specific corpora (humor, romance, and clickbait) mentioned in the paper have been placed in the folder "data".

2. Please download the pretrained model parameters of MASS from [this link](https://modelrelease.blob.core.windows.net/mass/mass-base-uncased.tar.gz), unzip it, and put the unzipped files into the folder "pretrained_model/MASS".

3. To train a headline generation model that can simultaneously generated a facutal and a stylistic headline, you can run the following command:
```
./train_mix_CNN_NYT_X.sh --style YOUR_TARGET_STYLE
```
Here the arugment YOUR_TARGET_STYLE specifies any style you would like to have, in this paper, we provide three options: humor, romance, clickbait. 

After running this command, the trained model parameters will be saved into the folder "tmp/exp".

4. If you want to evaluate the trained model and generate headlines (both factual and stylistic) using this model, please run the following command:

```
./evaluate_mix_CNN_NYT_X.sh --style YOUR_TARGET_STYLE --model_dir MODEL_STORED_DIRCTORY
```
In this command, the argument MODEL_STORED_DIRCTORY specifies the directory which stores the trained model.

5. If you want to train and evaluate the headline generation model for more than one style, run the following command:

```
./train_mix_CNN_NYT_multiX.sh
./evaluate_mix_CNN_NYT_multiX.sh --model_dir MODEL_STORED_DIRCTORY
```

## Extensions
