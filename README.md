# Metaphor_Detection_NLP


## Code Execution Steps

1. Download the dataset from Kaggle which is available under the [link](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset/data). And make sure that this folder will be present in the folder `data`.
Another file that is required for the model training is the labels / annotations file, which is present can be downloaded from [link](https://drive.google.com/file/d/1wAAZwDsHKhotxFkOYJAI57Z-LHxkiyBA/view?usp=drive_link). Place the annotations file in the `data` folder as well. So upon completing downlaod and code extraction the folder structure should look like -


## 📁 Project Structure
```
metaphor-detection/
├── data/                      # Dataset files
│   └── metaphor_dataset.csv   # Main dataset
├── models/                    # Saved model weights
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── svm.pkl
│   ├── bert_baseline/
│   └── melbert/
├── notebooks/                 # Jupyter notebooks for experiments
│   ├── naive_bayes.ipynb
│   ├── random_forest.ipynb
│   ├── svm.ipynb
│   ├── decision_tree.ipynb
│   ├── gradient_boosting.ipynb
│   ├── baseline_bert.ipynb
│   └── melbert.ipynb
├── src/                       # Source code
│   ├── __init__.py
│   ├── preprocessing.py       # Data preprocessing utilities
│   ├── models.py              # Model implementations
│   ├── evaluation.py          # Evaluation metrics
│   └── utils.py               # Helper functions
├── experiments/               # Experiment scripts
│   ├── run_traditional_ml.py
│   ├── run_bert_baseline.py
│   └── run_melbert.py
├── results/                   # Experiment results and logs
│   ├── metrics.csv
│   └── plots/
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── .gitignore                 # Git ignore file
```

2. Now once this step is complete the required packages must be installed. The required packages for the project are available in the `requirements.txt` file. So to install these packages run the command - `pip3 install -r requirements.txt`.

3. Upon completing the installtion an additional package must be installed the `clip` package. This can only be downloaded from the OpenAI GitHub repository so run the command - `pip3 install git+https://github.com/openai/CLIP.git`

4. Now all the necessary packages will be installed and we can execute the code. First run the `CLIP.ipynb` notebook to run obtain the CLIP model weights.

5. Now run the `VAE.ipynb` notebook and which also generate weights for the LDM model.

6. And finally in the `LDM.ipynb` file the weights generated will be used to run the entire model. So in the model code we have adjusted such that the model will load the weights obtained. And upon execution of this file the complete model should be trained.

---



## 🎯 Overview
This project implements multiple machine learning models to detect metaphorical language in text. The best-performing model (MelBERT) achieves 95% accuracy by combining BERT's contextualized embeddings with linguistic theories of metaphor identification.
Key Features

7 different models ranging from traditional ML to deep learning
1,870 labeled examples of metaphorical and literal text
State-of-the-art performance using MelBERT architecture
Comprehensive evaluation with accuracy, precision, recall, and F1-score

## 📊 Dataset
The dataset contains 1,870 text samples with the following structure:

metaphorID: Identifier for the metaphorical word (0-6)
label: Boolean indicating if the word is used metaphorically (True/False)
text: The paragraph containing the target word

### Class Distribution:

Positive (metaphorical): 1,432 samples
Negative (literal): 438 samples
Note: The dataset is imbalanced with ~4:1 ratio favoring metaphorical examples

### Best Worked Models:

### Baseline BERT

Architecture: Pre-trained BERT + Linear layer
Context: Full text context
Fine-tuning: Standard classification head


### MelBERT (Best Model) ⭐

Architecture: RoBERTa + Late interaction mechanism
Context: ±50 words around target word
Special features: Incorporates metaphor identification theories
Performance: 95% accuracy, 96% F1-score

---
## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **MelBERT** | **0.95** | **0.96** | **0.97** | **0.96** |
| Baseline BERT | 0.93 | 0.97 | 0.90 | 0.93 |
| Gradient Boosting | 0.83 | 0.83 | 0.97 | 0.90 |
| Decision Tree | 0.83 | 0.83 | 0.96 | 0.89 |
| SVM | 0.82 | 0.86 | 0.90 | 0.88 |
| Naive Bayes | 0.81 | 0.87 | 0.89 | 0.88 |
| Random Forest | 0.77 | 0.77 | 1.00 | 0.87 |

### Key Findings

 **BERT-based models significantly outperform traditional ML approaches**
- MelBERT (95%) vs. best traditional model (83%)

 **Linguistic theories improve performance**
- MelBERT with theories (95%) vs. Baseline BERT (93%)

 **Context matters**
- Contextualized embeddings capture metaphorical meaning better than static representations

## 🔮 Future Work

- Expand dataset with more diverse examples
- Apply data augmentation techniques
- Experiment with larger language models (GPT, T5)
- Implement ensemble methods (stacking, bagging)
- Add cross-lingual metaphor detection
- Fine-tune hyperparameters using grid search

---
