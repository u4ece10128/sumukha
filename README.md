# sumukha
Project sumukha - This is the official repository for building domain adapted embeddings for robust NLP tasks

---
### Motivation
Generic Embeddings for text representation may not be accurate in every domain of interest, this project is to 
leverage knowledge from the domain.
---
### Table of Contents
- [Description](#description)
- [How to Use](#how-to-use)
- [Author Info](#author-info)
---

### Description
sumukha uses the state of the art techniques like standard preprocessing techniques on texts, fasttext models to train 
embeddings, scikit-learn decomposition libraries for further processing.


## How to Use
- Run `pip install .` to install all the requirements.
- Run `sumukha -r ./ preprocess --input dataset_path --output preprocess_path`
- Run `sumukha -r ./ train --input preprocess_path --output trained_results_path `
- Run `sumukha -r ./ encode --input preprocess_path --gen_emb_path general_embeddings_path --dom_emb_path 
domain_embeddings_path`

[Back to The Top](#project-name)