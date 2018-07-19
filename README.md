# Code for PubMed PICO Element Detection

This is the code for PICO element detection introduced by *[Jin, Di, and Peter Szolovits. "PICO Element Detection in Medical Text via Long Short-Term Memory Neural Networks." Proceedings of the BioNLP 2018 workshop. 2018.](http://www.aclweb.org/anthology/W18-2308)*

Abstract

>Successful evidence-based medicine (EBM) applications rely on answering clinical questions by analyzing large medical literature databases. In order to formulate a well-defined, focused clinical question, a framework called PICO is widely used, which identifies the sentences in a given medical text that belong to the four components: Participants/Problem (P), Intervention (I), Comparison (C) and Outcome (O). In this work, we present a Long Short-Term Memory (LSTM) neural network based model to automatically detect PICO elements. By jointly classifying subsequent sentences in the given text, we achieve state-of-the-art results on PICO element classification compared to several strong baseline models. We also make our curated data public as a benchmarking dataset so that the community can benefit from it.

## How to use

1. First define the path to the word embeddings file, data file and output file, which are defined in the file `model/config.py`. The [data](https://github.com/jind11/PubMed-PICO-Detection) can be downloaded online.
2. Then run the command below to compile the raw data
```
python build_data.py
```
3. Finally run the command below to start training
```
python train.py
```
Note that, after each epoch, the validation set will be evaluated to get the prediction performance and if there are 3 epochs without improvement, the training will be terminated and the test set will be evaludated.

Welcome to post any questions you have and use our code for your work by citing us!
