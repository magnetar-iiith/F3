# F3: Fair and Federated Face Attribute Classification with Heterogeneous Data

In this paper, we study Fair Face Attribute Classification (FAC) problem in FL under data heterogeneity. We introduce F3, a FL framework for Fair Face Attribute Classification. Under the F3 framework, we propose two different methodologies (i) Heuristic-based F3 and (ii) Gradient-based F3.

- Heuristic-based F3 includes novel aggregation heuristics: (i) FairBest, (ii) α-FairAvg, and (iii) α-FairAccAvg which prioritize specific local client
model(s) to improve the accuracy and fairness trade-off.
- Gradient-based F3 introduces FairGrad, where the client training is modified to include fairness through gradients communicated by the aggregator, to train a fair and accurate global model.

## Requirements
- Torch
- Numpy
- Pandas
- Torchvision
- sklearn
- Matplotlib

## Code Files
- `data_loader.py` loads train and test datasets into different data loaders to facilitate Federated Learning
- `DH_data_loader.py` loads datasets into different data loaders while maintaining Data Heterogeneity setting
- `healper.py` has helper functions to calculate and plot the results 
- `param.yml` sets parameters for the model

## Citation
@inproceedings{kanaparthy2023f3,
  title={F3: Fair and Federated Face Attribute Classification with Heterogeneous Data},
  author={Kanaparthy, Samhita and Padala, Manisha and Damle, Sankarshan and Sarvadevabhatla, Ravi Kiran and Gujar, Sujit},
  booktitle={The 27th Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD)},
  year={2023}
}
