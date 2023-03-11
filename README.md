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
