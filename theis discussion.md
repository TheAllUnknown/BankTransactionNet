


- practical issue:
  - features with complex numbers
  - reference style, IEEE or Harvard
  - 



- Experiment Design
  Goal: To decide what would be good features to detect fraud pattern. by good features, I mean:
    1. must be computationally efficent to get these features
    2. a better performance in accuracy, precision, etc.
    3. the performance should be stable in different networks with different topologies 
   
  The synthezied network is just a random subgraph of the real network.

  - Regarding feature selection
    - Do I use one model with different sets of features?
    - or shoudl use mutiple models with different sets of features?

  - Trainning and evaluation:
    - Since our datasets/networks are synthesized, I can generate 7 networks for training and 3 for evaluation?
    - Or do I generate 10 networks, collects all the node representation together, and take a training & evaluation split? 
    
  - Robustness test of the classifier:
    Perhaps the classifier can detect fraud pattern in our synthesized network, but it may not have as good performance in other networks wih different topologies(the network might be more dense) 
    - other type of synthesized network