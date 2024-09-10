standard:
- Libra:
    degree mean : 3.02
    power law: 2.15
    avg_clustering: 0.017
    correlation:0.00027
- rabo:
    mean: 4.09
    power: 2.16
    clustering: 0.019
    correlation: -0.0031

we choose 10000 nodes as a network size
- BA Linear preferential, (1,2) degree
    power: 3.12
    degree:3 
    avg_clustering 0.002
    cor: -0.002
-  non-linear k^(beta), 
    power:2.9
    avg_clustering: 0.005
    cor -0.003

- BB: linear preferential, (1,2) degree
    power: 2.67
    degree: 3.01
    clustering: 0.008
    corre:--0.006

- BB: (1,2) degree, (0.5,1.5) fitness score degree*fitness
    - power: 2.91
    - degree:2.97
    - avg_cslutering:0.005
    - corre:-0.005

  - BB (1,2) degree, fitness np.exponential(1/2), if keep drecease the avlue, we will have a high clustering, but the power doen't seem to change much. doen't fit veyr well
    - power:2.5
    - clustering:0.04
    - correlation:-0.004

- homophilic model: 
  - config 1: homophily score between [0,1], homophily calculation: abs of difference, initial node 2
    - power: 2.83
    - degree:2.99
    - clustering: 0.004
    - corr: -0.004