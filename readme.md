# Lunar Lander Reinforement Learning Exercise


## Abgabe: A2C Algo

**Hyperparameter - entropy_coef: (alle mit LR =0.0001)**
0.01 -> zu klein (erreicht keinen smoothed return von 200; stuck bei 0)
0.15 -> optimal (relativ schnelle conversion zu smoothed return von 200)
    -> 0.2 converged schneller, erreicht dafür aber niedrigere returns gegen ende
0.3 -> zu hoch (erreicht keinen smoothed return von 200, stuck bei ca 150)

Learning Rate (bei entropy_coef = 0.2):

zu klein -> 0.000001 (1e-6) policy lernt nie zu landen -> stuck bei smoothed return von -150
optimal -> 0.0001 (1e-4)
zu hoch -> 0.01 (1e-2) policy plumits to -600 avg return and is stuck there

## Abgabe: DQN

