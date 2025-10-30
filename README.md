🌱 NeuroEvolution Grid – AIF25
This repository contains the project developed for the Artificial Intelligence Fundamentals (AIF25) course. The project investigates how evolutionary optimization can be combined with neural networks to create intelligent agents capable of navigating complex terrains.

Instead of evolving a single “best” agent, we explicitly promote behavioral diversity through a grid of options. The result is a collection of specialized agents, each mastering a different navigational strategy. This approach allows us to study not only individual performance but also the adaptability of the entire population when facing unseen environments.

🎯 Goals
Explore whether evolutionary strategies (ES) can effectively train neural network controllers without gradient-based methods.

Investigate how diversity promotion impacts robustness and generalization.

Evaluate how a grid of specialized agents adapts to new terrains and how many fine-tuning cycles are required.

🛠️ Methodology
Environment: Built on Gymnasium, ensuring reproducibility and comparability.

Agents: Small feed-forward neural networks receiving state inputs and producing continuous control outputs.

Optimization: Evolutionary Strategies (ES) applied to optimize NN weights.

Behavioral Diversity: A feature-map “grid” rewards agents that discover novel or improved strategies, preserving a wide range of solutions.

📊 Evaluation
Agents are trained on procedurally generated terrains and tested on unseen ones. We measure:

Training performance across terrains.

Adaptability of the diverse agent grid to new environments.

Visualizations of the behavioral grid and sample trajectories.