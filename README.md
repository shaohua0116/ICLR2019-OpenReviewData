# Crawl and Visualize ICLR 2019 OpenReview Data

<p align="center">
    <img src="asset/logo_wordcloud.png" width="720"/>
</p>

## Descriptions

This Jupyter Notebook contains the data and visualizations that are crawled ICLR 2019 OpenReview webpages. As some are the reviews are still missing (11.3299\% by the time the data is crawled), the results might not be accurate.  All the crawled data (sorted by the average ratings) can be found [here](#Data).

## Visualizations 


The word clouds formed by keywords of submissions show the hot topics including **reinforcement learning**, **generative adversarial networks**, **generative models**, **imitation learning**, **representation learning**, etc.
<p align="center">
    <img src="asset/wordcloud.png" width="720"/>
</p>

This figure is plotted with python [word cloud generator](https://github.com/amueller/word_cloud) 

```
from wordcloud import WordCloud
wordcloud = WordCloud(max_font_size=64, max_words=160, 
                      width=1280, height=640,
                      background_color="black").generate(' '.join(keywords))
plt.figure(figsize=(16, 8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
```

The distributions of reviewer ratings center around 5 to 6 (mean: 5.15).

<p align="center">
    <img src="asset/rating.png" width="640"/>
</p>

You can compute how many papers are beaten by yours with

```
def PR(rating_mean, your_rating):
    pr = np.sum(your_rating >= np.array(rating_mean))/len(rating_mean)*100
    return pr
my_rating = (7+7+9)/3  # your average rating here
print('Your papar beats {:.2f}% of submission '
      '(well, jsut based on the ratings...)'.format(PR(rating_mean, my_rating)))
# ICLR 2017: accept rate 39.1% (198/507) (15 orals and 183 posters)
# ICLR 2018: accept rate 32% (314/981) (23 orals and 291 posters)
# ICLR 2018: accept rate ?% (?/1580)
```

The top 50 common keywrods and their frequency.

<p align="center">
    <img src="asset/frequency.png" width="640"/>
</p>

The average reviewer ratings and the frequency of keywords indicate that to maximize your chance to get higher ratings would be using the keyowrds such as **theory**, **robustness**, or **graph neural network**.

<p align="center">
    <img src="asset/rating_frequency.png" width="800"/>
</p>

## How it works

To crawl data from dynamic websites such as OpenReview, a headless web simulator is created by

```
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
executable_path = '/Users/waltersun/Desktop/chromedriver'  # path to your executable browser
options = Options()
options.add_argument("--headless")
browser = webdriver.Chrome(options=options, executable_path=executable_path)  
```

Then, we can get the content of a webpage

```
browser.get(url)
```

To know what content we can crawl, we will need to inspect the webpage layout.

<p align="center">
    <img src="asset/inspect.png" width="720"/>
</p>

I chose to get the content by

```
key = browser.find_elements_by_class_name("note_content_field")
value = browser.find_elements_by_class_name("note_content_value")
```

The data includes the abstract, keywords, TL; DR, comments.


## <a id="Data"></a>All ICLR 2019 OpenReview data
Collected at 2018-11-10 06:10:40.864681

| Rank | Average Rating | Title | Ratings | Variance | 
| --- | --- | --- | --- | --- | 
| 1 | 9.00 | Benchmarking Neural Network Robustness To Common Corruptions And Perturbations | 9 | 0.00 |
| 2 | 8.67 | Exploration By Random Distillation | 9, 10, 7 | 1.25 |
| 3 | 8.50 | Knockoffgan: Generating Knockoffs For Feature Selection Using Generative Adversarial Networks | 10, 7 | 1.50 |
| 4 | 8.50 | Sparse Dictionary Learning By Dynamical Neural Networks | 9, 8 | 0.50 |
| 5 | 8.33 | Large Scale Gan Training For High Fidelity Natural Image Synthesis | 8, 7, 10 | 1.25 |
| 6 | 8.00 | Generating High Fidelity Images With Subscale Pixel Networks And Multidimensional Upscaling | 7, 10, 7 | 1.41 |
| 7 | 8.00 | Temporal Difference Variational Auto-encoder | 8, 9, 7 | 0.82 |
| 8 | 8.00 | Posterior Attention Models For Sequence To Sequence Learning | 8, 9, 7 | 0.82 |
| 9 | 8.00 | Variational Discriminator Bottleneck: Improving Imitation Learning, Inverse Rl, And Gans By Constraining Information Flow | 6, 10, 8 | 1.63 |
| 10 | 8.00 | Ordered Neurons: Integrating Tree Structures Into Recurrent Neural Networks | 9, 7, 8 | 0.82 |
| 11 | 8.00 | Enabling Factorized Piano Music Modeling And Generation With The Maestro Dataset | 8, 8, 8 | 0.00 |
| 12 | 8.00 | Alista: Analytic Weights Are As Good As Learned Weights In Lista | 10, 6, 8 | 1.63 |
| 13 | 8.00 | Slimmable Neural Networks | 8, 9, 7 | 0.82 |
| 14 | 7.67 | Pay Less Attention With Lightweight And Dynamic Convolutions | 8, 7, 8 | 0.47 |
| 15 | 7.67 | Learning Robust Representations By Projecting Superficial Statistics Out | 7, 7, 9 | 0.94 |
| 16 | 7.67 | Supervised Community Detection With Line Graph Neural Networks | 6, 9, 8 | 1.25 |
| 17 | 7.67 | Composing Complex Skills By Learning Transition Policies With Proximity Reward Induction | 7, 9, 7 | 0.94 |
| 18 | 7.67 | Slalom: Fast, Verifiable And Private Execution Of Neural Networks In Trusted Hardware | 7, 7, 9 | 0.94 |
| 19 | 7.67 | Smoothing The Geometry Of Probabilistic Box Embeddings | 8, 8, 7 | 0.47 |
| 20 | 7.67 | Identifying And Controlling Important Neurons In Neural Machine Translation | 7, 10, 6 | 1.70 |
| 21 | 7.67 | Critical Learning Periods In Deep Networks | 9, 8, 6 | 1.25 |
| 22 | 7.67 | A Variational Inequality Perspective On Generative Adversarial Networks | 8, 8, 7 | 0.47 |
| 23 | 7.67 | On Random Deep Autoencoders: Exact Asymptotic Analysis, Phase Transitions, And Implications To Training | 9, 6, 8 | 1.25 |
| 24 | 7.67 | Adaptive Input Representations For Neural Language Modeling | 7, 8, 8 | 0.47 |
| 25 | 7.67 | Learning Unsupervised Learning Rules | 8, 7, 8 | 0.47 |
| 26 | 7.50 | Differentiable Learning-to-normalize Via Switchable Normalization | 7, 8 | 0.50 |
| 27 | 7.50 | Near-optimal Representation Learning For Hierarchical Reinforcement Learning | 9, 6 | 1.50 |
| 28 | 7.50 | Snip: Single-shot Network Pruning Based On Connection Sensitivity | 6, 9 | 1.50 |
| 29 | 7.50 | Diffusion Scattering Transforms On Graphs | 9, 6 | 1.50 |
| 30 | 7.50 | Gradient Descent Provably Optimizes Over-parameterized Neural Networks | 8, 7 | 0.50 |
| 31 | 7.50 | Deep Learning 3d Shapes Using Alt-az Anisotropic 2-sphere Convolution | 8, 7 | 0.50 |
| 32 | 7.33 | Diversity Is All You Need: Learning Skills Without A Reward Function | 8, 7, 7 | 0.47 |
| 33 | 7.33 | Efficient Training On Very Large Corpora Via Gramian Estimation | 7, 8, 7 | 0.47 |
| 34 | 7.33 | Label Super-resolution Networks | 7, 6, 9 | 1.25 |
| 35 | 7.33 | Large-scale Study Of Curiosity-driven Learning | 6, 9, 7 | 1.25 |
| 36 | 7.33 | Approximability Of Discriminators Implies Diversity In Gans | 8, 7, 7 | 0.47 |
| 37 | 7.33 | Kernel Change-point Detection With Auxiliary Deep Generative Models | 8, 8, 6 | 0.94 |
| 38 | 7.33 | Gradient Descent Aligns The Layers Of Deep Linear Networks | 7, 9, 6 | 1.25 |
| 39 | 7.33 | Learning Deep Representations By Mutual Information Estimation And Maximization | 5, 7, 10 | 2.05 |
| 40 | 7.33 | Biologically-plausible Learning Algorithms Can Scale To Large Datasets | 9, 9, 4 | 2.36 |
| 41 | 7.33 | Evaluating Robustness Of Neural Networks With Mixed Integer Programming | 7, 8, 7 | 0.47 |
| 42 | 7.33 | Towards Metamerism Via Foveated Style Transfer | 7, 8, 7 | 0.47 |
| 43 | 7.33 | Time-agnostic Prediction: Predicting Predictable Video Frames | 7, 8, 7 | 0.47 |
| 44 | 7.33 | Small Nonlinearities In Activation Functions Create Bad Local Minima In Neural Networks | 7, 7, 8 | 0.47 |
| 45 | 7.33 | Understanding And Improving Interpolation In Autoencoders Via An Adversarial Regularizer | 5, 8, 9 | 1.70 |
| 46 | 7.33 | Lanczosnet: Multi-scale Deep Graph Convolutional Networks | 7, 7, 8 | 0.47 |
| 47 | 7.33 | Deep Decoder: Concise Image Representations From Untrained Non-convolutional Networks | 8, 8, 6 | 0.94 |
| 48 | 7.33 | Promp: Proximal Meta-policy Search | 6, 7, 9 | 1.25 |
| 49 | 7.33 | Visualizing And Understanding Generative Adversarial Networks | 7, 7, 8 | 0.47 |
| 50 | 7.00 | What Do You Learn From Context? Probing For Sentence Structure In Contextualized Word Representations | 7, 7, 7 | 0.00 |
| 51 | 7.00 | Deep, Skinny Neural Networks Are Not Universal Approximators | 6, 8, 7 | 0.82 |
| 52 | 7.00 | Scalable Reversible Generative Models With Free-form Continuous Dynamics | 7, 7, 7 | 0.00 |
| 53 | 7.00 | How Powerful Are Graph Neural Networks? | 7, 6, 8 | 0.82 |
| 54 | 7.00 | Invariant And Equivariant Graph Networks | 8, 4, 9 | 2.16 |
| 55 | 7.00 | On The Minimal Supervision For Training Any Binary Classifier From Only Unlabeled Data | 7 | 0.00 |
| 56 | 7.00 | Hierarchical Visuomotor Control Of Humanoids | 8, 6 | 1.00 |
| 57 | 7.00 | Clarinet: Parallel Wave Generation In End-to-end Text-to-speech | 9, 5, 7 | 1.63 |
| 58 | 7.00 | Predicting The Generalization Gap In Deep Networks With Margin Distributions | 9, 5 | 2.00 |
| 59 | 7.00 | Meta-learning Probabilistic Inference For Prediction | 7, 6, 8 | 0.82 |
| 60 | 7.00 | Adaptive Mixture Of Low-rank Factorizations For Compact Neural Modeling | 6, 8 | 1.00 |
| 61 | 7.00 | Deep Online Learning Via Meta-learning: Continual Adaptation For Model-based Rl | 7, 7, 7 | 0.00 |
| 62 | 7.00 | Learning A Sat Solver From Single-bit Supervision | 7, 7, 7 | 0.00 |
| 63 | 7.00 | Local Sgd Converges Fast And Communicates Little | 8, 5, 8 | 1.41 |
| 64 | 7.00 | Learning Abstract Models For Long-horizon Exploration | 6, 8 | 1.00 |
| 65 | 7.00 | On The Universal Approximability And Complexity Bounds Of Quantized Relu Neural Networks | 7, 6, 8 | 0.82 |
| 66 | 7.00 | The Role Of Over-parametrization In Generalization Of Neural Networks | 7 | 0.00 |
| 67 | 7.00 | Feature Intertwiners | 5, 9, 7 | 1.63 |
| 68 | 7.00 | Darts: Differentiable Architecture Search | 6, 7, 8 | 0.82 |
| 69 | 7.00 | Learning To Screen For Fast Softmax Inference On Large Vocabulary Neural Networks | 7, 6, 8 | 0.82 |
| 70 | 7.00 | Greedy Attack And Gumbel Attack: Generating Adversarial Examples For Discrete Data | 6, 8, 7 | 0.82 |
| 71 | 7.00 | Learning Sparse Relational Transition Models | 6, 7, 8 | 0.82 |
| 72 | 7.00 | Towards Consistent Performance On Atari Using Expert Demonstrations | 7, 7 | 0.00 |
| 73 | 7.00 | Global-to-local Memory Pointer Networks For Task-oriented Dialogue | 8, 8, 5 | 1.41 |
| 74 | 7.00 | Wizard Of Wikipedia: Knowledge-powered Conversational Agents | 7, 6, 8 | 0.82 |
| 75 | 7.00 | Robustness May Be At Odds With Accuracy | 6, 7, 8 | 0.82 |
| 76 | 7.00 | Learning Self-imitating Diverse Policies | 8, 5, 8 | 1.41 |
| 77 | 7.00 | Woulda, Coulda, Shoulda: Counterfactually-guided Policy Search | 7, 7 | 0.00 |
| 78 | 7.00 | Adversarial Domain Adaptation For Stable Brain-machine Interfaces | 9, 5, 7 | 1.63 |
| 79 | 7.00 | Relaxed Quantization For Discretized Neural Networks | 7, 7, 7 | 0.00 |
| 80 | 7.00 | Cot: Cooperative Training For Generative Modeling Of Discrete Data | 7, 7, 7 | 0.00 |
| 81 | 7.00 | Unsupervised Domain Adaptation For Distance Metric Learning | 8, 5, 8 | 1.41 |
| 82 | 7.00 | Auxiliary Variational Mcmc | 7, 7, 7 | 0.00 |
| 83 | 7.00 | How Important Is A Neuron | 7, 7, 7 | 0.00 |
| 84 | 7.00 | The Neuro-symbolic Concept Learner: Interpreting Scenes, Words, And Sentences From Natural Supervision | 7, 5, 9 | 1.63 |
| 85 | 7.00 | The Comparative Power Of Relu Networks And Polynomial Kernels In The Presence Of Sparse Latent Structure | 7, 7, 7 | 0.00 |
| 86 | 7.00 | Learning Implicitly Recurrent Cnns Through Parameter Sharing | 8, 7, 6 | 0.82 |
| 87 | 7.00 | Don't Settle For Average, Go For The Max: Fuzzy Sets And Max-pooled Word Vectors | 8, 8, 5 | 1.41 |
| 88 | 7.00 | The Effects Of Neural Resource Constraints On Early Visual Representations | 5, 8, 8 | 1.41 |
| 89 | 7.00 | An Analytic Theory Of Generalization Dynamics And Transfer Learning In Deep Linear Networks | 8, 7, 6 | 0.82 |
| 90 | 7.00 | Deep Graph Infomax | 9, 5, 7 | 1.63 |
| 91 | 7.00 | Neural Network Gradient-based Learning Of Black-box Function Interfaces | 7, 7, 7 | 0.00 |
| 92 | 7.00 | Detecting Egregious Responses In Neural Sequence-to-sequence Models | 6, 7, 8 | 0.82 |
| 93 | 7.00 | Learning To Navigate The Web | 7, 7, 7 | 0.00 |
| 94 | 7.00 | Towards Robust, Locally Linear Deep Networks | 8, 6, 7 | 0.82 |
| 95 | 7.00 | The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks | 5, 8, 8 | 1.41 |
| 96 | 7.00 | Gansynth: Adversarial Neural Audio Synthesis | 6, 7, 8 | 0.82 |
| 97 | 7.00 | A2bcd: Asynchronous Acceleration With Optimal Complexity | 5, 7, 9 | 1.63 |
| 98 | 7.00 | Learning Neural Pde Solvers With Convergence Guarantees | 7, 8, 6 | 0.82 |
| 99 | 7.00 | Improving The Differentiable Neural Computer Through Memory Masking, De-allocation, And Link Distribution Sharpness Control | 8, 6, 7 | 0.82 |
| 100 | 7.00 | Lagging Inference Networks And Posterior Collapse In Variational Autoencoders | 7, 6, 8 | 0.82 |
| 101 | 7.00 | Emi: Exploration With Mutual Information Maximizing State And Action Embeddings | 7, 7 | 0.00 |
| 102 | 7.00 | Riemannian Adaptive Optimization Methods | 7, 7, 7 | 0.00 |
| 103 | 6.67 | Discriminator-actor-critic: Addressing Sample Inefficiency And Reward Bias In Adversarial Imitation Learning | 8, 5, 7 | 1.25 |
| 104 | 6.67 | Generalized Tensor Models For Recurrent Neural Networks | 6, 7, 7 | 0.47 |
| 105 | 6.67 | Learning Grid-like Units With Vector Representation Of Self-position And Matrix Representation Of Self-motion | 8, 6, 6 | 0.94 |
| 106 | 6.67 | Learning Concise Representations For Regression By Evolving Networks Of Trees | 7, 5, 8 | 1.25 |
| 107 | 6.67 | Doubly Reparameterized Gradient Estimators For Monte Carlo Objectives | 7, 7, 6 | 0.47 |
| 108 | 6.67 | There Are Many Consistent Explanations Of Unlabeled Data: Why You Should Average | 6, 8, 6 | 0.94 |
| 109 | 6.67 | Disjoint Mapping Network For Cross-modal Matching Of Voices And Faces | 7, 6, 7 | 0.47 |
| 110 | 6.67 | Graph Hypernetworks For Neural Architecture Search | 7, 6, 7 | 0.47 |
| 111 | 6.67 | Learning A Meta-solver For Syntax-guided Program Synthesis | 7, 6, 7 | 0.47 |
| 112 | 6.67 | Bayesian Prediction Of Future Street Scenes Using Synthetic Likelihoods | 6, 8, 6 | 0.94 |
| 113 | 6.67 | Complement Objective Training | 5, 8, 7 | 1.25 |
| 114 | 6.67 | Detecting Adversarial Examples Via Neural Fingerprinting | 5, 9, 6 | 1.70 |
| 115 | 6.67 | Learning Localized Generative Models For 3d Point Clouds Via Graph Convolution | 9, 4, 7 | 2.05 |
| 116 | 6.67 | Recurrent Experience Replay In Distributed Reinforcement Learning | 7, 6, 7 | 0.47 |
| 117 | 6.67 | A Generative Model For Electron Paths | 8, 4, 8 | 1.89 |
| 118 | 6.67 | A Data-driven And Distributed Approach To Sparse Signal Representation And Recovery | 8, 6, 6 | 0.94 |
| 119 | 6.67 | Minimal Random Code Learning: Getting Bits Back From Compressed Model Parameters | 7, 6, 7 | 0.47 |
| 120 | 6.67 | Looking For Elmo's Friends: Sentence-level Pretraining Beyond Language Modeling | 5, 7, 8 | 1.25 |
| 121 | 6.67 | Quasi-hyperbolic Momentum And Adam For Deep Learning | 6, 6, 8 | 0.94 |
| 122 | 6.67 | Directed-info Gail: Learning Hierarchical Policies From Unsegmented Demonstrations Using Directed Information | 6, 6, 8 | 0.94 |
| 123 | 6.67 | Glue: A Multi-task Benchmark And Analysis Platform For Natural Language Understanding | 7, 5, 8 | 1.25 |
| 124 | 6.67 | Learning From Incomplete Data With Generative Adversarial Networks | 7, 6, 7 | 0.47 |
| 125 | 6.67 | Solving The Rubik's Cube With Approximate Policy Iteration | 7, 6, 7 | 0.47 |
| 126 | 6.67 | Trellis Networks For Sequence Modeling | 7, 6, 7 | 0.47 |
| 127 | 6.67 | Generative Question Answering: Learning To Answer The Whole Question | 6, 6, 8 | 0.94 |
| 128 | 6.67 | Fixing Variational Bayes: Deterministic Variational Inference For Bayesian Neural Networks | 6, 7, 7 | 0.47 |
| 129 | 6.67 | Learning To Learn Without Forgetting By Maximizing Transfer And Minimizing Interference | 5, 8, 7 | 1.25 |
| 130 | 6.67 | Query-efficient Hard-label Black-box Attack: An Optimization-based Approach | 7, 6, 7 | 0.47 |
| 131 | 6.67 | Hyperbolic Attention Networks | 6, 7, 7 | 0.47 |
| 132 | 6.67 | Off-policy Evaluation And Learning From Logged Bandit Feedback: Error Reduction Via Surrogate Policy | 6, 8, 6 | 0.94 |
| 133 | 6.67 | Adashift: Decorrelation And Convergence Of Adaptive Learning Rate Methods | 6, 5, 9 | 1.70 |
| 134 | 6.67 | Rotate: Knowledge Graph Embedding By Relational Rotation In Complex Space | 7, 7, 6 | 0.47 |
| 135 | 6.67 | G-sgd: Optimizing Relu Neural Networks In Its Positively Scale-invariant Space | 7, 7, 6 | 0.47 |
| 136 | 6.67 | Optimal Completion Distillation For Sequence Learning | 7, 7, 6 | 0.47 |
| 137 | 6.67 | Meta-learning For Stochastic Gradient Mcmc | 7, 7, 6 | 0.47 |
| 138 | 6.67 | Automatically Composing Representation Transformations As A Means For Generalization | 6, 7, 7 | 0.47 |
| 139 | 6.67 | Flowqa: Grasping Flow In History For Conversational Machine Comprehension | 7, 6, 7 | 0.47 |
| 140 | 6.67 | Towards The First Adversarially Robust Neural Network Model On Mnist | 7, 7, 6 | 0.47 |
| 141 | 6.67 | Sample Efficient Adaptive Text-to-speech | 7, 7, 6 | 0.47 |
| 142 | 6.67 | Adaptivity Of Deep Relu Network For Learning In Besov And Mixed Smooth Besov Spaces: Optimal Rate And Curse Of Dimensionality | 8, 6, 6 | 0.94 |
| 143 | 6.67 | K For The Price Of 1: Parameter Efficient Multi-task And Transfer Learning | 7, 5, 8 | 1.25 |
| 144 | 6.67 | Bounce And Learn: Modeling Scene Dynamics With Real-world Bounces | 5, 7, 8 | 1.25 |
| 145 | 6.67 | Visual Semantic Navigation Using Scene Priors | 7, 6, 7 | 0.47 |
| 146 | 6.67 | Non-vacuous Generalization Bounds At The Imagenet Scale: A Pac-bayesian Compression Approach | 6, 6, 8 | 0.94 |
| 147 | 6.67 | Learning Factorized Multimodal Representations | 7, 7, 6 | 0.47 |
| 148 | 6.67 | Learning To Solve Circuit-sat: An Unsupervised Differentiable Approach | 5, 8, 7 | 1.25 |
| 149 | 6.67 | Dynamic Sparse Graph For Efficient Deep Learning | 8, 5, 7 | 1.25 |
| 150 | 6.67 | Learning To Schedule Communication In Multi-agent Reinforcement Learning | 6, 8, 6 | 0.94 |
| 151 | 6.67 | Big-little Net: An Efficient Multi-scale Feature Representation For Visual And Speech Recognition | 7, 6, 7 | 0.47 |
| 152 | 6.67 | Differentiable Perturb-and-parse: Semi-supervised Parsing With A Structured Variational Autoencoder | 8, 7, 5 | 1.25 |
| 153 | 6.67 | Dimensionality Reduction For Representing The Knowledge Of Probabilistic Models | 6, 5, 9 | 1.70 |
| 154 | 6.67 | Deep Self-organization: Interpretable Discrete Representation Learning On Time Series | 9, 5, 6 | 1.70 |
| 155 | 6.67 | Adversarial Attacks On Graph Neural Networks Via Meta Learning | 7, 7, 6 | 0.47 |
| 156 | 6.67 | Unsupervised Speech Recognition Via Segmental Empirical Output Distribution Matching | 7, 7, 6 | 0.47 |
| 157 | 6.67 | Visual Explanation By Interpretation: Improving Visual Feedback Capabilities Of Deep Neural Networks | 8, 5, 7 | 1.25 |
| 158 | 6.67 | Analysis Of Quantized Deep Networks | 6, 7, 7 | 0.47 |
| 159 | 6.67 | Stochastic Optimization Of Sorting Networks Via Continuous Relaxations | 8, 6, 6 | 0.94 |
| 160 | 6.67 | A Mean Field Theory Of Batch Normalization | 7, 6, 7 | 0.47 |
| 161 | 6.67 | Learning Two-layer Neural Networks With Symmetric Inputs | 7, 6, 7 | 0.47 |
| 162 | 6.67 | Do Deep Generative Models Know What They Don't Know? | 7, 6, 7 | 0.47 |
| 163 | 6.67 | Provable Online Dictionary Learning And Sparse Coding | 7, 6, 7 | 0.47 |
| 164 | 6.67 | Deep Frank-wolfe For Neural Network Optimization | 7, 5, 8 | 1.25 |
| 165 | 6.67 | Nadpex: An On-policy Temporally Consistent Exploration Method For Deep Reinforcement Learning | 8, 5, 7 | 1.25 |
| 166 | 6.67 | Active Learning With Partial Feedback | 7, 6, 7 | 0.47 |
| 167 | 6.67 | Deep Layers As Stochastic Solvers | 7, 5, 8 | 1.25 |
| 168 | 6.67 | Padam: Closing The Generalization Gap Of Adaptive Gradient Methods In Training Deep Neural Networks | 6, 5, 9 | 1.70 |
| 169 | 6.67 | Learning To Infer And Execute 3d Shape Programs | 6, 7, 7 | 0.47 |
| 170 | 6.67 | Phase-aware Speech Enhancement With Deep Complex U-net | 6, 7, 7 | 0.47 |
| 171 | 6.50 | Invariance And Inverse Stability Under Relu | 6, 7 | 0.50 |
| 172 | 6.50 | An Adaptive Homeostatic Algorithm For The Unsupervised Learning Of Visual Features | 4, 9 | 2.50 |
| 173 | 6.50 | Spherical Cnns On Unstructured Grids | 6, 7 | 0.50 |
| 174 | 6.50 | Defensive Quantization: When Efficiency Meets Robustness | 6, 7 | 0.50 |
| 175 | 6.50 | Dyrep: Learning Representations Over Dynamic Graphs | 5, 8 | 1.50 |
| 176 | 6.50 | Efficient Two-step Adversarial Defense For Deep Neural Networks | 6, 7 | 0.50 |
| 177 | 6.50 | Dynamically Unfolding Recurrent Restorer: A Moving Endpoint Control Method For Image Restoration | 6, 7 | 0.50 |
| 178 | 6.50 | Learning To Remember More With Less Memorization | 8, 5 | 1.50 |
| 179 | 6.50 | No Training Required: Exploring Random Encoders For Sentence Classification | 5, 8 | 1.50 |
| 180 | 6.50 | An Alarm System For Segmentation Algorithm Based On Shape Model | 8, 5 | 1.50 |
| 181 | 6.50 | Rotdcf: Decomposition Of Convolutional Filters For Rotation-equivariant Deep Networks | 7, 6 | 0.50 |
| 182 | 6.50 | Theoretical Analysis Of Auto Rate-tuning By Batch Normalization | 7, 6 | 0.50 |
| 183 | 6.50 | Learning Recurrent Binary/ternary Weights | 8, 5 | 1.50 |
| 184 | 6.33 | Variance Reduction For Reinforcement Learning In Input-driven Environments | 6, 9, 4 | 2.05 |
| 185 | 6.33 | Execution-guided Neural Program Synthesis | 7, 5, 7 | 0.94 |
| 186 | 6.33 | Knowledge Flow: Improve Upon Your Teachers | 4, 8, 7 | 1.70 |
| 187 | 6.33 | Multi-domain Adversarial Learning | 5, 8, 6 | 1.25 |
| 188 | 6.33 | On The Convergence Of A Class Of Adam-type Algorithms For Non-convex Optimization | 6, 7, 6 | 0.47 |
| 189 | 6.33 | The Limitations Of Adversarial Training And The Blind-spot Attack | 7, 7, 5 | 0.94 |
| 190 | 6.33 | A Convergence Analysis Of Gradient Descent For Deep Linear Neural Networks | 7, 7, 5 | 0.94 |
| 191 | 6.33 | Attention, Learn To Solve Routing Problems! | 6, 6, 7 | 0.47 |
| 192 | 6.33 | Go Gradient For Expectation-based Objectives | 6, 7, 6 | 0.47 |
| 193 | 6.33 | Snas: Stochastic Neural Architecture Search | 6, 7, 6 | 0.47 |
| 194 | 6.33 | Fluctuation-dissipation Relations For Stochastic Gradient Descent | 8, 5, 6 | 1.25 |
| 195 | 6.33 | Bias-reduced Uncertainty Estimation For Deep Neural Classifiers | 7, 5, 7 | 0.94 |
| 196 | 6.33 | Regularized Learning For Domain Adaptation Under Label Shifts | 7, 6, 6 | 0.47 |
| 197 | 6.33 | Multilingual Neural Machine Translation With Knowledge Distillation | 6, 7, 6 | 0.47 |
| 198 | 6.33 | Laplacian Networks: Bounding Indicator Function Smoothness For Neural Networks Robustness | 9, 5, 5 | 1.89 |
| 199 | 6.33 | Antisymmetricrnn: A Dynamical System View On Recurrent Neural Networks | 6, 7, 6 | 0.47 |
| 200 | 6.33 | Statistical Verification Of Neural Networks | 4, 7, 8 | 1.70 |
| 201 | 6.33 | Empirical Bounds On Linear Regions Of Deep Rectifier Networks | 6, 7, 6 | 0.47 |
| 202 | 6.33 | Adaptive Estimators Show Information Compression In Deep Neural Networks | 6, 6, 7 | 0.47 |
| 203 | 6.33 | Eidetic 3d Lstm: A Model For Video Prediction And Beyond | 7, 6, 6 | 0.47 |
| 204 | 6.33 | Generative Code Modeling With Graphs | 7, 7, 5 | 0.94 |
| 205 | 6.33 | Improving Generalization And Stability Of Generative Adversarial Networks | 7, 7, 5 | 0.94 |
| 206 | 6.33 | Multilingual Neural Machine Translation With Soft Decoupled Encoding | 6, 6, 7 | 0.47 |
| 207 | 6.33 | Improved Gradient Estimators For Stochastic Discrete Variables | 7, 6, 6 | 0.47 |
| 208 | 6.33 | The Unreasonable Effectiveness Of (zero) Initialization In Deep Residual Learning | 7, 5, 7 | 0.94 |
| 209 | 6.33 | Improving Mmd-gan Training With Repulsive Loss Function | 6, 7, 6 | 0.47 |
| 210 | 6.33 | Exemplar Guided Unsupervised Image-to-image Translation With Semantic Consistency | 6, 5, 8 | 1.25 |
| 211 | 6.33 | Generating Multiple Objects At Spatially Distinct Locations | 5, 8, 6 | 1.25 |
| 212 | 6.33 | Building Dynamic Knowledge Graphs From Text Using Machine Reading Comprehension | 6, 6, 7 | 0.47 |
| 213 | 6.33 | Janossy Pooling: Learning Deep Permutation-invariant Functions For Variable-size Inputs | 7, 4, 8 | 1.70 |
| 214 | 6.33 | Generative Predecessor Models For Sample-efficient Imitation Learning | 6, 5, 8 | 1.25 |
| 215 | 6.33 | Modeling Uncertainty With Hedged Instance Embeddings | 7, 7, 5 | 0.94 |
| 216 | 6.33 | Three Mechanisms Of Weight Decay Regularization | 6, 7, 6 | 0.47 |
| 217 | 6.33 | On Computation And Generalization Of Generative Adversarial Networks Under Spectrum Control | 6, 6, 7 | 0.47 |
| 218 | 6.33 | Information Asymmetry In Kl-regularized Rl | 7, 5, 7 | 0.94 |
| 219 | 6.33 | Rnns Implicitly Implement Tensor-product Representations | 7, 6, 6 | 0.47 |
| 220 | 6.33 | Understanding Composition Of Word Embeddings Via Tensor Decomposition | 7, 6, 6 | 0.47 |
| 221 | 6.33 | Timbretron: A Wavenet(cyclegan(cqt(audio))) Pipeline For Musical Timbre Transfer | 4, 7, 8 | 1.70 |
| 222 | 6.33 | Maximal Divergence Sequential Autoencoder For Binary Software Vulnerability Detection | 7, 6, 6 | 0.47 |
| 223 | 6.33 | Multiple-attribute Text Rewriting | 7, 6, 6 | 0.47 |
| 224 | 6.33 | On The Sensitivity Of Adversarial Robustness To Input Data Distributions | 7, 5, 7 | 0.94 |
| 225 | 6.33 | Visceral Machines: Reinforcement Learning With Intrinsic Physiological Rewards | 6, 6, 7 | 0.47 |
| 226 | 6.33 | Sinkhorn Autoencoders | 6, 6, 7 | 0.47 |
| 227 | 6.33 | Learning Preconditioners On Lie Groups | 8, 4, 7 | 1.70 |
| 228 | 6.33 | Discriminator Rejection Sampling | 7, 6, 6 | 0.47 |
| 229 | 6.33 | Hierarchical Rl Using An Ensemble Of Proprioceptive Periodic Policies | 6, 6, 7 | 0.47 |
| 230 | 6.33 | Stochastic Gradient Descent Learns State Equations With Nonlinear Activations | 7, 5, 7 | 0.94 |
| 231 | 6.33 | Dialogwae: Multimodal Response Generation With Conditional Wasserstein Auto-encoder | 7, 7, 5 | 0.94 |
| 232 | 6.33 | Improving The Generalization Of Adversarial Training With Domain Adaptation | 6, 6, 7 | 0.47 |
| 233 | 6.33 | Machine Translation With Weakly Paired Bilingual Documents | 6, 5, 8 | 1.25 |
| 234 | 6.33 | Spigan: Privileged Adversarial Learning From Simulation | 6, 6, 7 | 0.47 |
| 235 | 6.33 | Principled Deep Neural Network Training Through Linear Programming | 5, 6, 8 | 1.25 |
| 236 | 6.33 | Functional Variational Bayesian Neural Networks | 7, 6, 6 | 0.47 |
| 237 | 6.33 | Universal Stagewise Learning For Non-convex Problems With Convergence On Averaged Solutions | 8, 6, 5 | 1.25 |
| 238 | 6.33 | Beyond Pixel Norm-balls: Parametric Adversaries Using An Analytically Differentiable Renderer | 7, 6, 6 | 0.47 |
| 239 | 6.33 | Are Adversarial Examples Inevitable? | 7, 8, 4 | 1.70 |
| 240 | 6.33 | Universal Transformers | 6, 6, 7 | 0.47 |
| 241 | 6.33 | On The Loss Landscape Of A Class Of Deep Neural Networks With No Bad Local Valleys | 6, 8, 5 | 1.25 |
| 242 | 6.33 | Post Selection Inference With Incomplete Maximum Mean Discrepancy Estimator | 6, 5, 8 | 1.25 |
| 243 | 6.33 | Learning Latent Superstructures In Variational Autoencoders For Deep Multidimensional Clustering | 8, 5, 6 | 1.25 |
| 244 | 6.33 | Camou: Learning Physical Vehicle Camouflages To Adversarially Attack Detectors In The Wild | 4, 8, 7 | 1.70 |
| 245 | 6.33 | Harmonizing Maximum Likelihood With Gans For Multimodal Conditional Generation | 7, 7, 5 | 0.94 |
| 246 | 6.33 | The Laplacian In Rl: Learning Representations With Efficient Approximations | 7, 5, 7 | 0.94 |
| 247 | 6.33 | Why Do Deep Convolutional Networks Generalize So Poorly To Small Image Transformations? | 7, 7, 5 | 0.94 |
| 248 | 6.33 | L-shapley And C-shapley: Efficient Model Interpretation For Structured Data | 7, 7, 5 | 0.94 |
| 249 | 6.33 | Emergent Coordination Through Competition | 7, 7, 5 | 0.94 |
| 250 | 6.33 | A Rotation And A Translation Suffice: Fooling Cnns With Simple Transformations | 8, 6, 5 | 1.25 |
| 251 | 6.33 | Toward Understanding The Impact Of Staleness In Distributed Machine Learning | 4, 9, 6 | 2.05 |
| 252 | 6.33 | Structured Adversarial Attack: Towards General Implementation And Better Interpretability | 7, 7, 5 | 0.94 |
| 253 | 6.33 | Autoencoder-based Music Translation | 8, 6, 5 | 1.25 |
| 254 | 6.33 | Don't Let Your Discriminator Be Fooled | 7, 7, 5 | 0.94 |
| 255 | 6.33 | Self-aware Visual-textual Co-grounded Navigation Agent | 8, 5, 6 | 1.25 |
| 256 | 6.33 | Deep Reinforcement Learning With Relational Inductive Biases | 5, 7, 7 | 0.94 |
| 257 | 6.33 | Attentive Neural Processes | 6, 6, 7 | 0.47 |
| 258 | 6.33 | Pate-gan: Generating Synthetic Data With Differential Privacy Guarantees | 7, 5, 7 | 0.94 |
| 259 | 6.33 | Ba-net: Dense Bundle Adjustment Networks | 9, 6, 4 | 2.05 |
| 260 | 6.33 | Fixing Posterior Collapse With Delta-vaes | 6, 7, 6 | 0.47 |
| 261 | 6.33 | Signsgd Via Zeroth-order Oracle | 8, 5, 6 | 1.25 |
| 262 | 6.33 | Contingency-aware Exploration In Reinforcement Learning | 6, 7, 6 | 0.47 |
| 263 | 6.33 | Diversity-sensitive Conditional Generative Adversarial Networks | 7, 5, 7 | 0.94 |
| 264 | 6.33 | Bayesian Policy Optimization For Model Uncertainty | 6, 6, 7 | 0.47 |
| 265 | 6.33 | Diagnosing And Enhancing Vae Models | 9, 5, 5 | 1.89 |
| 266 | 6.33 | Training For Faster Adversarial Robustness Verification Via Inducing Relu Stability | 8, 7, 4 | 1.70 |
| 267 | 6.33 | Deep Weight Prior | 4, 8, 7 | 1.70 |
| 268 | 6.33 | Neural Graph Evolution: Automatic Robot Design | 5, 8, 6 | 1.25 |
| 269 | 6.33 | Robust Estimation Via Generative Adversarial Networks | 7, 5, 7 | 0.94 |
| 270 | 6.33 | From Hard To Soft: Understanding Deep Network Nonlinearities Via Vector Quantization And Statistical Inference | 6, 6, 7 | 0.47 |
| 271 | 6.33 | Marginal Policy Gradients: A Unified Family Of Estimators For Bounded Action Spaces With Applications | 7, 6, 6 | 0.47 |
| 272 | 6.33 | A Novel Variational Family For Hidden Non-linear Markov Models | 5, 8, 6 | 1.25 |
| 273 | 6.33 | Quaternion Recurrent Neural Networks | 8, 6, 5 | 1.25 |
| 274 | 6.33 | On Self Modulation For Generative Adversarial Networks | 7, 5, 7 | 0.94 |
| 275 | 6.33 | Probabilistic Neural-symbolic Models For Interpretable Visual Question Answering | 8, 4, 7 | 1.70 |
| 276 | 6.25 | Competitive Experience Replay | 5, 7, 6, 7 | 0.83 |
| 277 | 6.25 | Efficiently Testing Local Optimality And Escaping Saddles For Relu Networks | 5, 6, 6, 8 | 1.09 |
| 278 | 6.25 | The Implicit Information In An Initial State | 6, 7, 5, 7 | 0.83 |
| 279 | 6.00 | Fortified Networks: Improving The Robustness Of Deep Networks By Modeling The Manifold Of Hidden Representations | 4, 5, 9, 6 | 1.87 |
| 280 | 6.00 | Reward Constrained Policy Optimization | 6, 7, 5 | 0.82 |
| 281 | 6.00 | Direct Optimization Through $\arg \max$ For Discrete Variational Auto-encoder | 7, 7, 4 | 1.41 |
| 282 | 6.00 | Lemonade: Learned Motif And Neuronal Assembly Detection In Calcium Imaging Videos | 4, 8 | 2.00 |
| 283 | 6.00 | Decoupled Weight Decay Regularization | 6, 7, 5 | 0.82 |
| 284 | 6.00 | On The Computational Inefficiency Of Large Batch Sizes For Stochastic Gradient Descent | 5, 8, 5 | 1.41 |
| 285 | 6.00 | Semi-supervised Learning With Multi-domain Sentiment Word Embeddings | 6, 6, 6 | 0.00 |
| 286 | 6.00 | Algorithmic Framework For Model-based Deep Reinforcement Learning With Theoretical Guarantees | 6, 6 | 0.00 |
| 287 | 6.00 | Neural Networks For Modeling Source Code Edits | 6, 6, 6 | 0.00 |
| 288 | 6.00 | Characterizing Audio Adversarial Examples Using Temporal Dependency | 6, 6 | 0.00 |
| 289 | 6.00 | On The Relation Between The Sharpest Directions Of Dnn Loss And The Sgd Step Length | 4, 6, 8 | 1.63 |
| 290 | 6.00 | Unsupervised Adversarial Image Reconstruction | 6, 8, 4 | 1.63 |
| 291 | 6.00 | Aligning Artificial Neural Networks To The Brain Yields Shallow Recurrent Architectures | 5, 7, 6 | 0.82 |
| 292 | 6.00 | Aggregated Momentum: Stability Through Passive Damping | 7, 6, 5 | 0.82 |
| 293 | 6.00 | Dynamic Early Terminating Of Multiply Accumulate Operations For Saving Computation Cost In Convolutional Neural Networks | 6, 6 | 0.00 |
| 294 | 6.00 | Lyapunov-based Safe Policy Optimization | 6, 8, 4 | 1.63 |
| 295 | 6.00 | Learning What And Where To Attend With Humans In The Loop | 6, 4, 8 | 1.63 |
| 296 | 6.00 | Learning To Learn With Conditional Class Dependencies | 6, 8, 4 | 1.63 |
| 297 | 6.00 | Multi-class Classification Without Multi-class Labels | 6, 7, 5 | 0.82 |
| 298 | 6.00 | Countering Language Drift Via Grounding | 6, 6, 6 | 0.00 |
| 299 | 6.00 | Dadam: A Consensus-based Distributed Adaptive Gradient Method For Online Optimization | 8, 4, 6 | 1.63 |
| 300 | 6.00 | Autoloss: Learning Discrete Schedule For Alternate Optimization | 7, 4, 7 | 1.41 |
| 301 | 6.00 | Supervised Policy Update | 9, 3, 6 | 2.45 |
| 302 | 6.00 | Precision Highway For Ultra Low-precision Quantization | 6, 7, 5 | 0.82 |
| 303 | 6.00 | Prior Convictions: Black-box Adversarial Attacks With Bandits And Priors | 5, 8, 5 | 1.41 |
| 304 | 6.00 | Learning Global Additive Explanations For Neural Nets Using Model Distillation | 6 | 0.00 |
| 305 | 6.00 | Image Deformation Meta-network For One-shot Learning | 5, 7, 6 | 0.82 |
| 306 | 6.00 | Coarse-grain Fine-grain Coattention Network For Multi-evidence Question Answering | 7, 7, 4 | 1.41 |
| 307 | 6.00 | Learning Kolmogorov Models For Binary Random Variables | 5, 5, 8 | 1.41 |
| 308 | 6.00 | Variational Bayesian Phylogenetic Inference | 5, 7 | 1.00 |
| 309 | 6.00 | On Tighter Generalization Bounds For Deep Neural Networks: Cnns, Resnets, And Beyond | 5, 7, 6 | 0.82 |
| 310 | 6.00 | Graph Convolutional Network With Sequential Attention For Goal-oriented Dialogue Systems | 5, 6, 7 | 0.82 |
| 311 | 6.00 | Value Propagation Networks | 7, 5, 6 | 0.82 |
| 312 | 6.00 | Learning Finite State Representations Of Recurrent Policy Networks | 6, 6, 6 | 0.00 |
| 313 | 6.00 | Self-tuning Networks: Bilevel Optimization Of Hyperparameters Using Structured Best-response Functions | 7, 6, 5 | 0.82 |
| 314 | 6.00 | A Direct Approach To Robust Deep Learning Using Adversarial Networks | 5, 7, 6 | 0.82 |
| 315 | 6.00 | Combinatorial Attacks On Binarized Neural Networks | 5, 6, 7 | 0.82 |
| 316 | 6.00 | Diverse Machine Translation With A Single Multinomial Latent Variable | 6, 5, 7 | 0.82 |
| 317 | 6.00 | Meta-learning With Latent Embedding Optimization | 5, 5, 8 | 1.41 |
| 318 | 6.00 | A Max-affine Spline Perspective Of Recurrent Neural Networks | 6, 6, 6 | 0.00 |
| 319 | 6.00 | Feed-forward Propagation In Probabilistic Neural Networks With Categorical And Max Layers | 6, 6, 6 | 0.00 |
| 320 | 6.00 | Stable Opponent Shaping In Differentiable Games | 6, 6, 6 | 0.00 |
| 321 | 6.00 | Spreading Vectors For Similarity Search | 6, 7, 5 | 0.82 |
| 322 | 6.00 | Minimal Images In Deep Neural Networks: Fragile Object Recognition In Natural Images | 6, 6, 6 | 0.00 |
| 323 | 6.00 | Representation Degeneration Problem In Training Natural Language Generation Models | 7, 6, 5 | 0.82 |
| 324 | 6.00 | Learning To Propagate Labels: Transductive Propagation Network For Few-shot Learning | 5, 6, 7 | 0.82 |
| 325 | 6.00 | Composing Entropic Policies Using Divergence Correction | 7, 5 | 1.00 |
| 326 | 6.00 | Peernets: Exploiting Peer Wisdom Against Adversarial Attacks | 7, 5 | 1.00 |
| 327 | 6.00 | Hierarchical Reinforcement Learning With Limited Policies And Hindsight | 6, 7, 5 | 0.82 |
| 328 | 6.00 | Deepobs: A Deep Learning Optimizer Benchmark Suite | 5, 6, 7 | 0.82 |
| 329 | 6.00 | Wasserstein Barycenter Model Ensembling | 6, 6, 6 | 0.00 |
| 330 | 6.00 | The Variational Deficiency Bottleneck | 5, 7, 6 | 0.82 |
| 331 | 6.00 | Multi-step Reasoning For Open-domain Question Answering | 6, 6, 6 | 0.00 |
| 332 | 6.00 | Language Modeling Teaches You More Syntax Than Translation Does: Lessons Learned Through Auxiliary Task Analysis | 6, 5, 7 | 0.82 |
| 333 | 6.00 | Dom-q-net: Grounded Rl On Structured Language | 6, 7, 5 | 0.82 |
| 334 | 6.00 | Stochastic Gradient Push For Distributed Deep Learning | 6, 6, 6 | 0.00 |
| 335 | 6.00 | Adv-bnn: Improved Adversarial Defense Through Robust Bayesian Neural Network | 7, 4, 7 | 1.41 |
| 336 | 6.00 | Learning From Positive And Unlabeled Data With A Selection Bias | 7, 6, 5 | 0.82 |
| 337 | 6.00 | On Difficulties Of Probability Distillation | 7, 5 | 1.00 |
| 338 | 6.00 | Individualized Controlled Continuous Communication Model For Multiagent Cooperative And Competitive Tasks | 6, 6 | 0.00 |
| 339 | 6.00 | Information-directed Exploration For Deep Reinforcement Learning | 6, 6, 6 | 0.00 |
| 340 | 6.00 | How To Train Your Maml | 5, 6, 7 | 0.82 |
| 341 | 6.00 | Policy Generalization In Capacity-limited Reinforcement Learning | 7, 6, 5 | 0.82 |
| 342 | 6.00 | Environment Probing Interaction Policies | 6, 6, 6 | 0.00 |
| 343 | 6.00 | A Closer Look At Few-shot Classification | 6, 6, 6 | 0.00 |
| 344 | 6.00 | Overcoming Catastrophic Forgetting Via Model Adaptation | 5, 7 | 1.00 |
| 345 | 6.00 | Manifold Mixup: Learning Better Representations By Interpolating Hidden States | 6, 4, 8 | 1.63 |
| 346 | 6.00 | Large Scale Graph Learning From Smooth Signals | 6, 5, 7 | 0.82 |
| 347 | 6.00 | Gradient-based Training Of Slow Feature Analysis By Differentiable Approximate Whitening | 6, 6 | 0.00 |
| 348 | 6.00 | Von Mises-fisher Loss For Training Sequence To Sequence Models With Continuous Outputs | 6, 7, 5 | 0.82 |
| 349 | 6.00 | Towards Understanding Regularization In Batch Normalization | 6, 6 | 0.00 |
| 350 | 6.00 | Learning Heuristics For Automated Reasoning Through Reinforcement Learning | 5, 6, 7 | 0.82 |
| 351 | 6.00 | Mean-field Analysis Of Batch Normalization | 7, 6, 5 | 0.82 |
| 352 | 6.00 | Residual Non-local Attention Networks For Image Restoration | 7, 5, 6 | 0.82 |
| 353 | 6.00 | Layoutgan: Generating Graphic Layouts With Wireframe Discriminator | 7, 5, 6 | 0.82 |
| 354 | 6.00 | Dirichlet Variational Autoencoder | 6, 5, 7 | 0.82 |
| 355 | 6.00 | Measuring Compositionality In Representation Learning | 6, 6, 6 | 0.00 |
| 356 | 6.00 | Probabilistic Planning With Sequential Monte Carlo | 8, 6, 4 | 1.63 |
| 357 | 6.00 | Detecting Out-of-distribution Samples Using Low-order Deep Features Statistics | 5, 5, 8 | 1.41 |
| 358 | 6.00 | On The Spectral Bias Of Neural Networks | 6, 5, 7 | 0.82 |
| 359 | 6.00 | Dpsnet: End-to-end Deep Plane Sweep Stereo | 7, 5, 6 | 0.82 |
| 360 | 6.00 | Adef: An Iterative Algorithm To Construct Adversarial Deformations | 7, 7, 4 | 1.41 |
| 361 | 6.00 | Delta: Deep Learning Transfer Using Feature Map With Attention For Convolutional Networks | 7, 5, 6 | 0.82 |
| 362 | 6.00 | Dimension-free Bounds For Low-precision Training | 6, 6 | 0.00 |
| 363 | 6.00 | Incremental Training Of Multi-generative Adversarial Networks | 6, 6 | 0.00 |
| 364 | 6.00 | Datnet: Dual Adversarial Transfer For Low-resource Named Entity Recognition | 6, 6 | 0.00 |
| 365 | 6.00 | Learning Protein Structure With A Differentiable Simulator | 7, 5 | 1.00 |
| 366 | 6.00 | Information Theoretic Lower Bounds On Negative Log Likelihood | 5, 7, 6 | 0.82 |
| 367 | 6.00 | Graphseq2seq: Graph-sequence-to-sequence For Neural Machine Translation | 6, 6, 6 | 0.00 |
| 368 | 6.00 | Monge-amp\`ere Flow For Generative Modeling | 7, 4, 7 | 1.41 |
| 369 | 6.00 | Mae: Mutual Posterior-divergence Regularization For Variational Autoencoders | 7, 6, 5 | 0.82 |
| 370 | 6.00 | Graph Transformer | 6, 6, 6 | 0.00 |
| 371 | 6.00 | Reasoning About Physical Interactions With Object-centric Models | 5, 5, 8 | 1.41 |
| 372 | 6.00 | Diversity And Depth In Per-example Routing Models | 6, 6 | 0.00 |
| 373 | 6.00 | Backpropamine: Training Self-modifying Neural Networks With Differentiable Neuromodulated Plasticity | 4, 9, 5 | 2.16 |
| 374 | 6.00 | Recall Traces: Backtracking Models For Efficient Reinforcement Learning | 5, 7, 6 | 0.82 |
| 375 | 6.00 | Stable Recurrent Models | 7, 5, 6 | 0.82 |
| 376 | 6.00 | Detecting Memorization In Relu Networks | 5, 4, 9 | 2.16 |
| 377 | 6.00 | Proxy-less Architecture Search Via Binarized Path Learning | 6, 6, 6 | 0.00 |
| 378 | 6.00 | Gamepad: A Learning Environment For Theorem Proving | 4, 7, 7 | 1.41 |
| 379 | 6.00 | Single Shot Neural Architecture Search Via Direct Sparse Optimization | 6, 6, 6 | 0.00 |
| 380 | 6.00 | Multi-way Encoding For Robustness To Adversarial Attacks | 6, 6, 6 | 0.00 |
| 381 | 6.00 | A Differentiable Self-disambiguated Sense Embedding Model Via Scaled Gumbel Softmax | 7, 6, 5 | 0.82 |
| 382 | 6.00 | Adversarial Vulnerability Of Neural Networks Increases With Input Dimension | 4, 9, 5 | 2.16 |
| 383 | 6.00 | Practical Lossless Compression With Latent Variables Using Bits Back Coding | 4, 8 | 2.00 |
| 384 | 6.00 | Tarmac: Targeted Multi-agent Communication | 6, 6, 6 | 0.00 |
| 385 | 6.00 | Neural Logic Machines | 6, 7, 5 | 0.82 |
| 386 | 6.00 | Neural Speed Reading With Structural-jump-lstm | 6, 5, 7 | 0.82 |
| 387 | 6.00 | Rigorous Agent Evaluation: An Adversarial Approach To Uncover Catastrophic Failures | 6, 6, 6 | 0.00 |
| 388 | 6.00 | Generating Multi-agent Trajectories Using Programmatic Weak Supervision | 6, 6 | 0.00 |
| 389 | 6.00 | Anytime Minibatch: Exploiting Stragglers In Online Distributed Optimization | 4, 7, 7 | 1.41 |
| 390 | 6.00 | Instance-aware Image-to-image Translation | 6, 6 | 0.00 |
| 391 | 6.00 | Unsupervised Hyper-alignment For Multilingual Word Embeddings | 5, 6, 7 | 0.82 |
| 392 | 6.00 | Proxquant: Quantized Neural Networks Via Proximal Operators | 8, 5, 5 | 1.41 |
| 393 | 6.00 | Optimistic Mirror Descent In Saddle-point Problems: Going The Extra(-gradient) Mile | 7, 6, 5 | 0.82 |
| 394 | 6.00 | Deep Lagrangian Networks: Using Physics As Model Prior For Deep Learning | 7, 4, 7 | 1.41 |
| 395 | 6.00 | Are Generative Classifiers More Robust To Adversarial Attacks? | 4, 8 | 2.00 |
| 396 | 6.00 | A Rotation-equivariant Convolutional Neural Network Model Of Primary Visual Cortex | 7, 3, 8 | 2.16 |
| 397 | 6.00 | Accumulation Bit-width Scaling For Ultra-low Precision Training Of Deep Networks | 6, 6, 6 | 0.00 |
| 398 | 6.00 | Variance Networks: When Expectation Does Not Meet Your Expectations | 6, 6, 6 | 0.00 |
| 399 | 6.00 | L2-nonexpansive Neural Networks | 7, 6, 5 | 0.82 |
| 400 | 6.00 | Distributional Concavity Regularization For Gans | 8, 3, 7 | 2.16 |
| 401 | 6.00 | A Systematic Study Of Binary Neural Networks' Optimisation | 8, 6, 4 | 1.63 |
| 402 | 6.00 | The Singular Values Of Convolutional Layers | 7, 4, 7 | 1.41 |
| 403 | 6.00 | Temporal Gaussian Mixture Layer For Videos | 6, 5, 7 | 0.82 |
| 404 | 6.00 | Relgan: Relational Generative Adversarial Networks For Text Generation | 6, 6, 6 | 0.00 |
| 405 | 6.00 | Analyzing Inverse Problems With Invertible Neural Networks | 6, 6, 6 | 0.00 |
| 406 | 6.00 | Babyai: First Steps Towards Grounded Language Learning With A Human In The Loop | 6, 6, 6 | 0.00 |
| 407 | 6.00 | Deep Convolutional Networks As Shallow Gaussian Processes | 5, 8, 5 | 1.41 |
| 408 | 6.00 | Sgd Converges To Global Minimum In Deep Learning Via Star-convex Path | 5, 5, 8 | 1.41 |
| 409 | 6.00 | Revealing Interpretable Object Representations From Human Behavior | 7, 7, 4 | 1.41 |
| 410 | 6.00 | Local Critic Training Of Deep Neural Networks | 6, 6 | 0.00 |
| 411 | 6.00 | Model-predictive Policy Learning With Uncertainty Regularization For Driving In Dense Traffic | 6, 5, 7 | 0.82 |
| 412 | 5.75 | Modeling Parts, Structure, And System Dynamics Via Predictive Learning | 5, 6, 7, 5 | 0.83 |
| 413 | 5.75 | Caml: Fast Context Adaptation Via Meta-learning | 4, 6, 7, 6 | 1.09 |
| 414 | 5.67 | Finite Automata Can Be Linearly Decoded From Language-recognizing Rnns | 7, 5, 5 | 0.94 |
| 415 | 5.67 | Guiding Physical Intuition With Neural Stethoscopes | 6, 4, 7 | 1.25 |
| 416 | 5.67 | Spectral Inference Networks: Unifying Deep And Spectral Learning | 5, 7, 5 | 0.94 |
| 417 | 5.67 | On The Margin Theory Of Feedforward Neural Networks | 5, 5, 7 | 0.94 |
| 418 | 5.67 | Hierarchical Interpretations For Neural Network Predictions | 7, 5, 5 | 0.94 |
| 419 | 5.67 | Discovery Of Natural Language Concepts In Individual Units | 6, 5, 6 | 0.47 |
| 420 | 5.67 | Dana: Scalable Out-of-the-box Distributed Asgd Without Retuning | 5, 7, 5 | 0.94 |
| 421 | 5.67 | Code2seq: Generating Sequences From Structured Representations Of Code | 4, 8, 5 | 1.70 |
| 422 | 5.67 | Learning To Represent Edits | 7, 4, 6 | 1.25 |
| 423 | 5.67 | Modeling The Long Term Future In Model-based Reinforcement Learning | 5, 6, 6 | 0.47 |
| 424 | 5.67 | Efficient Augmentation Via Data Subsampling | 4, 7, 6 | 1.25 |
| 425 | 5.67 | A Variational Dirichlet Framework For Out-of-distribution Detection | 6, 5, 6 | 0.47 |
| 426 | 5.67 | Augment Your Batch: Better Training With Larger Batches | 4, 8, 5 | 1.70 |
| 427 | 5.67 | Approximating Cnns With Bag-of-local-features Models Works Surprisingly Well On Imagenet | 3, 7, 7 | 1.89 |
| 428 | 5.67 | Dl2: Training And Querying Neural Networks With Logic | 7, 5, 5 | 0.94 |
| 429 | 5.67 | Fast Adversarial Training For Semi-supervised Learning | 7, 5, 5 | 0.94 |
| 430 | 5.67 | Policy Transfer With Strategy Optimization | 5, 7, 5 | 0.94 |
| 431 | 5.67 | Episodic Curiosity Through Reachability | 6, 4, 7 | 1.25 |
| 432 | 5.67 | Trace-back Along Capsules And Its Application On Semantic Segmentation | 6, 6, 5 | 0.47 |
| 433 | 5.67 | Analysing Mathematical Reasoning Abilities Of Neural Models | 5, 6, 6 | 0.47 |
| 434 | 5.67 | Explicit Information Placement On Latent Variables Using Auxiliary Generative Modelling Task | 6, 7, 4 | 1.25 |
| 435 | 5.67 | Emerging Disentanglement In Auto-encoder Based Unsupervised Image Content Transfer | 6, 6, 5 | 0.47 |
| 436 | 5.67 | A Resizable Mini-batch Gradient Descent Based On A Multi-armed Bandit | 6, 7, 4 | 1.25 |
| 437 | 5.67 | Learning Procedural Abstractions And Evaluating Discrete Latent Temporal Structure | 6, 4, 7 | 1.25 |
| 438 | 5.67 | Minimum Divergence Vs. Maximum Margin: An Empirical Comparison On Seq2seq Models | 5, 6, 6 | 0.47 |
| 439 | 5.67 | The Problem Of Model Completion | 4, 4, 9 | 2.36 |
| 440 | 5.67 | Convolutional Crfs For Semantic Segmentation | 7, 4, 6 | 1.25 |
| 441 | 5.67 | Identifying Generalization Properties In Neural Networks | 6, 5, 6 | 0.47 |
| 442 | 5.67 | Bayesian Modelling And Monte Carlo Inference For Gan | 4, 4, 9 | 2.36 |
| 443 | 5.67 | Reinforcement Learning With Perturbed Rewards | 6, 6, 5 | 0.47 |
| 444 | 5.67 | Tensor Ring Nets Adapted Deep Multi-task Learning | 6, 7, 4 | 1.25 |
| 445 | 5.67 | Mile: A Multi-level Framework For Scalable Graph Embedding | 7, 4, 6 | 1.25 |
| 446 | 5.67 | Super-resolution Via Conditional Implicit Maximum Likelihood Estimation | 5, 6, 6 | 0.47 |
| 447 | 5.67 | Laplacian Smoothing Gradient Descent | 6, 6, 5 | 0.47 |
| 448 | 5.67 | Batch-constrained Reinforcement Learning | 7, 5, 5 | 0.94 |
| 449 | 5.67 | Improving Sequence-to-sequence Learning Via Optimal Transport | 5, 7, 5 | 0.94 |
| 450 | 5.67 | Adaptive Sample-space & Adaptive Probability Coding: A Neural-network Based Approach For Compression | 5, 7, 5 | 0.94 |
| 451 | 5.67 | M^3rl: Mind-aware Multi-agent Management Reinforcement Learning | 6, 5, 6 | 0.47 |
| 452 | 5.67 | Stochastic Adversarial Video Prediction | 6, 6, 5 | 0.47 |
| 453 | 5.67 | Better Generalization With On-the-fly Dataset Denoising | 5, 6, 6 | 0.47 |
| 454 | 5.67 | Selfless Sequential Learning | 6, 6, 5 | 0.47 |
| 455 | 5.67 | An Information-theoretic Metric Of Transferability For Task Transfer Learning | 5, 6, 6 | 0.47 |
| 456 | 5.67 | Generating Liquid Simulations With Deformation-aware Neural Networks | 7, 6, 4 | 1.25 |
| 457 | 5.67 | Whitening And Coloring Transform For Gans | 5, 7, 5 | 0.94 |
| 458 | 5.67 | Unsupervised Learning Of Sentence Representations Using Sequence Consistency | 7, 5, 5 | 0.94 |
| 459 | 5.67 | Transferring Knowledge Across Learning Processes | 6, 5, 6 | 0.47 |
| 460 | 5.67 | Deterministic Pac-bayesian Generalization Bounds For Deep Networks Via Generalizing Noise-resilience | 7, 6, 4 | 1.25 |
| 461 | 5.67 | Optimal Transport Maps For Distribution Preserving Operations On Latent Spaces Of Generative Models | 7, 5, 5 | 0.94 |
| 462 | 5.67 | Function Space Particle Optimization For Bayesian Neural Networks | 5, 7, 5 | 0.94 |
| 463 | 5.67 | Feature-wise Bias Amplification | 6, 7, 4 | 1.25 |
| 464 | 5.67 | Deep Recurrent Gaussian Process With Variational Sparse Spectrum Approximation | 5, 5, 7 | 0.94 |
| 465 | 5.67 | A More Globally Accurate Dimensionality Reduction Method Using Triplets | 5, 6, 6 | 0.47 |
| 466 | 5.67 | Talk The Walk: Navigating Grids In New York City Through Grounded Dialogue | 6, 7, 4 | 1.25 |
| 467 | 5.67 | Adaptive Gradient Methods With Dynamic Bound Of Learning Rate | 7, 4, 6 | 1.25 |
| 468 | 5.67 | Adversarial Exploration Strategy For Self-supervised Imitation Learning | 7, 5, 5 | 0.94 |
| 469 | 5.67 | Cross-task Knowledge Transfer For Visually-grounded Navigation | 7, 5, 5 | 0.94 |
| 470 | 5.67 | Adversarially Learned Mixture Model | 6, 5, 6 | 0.47 |
| 471 | 5.67 | Subgradient Descent Learns Orthogonal Dictionaries | 7, 7, 3 | 1.89 |
| 472 | 5.67 | Cross-entropy Loss Leads To Poor Margins | 8, 5, 4 | 1.70 |
| 473 | 5.67 | Graph U-net | 6, 4, 7 | 1.25 |
| 474 | 5.67 | Imagenet-trained Cnns Are Biased Towards Texture; Increasing Shape Bias Improves Accuracy And Robustness. | 4, 7, 6 | 1.25 |
| 475 | 5.67 | State-regularized Recurrent Networks | 6, 6, 5 | 0.47 |
| 476 | 5.67 | Data-dependent Coresets For Compressing Neural Networks With Applications To Generalization Bounds | 6, 6, 5 | 0.47 |
| 477 | 5.67 | Interactive Agent Modeling By Learning To Probe | 6, 6, 5 | 0.47 |
| 478 | 5.67 | Clean-label Backdoor Attacks | 6, 7, 4 | 1.25 |
| 479 | 5.67 | Learning To Augment Influential Data | 6, 6, 5 | 0.47 |
| 480 | 5.67 | (unconstrained) Beam Search Is Sensitive To Large Search Discrepancies | 5, 5, 7 | 0.94 |
| 481 | 5.67 | Top-down Neural Model For Formulae | 6, 6, 5 | 0.47 |
| 482 | 5.67 | Learning To Design Rna | 5, 6, 6 | 0.47 |
| 483 | 5.67 | Adversarial Imitation Via Variational Inverse Reinforcement Learning | 5, 7, 5 | 0.94 |
| 484 | 5.67 | Efficient Multi-objective Neural Architecture Search Via Lamarckian Evolution | 6, 6, 5 | 0.47 |
| 485 | 5.67 | Accelerating Nonconvex Learning Via Replica Exchange Langevin Diffusion | 4, 7, 6 | 1.25 |
| 486 | 5.67 | Bnn+: Improved Binary Network Training | 6, 6, 5 | 0.47 |
| 487 | 5.67 | Ppd: Permutation Phase Defense Against Adversarial Examples In Deep Learning | 6, 7, 4 | 1.25 |
| 488 | 5.67 | Adaptive Posterior Learning | 6, 7, 4 | 1.25 |
| 489 | 5.67 | Remember And Forget For Experience Replay | 6, 5, 6 | 0.47 |
| 490 | 5.67 | Efficient Lifelong Learning With A-gem | 6, 6, 5 | 0.47 |
| 491 | 5.67 | Learning Implicit Generative Models By Teaching Explicit Ones | 7, 5, 5 | 0.94 |
| 492 | 5.67 | Overcoming The Disentanglement Vs Reconstruction Trade-off Via Jacobian Supervision | 7, 6, 4 | 1.25 |
| 493 | 5.67 | Reliable Uncertainty Estimates In Deep Neural Networks Using Noise Contrastive Priors | 7, 4, 6 | 1.25 |
| 494 | 5.67 | Adversarial Audio Synthesis | 5, 6, 6 | 0.47 |
| 495 | 5.67 | Alignment Based Mathching Networks For One-shot Classification And Open-set Recognition | 6, 7, 4 | 1.25 |
| 496 | 5.67 | Signsgd With Majority Vote Is Communication Efficient And Byzantine Fault Tolerant | 5, 5, 7 | 0.94 |
| 497 | 5.67 | Projective Subspace Networks For Few Shot Learning | 6, 5, 6 | 0.47 |
| 498 | 5.67 | Texttovec: Deep Contextualized Neural Autoregressive Models Of Language With Distributed Compositional Prior | 6, 6, 5 | 0.47 |
| 499 | 5.67 | Nested Dithered Quantization For Communication Reduction In Distributed Training | 5, 5, 7 | 0.94 |
| 500 | 5.67 | The Meaning Of "most" For Visual Question Answering Models | 7, 5, 5 | 0.94 |
| 501 | 5.67 | Controlling Covariate Shift Using Equilibrium Normalization Of Weights | 4, 6, 7 | 1.25 |
| 502 | 5.67 | Kernel Recurrent Learning (kerl) | 5, 6, 6 | 0.47 |
| 503 | 5.67 | Amortized Context Vector Inference For Sequence-to-sequence Networks | 6, 6, 5 | 0.47 |
| 504 | 5.67 | A New Dog Learns Old Tricks: Rl Finds Classic Optimization Algorithms | 5, 5, 7 | 0.94 |
| 505 | 5.67 | Unsupervised Neural Multi-document Abstractive Summarization | 5, 3, 9 | 2.49 |
| 506 | 5.67 | Relational Forward Models For Multi-agent Learning | 6, 6, 5 | 0.47 |
| 507 | 5.67 | Learning Discrete Wasserstein Embeddings | 7, 7, 3 | 1.89 |
| 508 | 5.67 | Large-scale Answerer In Questioner's Mind For Visual Dialog Question Generation | 5, 6, 6 | 0.47 |
| 509 | 5.67 | Efficient Codebook And Factorization For Second Order Representation Learning | 6, 6, 5 | 0.47 |
| 510 | 5.67 | Max-mig: An Information Theoretic Approach For Joint Learning From Crowds | 6, 5, 6 | 0.47 |
| 511 | 5.67 | Bayesian Convolutional Neural Networks With Many Channels Are Gaussian Processes | 8, 3, 6 | 2.05 |
| 512 | 5.67 | Learning Multimodal Graph-to-graph Translation For Molecule Optimization | 7, 4, 6 | 1.25 |
| 513 | 5.67 | Domain Adaptation For Structured Output Via Disentangled Patch Representations | 7, 5, 5 | 0.94 |
| 514 | 5.67 | Doubly Sparse: Sparse Mixture Of Sparse Experts For Efficient Softmax Inference | 6, 4, 7 | 1.25 |
| 515 | 5.67 | Open-ended Content-style Recombination Via Leakage Filtering | 7, 5, 5 | 0.94 |
| 516 | 5.67 | Learning Exploration Policies For Navigation | 3, 7, 7 | 1.89 |
| 517 | 5.67 | Rethinking Knowledge Graph Propagation For Zero-shot Learning | 7, 5, 5 | 0.94 |
| 518 | 5.67 | Generative Feature Matching Networks | 6, 5, 6 | 0.47 |
| 519 | 5.67 | Overcoming Neural Brainwashing | 6, 5, 6 | 0.47 |
| 520 | 5.67 | Unsupervised Control Through Non-parametric Discriminative Rewards | 6, 5, 6 | 0.47 |
| 521 | 5.67 | Explaining Image Classifiers By Counterfactual Generation | 5, 7, 5 | 0.94 |
| 522 | 5.67 | Language Model Pre-training For Hierarchical Document Representations | 5, 6, 6 | 0.47 |
| 523 | 5.67 | Hierarchical Generative Modeling For Controllable Speech Synthesis | 6, 6, 5 | 0.47 |
| 524 | 5.67 | Out-of-sample Extrapolation With Neuron Editing | 5, 6, 6 | 0.47 |
| 525 | 5.67 | Adversarial Attacks On Node Embeddings | 5, 6, 6 | 0.47 |
| 526 | 5.67 | Ppo-cma: Proximal Policy Optimization With Covariance Matrix Adaptation | 4, 9, 4 | 2.36 |
| 527 | 5.67 | Predicted Variables In Programming | 5, 5, 7 | 0.94 |
| 528 | 5.67 | Manifold Regularization With Gans For Semi-supervised Learning | 7, 5, 5 | 0.94 |
| 529 | 5.67 | Small Steps And Giant Leaps: Minimal Newton Solvers For Deep Learning | 7, 7, 3 | 1.89 |
| 530 | 5.67 | A Kernel Random Matrix-based Approach For Sparse Pca | 5, 7, 5 | 0.94 |
| 531 | 5.67 | Recurrent Kalman Networks: Factorized Inference In High-dimensional Deep Feature Spaces | 6, 6, 5 | 0.47 |
| 532 | 5.67 | Visual Reasoning By Progressive Module Networks | 6, 7, 4 | 1.25 |
| 533 | 5.67 | Hallucinations In Neural Machine Translation | 6, 4, 7 | 1.25 |
| 534 | 5.67 | Shallow Learning For Deep Networks | 5, 5, 7 | 0.94 |
| 535 | 5.67 | Learning Models For Visual 3d Localization With Implicit Mapping | 7, 5, 5 | 0.94 |
| 536 | 5.67 | Codraw: Collaborative Drawing As A Testbed For Grounded Goal-driven Communication | 4, 6, 7 | 1.25 |
| 537 | 5.67 | Rethinking The Value Of Network Pruning | 6, 5, 6 | 0.47 |
| 538 | 5.67 | Deep Denoising: Rate-optimal Recovery Of Structured Signals With A Deep Prior | 6, 5, 6 | 0.47 |
| 539 | 5.67 | Variational Autoencoder With Arbitrary Conditioning | 7, 6, 4 | 1.25 |
| 540 | 5.67 | Attentive Task-agnostic Meta-learning For Few-shot Text Classification | 5, 5, 7 | 0.94 |
| 541 | 5.67 | Backprop With Approximate Activations For Memory-efficient Network Training | 5, 7, 5 | 0.94 |
| 542 | 5.67 | Rotation Equivariant Networks Via Conic Convolution And The Dft | 4, 7, 6 | 1.25 |
| 543 | 5.67 | Transfer Learning For Related Reinforcement Learning Tasks Via Image-to-image Translation | 7, 7, 3 | 1.89 |
| 544 | 5.67 | Understanding Gans Via Generalization Analysis For Disconnected Support | 6, 5, 6 | 0.47 |
| 545 | 5.67 | Learning Neural Random Fields With Inclusive Auxiliary Generators | 6, 6, 5 | 0.47 |
| 546 | 5.67 | A Closer Look At Deep Learning Heuristics: Learning Rate Restarts, Warmup And Distillation | 4, 7, 6 | 1.25 |
| 547 | 5.67 | Deep Probabilistic Video Compression | 6, 5, 6 | 0.47 |
| 548 | 5.67 | Learning Disentangled Representations With Reference-based Variational Autoencoders | 6, 6, 5 | 0.47 |
| 549 | 5.67 | Learning What You Can Do Before Doing Anything | 6, 5, 6 | 0.47 |
| 550 | 5.67 | Unsupervised Learning Via Meta-learning | 4, 8, 5 | 1.70 |
| 551 | 5.67 | Adaptive Network Sparsification Via Dependent Variational Beta-bernoulli Dropout | 5, 5, 7 | 0.94 |
| 552 | 5.67 | A Biologically Inspired Visual Working Memory For Deep Networks | 4, 5, 8 | 1.70 |
| 553 | 5.67 | Cnnsat: Fast, Accurate Boolean Satisfiability Using Convolutional Neural Networks | 4, 8, 5 | 1.70 |
| 554 | 5.67 | Beyond Greedy Ranking: Slate Optimization Via List-cvae | 6, 4, 7 | 1.25 |
| 555 | 5.67 | Discriminative Active Learning | 8, 5, 4 | 1.70 |
| 556 | 5.67 | Learning To Make Analogies By Contrasting Abstract Relational Structure | 5, 5, 7 | 0.94 |
| 557 | 5.67 | Transfer Learning Via Unsupervised Task Discovery For Visual Question Answering | 4, 5, 8 | 1.70 |
| 558 | 5.67 | Zero-resource Multilingual Model Transfer: Learning What To Share | 6, 5, 6 | 0.47 |
| 559 | 5.50 | Interpretable Continual Learning | 5, 6 | 0.50 |
| 560 | 5.50 | Isa-vae: Independent Subspace Analysis With Variational Autoencoders | 7, 4 | 1.50 |
| 561 | 5.50 | Scalable Unbalanced Optimal Transport Using Generative Adversarial Networks | 6, 5 | 0.50 |
| 562 | 5.50 | Generative Adversarial Models For Learning Private And Fair Representations | 4, 7 | 1.50 |
| 563 | 5.50 | Dynamic Channel Pruning: Feature Boosting And Suppression | 7, 4 | 1.50 |
| 564 | 5.50 | Causal Reasoning From Meta-learning | 4, 7 | 1.50 |
| 565 | 5.50 | N-ary Quantization For Cnn Model Compression And Inference Acceleration | 4, 7 | 1.50 |
| 566 | 5.50 | Probabilistic Recursive Reasoning For Multi-agent Reinforcement Learning | 6, 5 | 0.50 |
| 567 | 5.50 | Unified Recurrent Network For Many Feature Types | 4, 7 | 1.50 |
| 568 | 5.50 | Low Latency Privacy Preserving Inference | 6, 5 | 0.50 |
| 569 | 5.50 | Perception-aware Point-based Value Iteration For Partially Observable Markov Decision Processes | 4, 7 | 1.50 |
| 570 | 5.50 | The Universal Approximation Power Of Finite-width Deep Relu Networks | 5, 6 | 0.50 |
| 571 | 5.50 | Tangent-normal Adversarial Regularization For Semi-supervised Learning | 4, 7 | 1.50 |
| 572 | 5.50 | Accelerated Gradient Flow For Probability Distributions | 5, 6 | 0.50 |
| 573 | 5.50 | Per-tensor Fixed-point Quantization Of The Back-propagation Algorithm | 3, 8 | 2.50 |
| 574 | 5.50 | Scalable Neural Theorem Proving On Knowledge Bases And Natural Language | 5, 6 | 0.50 |
| 575 | 5.50 | Mahinet: A Neural Network For Many-class Few-shot Learning With Class Hierarchy | 6, 5 | 0.50 |
| 576 | 5.50 | Integer Networks For Data Compression With Latent-variable Models | 4, 7 | 1.50 |
| 577 | 5.50 | Necst: Neural Joint Source-channel Coding | 4, 7 | 1.50 |
| 578 | 5.50 | Hindsight Policy Gradients | 5, 6 | 0.50 |
| 579 | 5.50 | Improved Learning Of One-hidden-layer Convolutional Neural Networks With Overlaps | 5, 6 | 0.50 |
| 580 | 5.50 | Knowledge Representation For Reinforcement Learning Using General Value Functions | 7, 4 | 1.50 |
| 581 | 5.50 | An Empirical Study Of Example Forgetting During Deep Neural Network Learning | 5, 6 | 0.50 |
| 582 | 5.50 | Interpreting Adversarial Robustness: A View From Decision Surface In Input Space | 6, 5 | 0.50 |
| 583 | 5.50 | Universal Successor Features Approximators | 5, 6 | 0.50 |
| 584 | 5.50 | Coverage And Quality Driven Training Of Generative Image Models | 4, 7 | 1.50 |
| 585 | 5.50 | Policy Optimization Via Stochastic Recursive Gradient Algorithm | 6, 5 | 0.50 |
| 586 | 5.50 | Automata Guided Skill Composition | 6, 5 | 0.50 |
| 587 | 5.50 | Surprising Negative Results For Generative Adversarial Tree Search | 5, 6 | 0.50 |
| 588 | 5.50 | Cramer-wold Autoencoder | 6, 5 | 0.50 |
| 589 | 5.50 | Seq2slate: Re-ranking And Slate Optimization With Rnns | 6, 5 | 0.50 |
| 590 | 5.50 | Mixed Precision Quantization Of Convnets Via Differentiable Neural Architecture Search | 5, 6 | 0.50 |
| 591 | 5.33 | Unseen Action Recognition With Multimodal Learning | 7, 5, 4 | 1.25 |
| 592 | 5.33 | Bliss In Non-isometric Embedding Spaces | 4, 6, 6 | 0.94 |
| 593 | 5.33 | Meta Learning With Fast/slow Learners | 5, 6, 5 | 0.47 |
| 594 | 5.33 | Mimicking Actions Is A Good Strategy For Beginners: Fast Reinforcement Learning With Expert Action Sequences | 5, 5, 6 | 0.47 |
| 595 | 5.33 | Adapting Auxiliary Losses Using Gradient Similarity | 4, 6, 6 | 0.94 |
| 596 | 5.33 | Pathologies In Information Bottleneck For Deterministic Supervised Learning | 2, 8, 6 | 2.49 |
| 597 | 5.33 | Knows When It Doesn’t Know: Deep Abstaining Classifiers | 6, 5, 5 | 0.47 |
| 598 | 5.33 | An Experimental Study Of Layer-level Training Speed And Its Impact On Generalization | 6, 5, 5 | 0.47 |
| 599 | 5.33 | Lipschitz Regularized Deep Neural Networks Converge And Generalize | 4, 6, 6 | 0.94 |
| 600 | 5.33 | Generative Adversarial Networks For Extreme Learned Image Compression | 6, 6, 4 | 0.94 |
| 601 | 5.33 | Formal Limitations On The Measurement Of Mutual Information | 7, 5, 4 | 1.25 |
| 602 | 5.33 | Stochastic Gradient/mirror Descent: Minimax Optimality And Implicit Regularization | 7, 4, 5 | 1.25 |
| 603 | 5.33 | Generalizable Adversarial Training Via Spectral Normalization | 6, 5, 5 | 0.47 |
| 604 | 5.33 | Learning-based Frequency Estimation Algorithms | 7, 3, 6 | 1.70 |
| 605 | 5.33 | Exploring The Interpretability Of Lstm Neural Networks Over Multi-variable Data | 6, 5, 5 | 0.47 |
| 606 | 5.33 | Improving Sentence Representations With Multi-view Frameworks | 7, 5, 4 | 1.25 |
| 607 | 5.33 | Probabilistic Knowledge Graph Embeddings | 5, 6, 5 | 0.47 |
| 608 | 5.33 | Decaynet: A Study On The Cell States Of Long Short Term Memories | 8, 4, 4 | 1.89 |
| 609 | 5.33 | Can I Trust You More? Model-agnostic Hierarchical Explanations | 6, 6, 4 | 0.94 |
| 610 | 5.33 | Learning To Coordinate Multiple Reinforcement Learning Agents For Diverse Query Reformulation | 4, 7, 5 | 1.25 |
| 611 | 5.33 | Knowledge Distillation From Few Samples | 4, 6, 6 | 0.94 |
| 612 | 5.33 | Graph Matching Networks For Learning The Similarity Of Graph Structured Objects | 5, 5, 6 | 0.47 |
| 613 | 5.33 | An Active Learning Framework For Efficient Robust Policy Search | 5, 6, 5 | 0.47 |
| 614 | 5.33 | Training Generative Latent Models By Variational F-divergence Minimization | 6, 5, 5 | 0.47 |
| 615 | 5.33 | Mitigating Bias In Natural Language Inference Using Adversarial Learning | 4, 4, 8 | 1.89 |
| 616 | 5.33 | A Unified Theory Of Adaptive Stochastic Gradient Descent As Bayesian Filtering | 4, 7, 5 | 1.25 |
| 617 | 5.33 | Domain Adaptation Via Distribution And Representation Matching: A Case Study On Training Data Selection Via Reinforcement Learning | 4, 7, 5 | 1.25 |
| 618 | 5.33 | Learning Generative Models For Demixing Of Structured Signals From Their Superposition Using Gans | 4, 5, 7 | 1.25 |
| 619 | 5.33 | Learning To Encode Spatial Relations From Natural Language | 6, 5, 5 | 0.47 |
| 620 | 5.33 | Meta-learning Language-guided Policy Learning | 6, 6, 4 | 0.94 |
| 621 | 5.33 | Hint-based Training For Non-autoregressive Translation | 6, 6, 4 | 0.94 |
| 622 | 5.33 | Adaptive Pruning Of Neural Language Models For Mobile Devices | 6, 5, 5 | 0.47 |
| 623 | 5.33 | Meta-learning For Contextual Bandit Exploration | 7, 6, 3 | 1.70 |
| 624 | 5.33 | Learning Domain-invariant Representation Under Domain-class Dependency | 4, 7, 5 | 1.25 |
| 625 | 5.33 | Towards Decomposed Linguistic Representation With Holographic Reduced Representation | 5, 5, 6 | 0.47 |
| 626 | 5.33 | Adaptive Neural Trees | 4, 5, 7 | 1.25 |
| 627 | 5.33 | Engan: Latent Space Mcmc And Maximum Entropy Generators For Energy-based Models | 6, 5, 5 | 0.47 |
| 628 | 5.33 | Cem-rl: Combining Evolutionary And Gradient-based Methods For Policy Search | 3, 7, 6 | 1.70 |
| 629 | 5.33 | Nlprolog: Reasoning With Weak Unification For Natural Language Question Answering | 6, 4, 6 | 0.94 |
| 630 | 5.33 | Set Transformer | 5, 6, 5 | 0.47 |
| 631 | 5.33 | Lorentzian Distance Learning | 6, 5, 5 | 0.47 |
| 632 | 5.33 | The Relativistic Discriminator: A Key Element Missing From Standard Gan | 3, 6, 7 | 1.70 |
| 633 | 5.33 | Learning Mixed-curvature Representations In Product Spaces | 7, 2, 7 | 2.36 |
| 634 | 5.33 | Roc-gan: Robust Conditional Gan | 6, 5, 5 | 0.47 |
| 635 | 5.33 | Dher: Hindsight Experience Replay For Dynamic Goals | 5, 7, 4 | 1.25 |
| 636 | 5.33 | Large-scale Visual Speech Recognition | 4, 3, 9 | 2.62 |
| 637 | 5.33 | A Modern Take On The Bias-variance Tradeoff In Neural Networks | 5, 7, 4 | 1.25 |
| 638 | 5.33 | Amortized Bayesian Meta-learning | 6, 5, 5 | 0.47 |
| 639 | 5.33 | Gaussian-gated Lstm: Improved Convergence By Reducing State Updates | 5, 5, 6 | 0.47 |
| 640 | 5.33 | Hierarchically-structured Variational Autoencoders For Long Text Generation | 5, 5, 6 | 0.47 |
| 641 | 5.33 | Generalization And Regularization In Dqn | 6, 5, 5 | 0.47 |
| 642 | 5.33 | Cost-sensitive Robustness Against Adversarial Examples | 4, 5, 7 | 1.25 |
| 643 | 5.33 | Reducing Overconfident Errors Outside The Known Distribution | 6, 4, 6 | 0.94 |
| 644 | 5.33 | Eddi: Efficient Dynamic Discovery Of High-value Information With Partial Vae | 5, 5, 6 | 0.47 |
| 645 | 5.33 | State-denoised Recurrent Neural Networks | 6, 5, 5 | 0.47 |
| 646 | 5.33 | Local Binary Pattern Networks For Character Recognition | 5, 6, 5 | 0.47 |
| 647 | 5.33 | Aim: Adversarial Inference By Matching Priors And Conditionals | 7, 4, 5 | 1.25 |
| 648 | 5.33 | Diffranet: Automatic Classification Of Serial Crystallography Diffraction Patterns | 5, 3, 8 | 2.05 |
| 649 | 5.33 | Exploring Curvature Noise In Large-batch Stochastic Optimization | 5, 6, 5 | 0.47 |
| 650 | 5.33 | Actor-attention-critic For Multi-agent Reinforcement Learning | 6, 6, 4 | 0.94 |
| 651 | 5.33 | Equi-normalization Of Neural Networks | 6, 5, 5 | 0.47 |
| 652 | 5.33 | Interpolation-prediction Networks For Irregularly Sampled Time Series | 6, 5, 5 | 0.47 |
| 653 | 5.33 | The Case For Full-matrix Adaptive Regularization | 6, 5, 5 | 0.47 |
| 654 | 5.33 | Switching Linear Dynamics For Variational Bayes Filtering | 5, 4, 7 | 1.25 |
| 655 | 5.33 | On The Turing Completeness Of Modern Neural Network Architectures | 4, 5, 7 | 1.25 |
| 656 | 5.33 | Improving Composition Of Sentence Embeddings Through The Lens Of Statistical Relational Learning | 5, 5, 6 | 0.47 |
| 657 | 5.33 | Synonymnet: Multi-context Bilateral Matching For Entity Synonyms | 5, 7, 4 | 1.25 |
| 658 | 5.33 | Verification Of Non-linear Specifications For Neural Networks | 7, 5, 4 | 1.25 |
| 659 | 5.33 | Ib-gan: Disentangled Representation Learning With Information Bottleneck Gan | 6, 6, 4 | 0.94 |
| 660 | 5.33 | Probabilistic Model-based Dynamic Architecture Search | 5, 6, 5 | 0.47 |
| 661 | 5.33 | Advocacy Learning | 4, 4, 8 | 1.89 |
| 662 | 5.33 | On The Ineffectiveness Of Variance Reduced Optimization For Deep Learning | 5, 6, 5 | 0.47 |
| 663 | 5.33 | Skip-gram Word Embeddings In Hyperbolic Space | 5, 5, 6 | 0.47 |
| 664 | 5.33 | Unsupervised Conditional Generation Using Noise Engineered Mode Matching Gan | 5, 5, 6 | 0.47 |
| 665 | 5.33 | Personalized Embedding Propagation: Combining Neural Networks On Graphs With Personalized Pagerank | 4, 5, 7 | 1.25 |
| 666 | 5.33 | Clinical Risk: Wavelet Reconstruction Networks For Marked Point Processes | 7, 4, 5 | 1.25 |
| 667 | 5.33 | I Know The Feeling: Learning To Converse With Empathy | 4, 7, 5 | 1.25 |
| 668 | 5.33 | Distribution-interpolation Trade Off In Generative Models | 4, 7, 5 | 1.25 |
| 669 | 5.33 | Human-level Protein Localization With Convolutional Neural Networks | 3, 5, 8 | 2.05 |
| 670 | 5.33 | Heated-up Softmax Embedding | 8, 3, 5 | 2.05 |
| 671 | 5.33 | Infinitely Deep Infinite-width Networks | 5, 6, 5 | 0.47 |
| 672 | 5.33 | Learning To Describe Scenes With Programs | 6, 4, 6 | 0.94 |
| 673 | 5.33 | Soft Q-learning With Mutual-information Regularization | 6, 6, 4 | 0.94 |
| 674 | 5.33 | Learning Space Time Dynamics With Pde Guided Neural Networks | 6, 5, 5 | 0.47 |
| 675 | 5.33 | Online Hyperparameter Adaptation Via Amortized Proximal Optimization | 6, 5, 5 | 0.47 |
| 676 | 5.33 | Computing Committor Functions For The Study Of Rare Events Using Deep Learning With Importance Sampling | 4, 5, 7 | 1.25 |
| 677 | 5.33 | Neural Predictive Belief Representations | 4, 7, 5 | 1.25 |
| 678 | 5.33 | Graph Wavelet Neural Network | 4, 6, 6 | 0.94 |
| 679 | 5.33 | Antman: Sparse Low-rank Compression To Accelerate Rnn Inference | 6, 5, 5 | 0.47 |
| 680 | 5.33 | Structured Neural Summarization | 6, 6, 4 | 0.94 |
| 681 | 5.33 | Learning Graph Decomposition | 7, 4, 5 | 1.25 |
| 682 | 5.33 | Curiosity-driven Experience Prioritization Via Density Estimation | 6, 4, 6 | 0.94 |
| 683 | 5.33 | Understanding Straight-through Estimator In Training Activation Quantized Neural Nets | 5, 6, 5 | 0.47 |
| 684 | 5.33 | Learning To Generate Parameters From Natural Languages For Graph Neural Networks | 4, 6, 6 | 0.94 |
| 685 | 5.33 | Neural Model-based Reinforcement Learning For Recommendation | 5, 6, 5 | 0.47 |
| 686 | 5.33 | Search-guided, Lightly-supervised Training Of Structured Prediction Energy Networks | 5, 7, 4 | 1.25 |
| 687 | 5.33 | Hierarchically Clustered Representation Learning | 5, 5, 6 | 0.47 |
| 688 | 5.33 | Uncovering Surprising Behaviors In Reinforcement Learning Via Worst-case Analysis | 5, 5, 6 | 0.47 |
| 689 | 5.33 | Learning Backpropagation-free Deep Architectures With Kernels | 6, 5, 5 | 0.47 |
| 690 | 5.33 | Learning To Separate Domains In Generalized Zero-shot And Open Set Learning: A Probabilistic Perspective | 5, 5, 6 | 0.47 |
| 691 | 5.33 | Exploring And Enhancing The Transferability Of Adversarial Examples | 4, 6, 6 | 0.94 |
| 692 | 5.33 | Neural Causal Discovery With Learnable Input Noise | 4, 4, 8 | 1.89 |
| 693 | 5.33 | Cdeepex: Contrastive Deep Explanations | 5, 6, 5 | 0.47 |
| 694 | 5.33 | Optimal Control Via Neural Networks: A Convex Approach | 1, 8, 7 | 3.09 |
| 695 | 5.33 | Cbow Is Not All You Need: Combining Cbow With The Compositional Matrix Space Model | 6, 5, 5 | 0.47 |
| 696 | 5.33 | Purchase As Reward : Session-based Recommendation By Imagination Reconstruction | 5, 6, 5 | 0.47 |
| 697 | 5.33 | Learning Particle Dynamics For Manipulating Rigid Bodies, Deformable Objects, And Fluids | 5, 5, 6 | 0.47 |
| 698 | 5.33 | Convolutional Neural Networks On Non-uniform Geometrical Signals Using Euclidean Spectral Transformation | 5, 7, 4 | 1.25 |
| 699 | 5.33 | Soseleto: A Unified Approach To Transfer Learning And Training With Noisy Labels | 6, 5, 5 | 0.47 |
| 700 | 5.33 | Universal Lipschitz Functions | 7, 5, 4 | 1.25 |
| 701 | 5.33 | Universal Successor Features For Transfer Reinforcement Learning | 4, 7, 5 | 1.25 |
| 702 | 5.33 | Random Mesh Projectors For Inverse Problems | 5, 7, 4 | 1.25 |
| 703 | 5.33 | Classification From Positive, Unlabeled And Biased Negative Data | 5, 6, 5 | 0.47 |
| 704 | 5.33 | Deep Anomaly Detection With Outlier Exposure | 4, 4, 8 | 1.89 |
| 705 | 5.33 | Deep Graph Translation | 5, 5, 6 | 0.47 |
| 706 | 5.33 | Learning Protein Sequence Embeddings Using Information From Structure | 5, 4, 7 | 1.25 |
| 707 | 5.33 | Deep Learning Generalizes Because The Parameter-function Map Is Biased Towards Simple Functions | 7, 5, 4 | 1.25 |
| 708 | 5.33 | Invariant-covariant Representation Learning | 7, 5, 4 | 1.25 |
| 709 | 5.33 | Negotiating Team Formation Using Deep Reinforcement Learning | 5, 6, 5 | 0.47 |
| 710 | 5.33 | Multi-task Learning With Gradient Communication | 5, 4, 7 | 1.25 |
| 711 | 5.33 | Learning Representations Of Sets Through Optimised Permutations | 5, 5, 6 | 0.47 |
| 712 | 5.33 | Open Loop Hyperparameter Optimization And Determinantal Point Processes | 5, 6, 5 | 0.47 |
| 713 | 5.33 | Identifying Bias In Ai Using Simulation | 4, 6, 6 | 0.94 |
| 714 | 5.33 | Learning State Representations In Complex Systems With Multimodal Data | 5, 6, 5 | 0.47 |
| 715 | 5.33 | Provable Guarantees On Learning Hierarchical Generative Models With Deep Cnns | 6, 6, 4 | 0.94 |
| 716 | 5.33 | Estimating Information Flow In Dnns | 6, 6, 4 | 0.94 |
| 717 | 5.33 | A Deep Learning Approach For Dynamic Survival Analysis With Competing Risks | 4, 8, 4 | 1.89 |
| 718 | 5.33 | Learning Factorized Representations For Open-set Domain Adaptation | 6, 4, 6 | 0.94 |
| 719 | 5.33 | Learning Data-derived Privacy Preserving Representations From Information Metrics | 6, 5, 5 | 0.47 |
| 720 | 5.33 | Understanding & Generalizing Alphago Zero | 5, 7, 4 | 1.25 |
| 721 | 5.33 | Complementary-label Learning For Arbitrary Losses And Models | 5, 5, 6 | 0.47 |
| 722 | 5.33 | Perfect Match: A Simple Method For Learning Representations For Counterfactual Inference With Neural Networks | 5, 5, 6 | 0.47 |
| 723 | 5.33 | Learning Internal Dense But External Sparse Structures Of Deep Neural Network | 5, 5, 6 | 0.47 |
| 724 | 5.33 | Transformer-xl: Language Modeling With Longer-term Dependency | 6, 6, 4 | 0.94 |
| 725 | 5.33 | Meta Domain Adaptation: Meta-learning For Few-shot Learning Under Domain Shift | 6, 4, 6 | 0.94 |
| 726 | 5.33 | The Unusual Effectiveness Of Averaging In Gan Training | 5, 6, 5 | 0.47 |
| 727 | 5.33 | Latent Convolutional Models | 6, 7, 3 | 1.70 |
| 728 | 5.33 | Playing The Game Of Universal Adversarial Perturbations | 6, 5, 5 | 0.47 |
| 729 | 5.33 | Graph Classification With Geometric Scattering | 5, 6, 5 | 0.47 |
| 730 | 5.33 | Imitative Models: Perception-driven Forecasting For Flexible Planning And Control | 5, 5, 6 | 0.47 |
| 731 | 5.33 | Learning To Decompose Compound Questions With Reinforcement Learning | 6, 5, 5 | 0.47 |
| 732 | 5.33 | H-detach: Modifying The Lstm Gradient Towards Better Optimization | 5, 6, 5 | 0.47 |
| 733 | 5.33 | Stackelberg Gan: Towards Provable Minimax Equilibrium Via Multi-generator Architectures | 5, 7, 4 | 1.25 |
| 734 | 5.33 | Don’t Judge A Book By Its Cover - On The Dynamics Of Recurrent Neural Networks | 5, 7, 4 | 1.25 |
| 735 | 5.33 | Learning Cross-lingual Sentence Representations Via A Multi-task Dual-encoder Model | 6, 4, 6 | 0.94 |
| 736 | 5.33 | The Loss Landscape Of Overparameterized Neural Networks | 4, 7, 5 | 1.25 |
| 737 | 5.33 | Probabilistic Federated Neural Matching | 4, 6, 6 | 0.94 |
| 738 | 5.33 | Tree-structured Recurrent Switching Linear Dynamical Systems For Multi-scale Modeling | 6, 5, 5 | 0.47 |
| 739 | 5.25 | Sample Efficient Imitation Learning For Continuous Control | 5, 6, 5, 5 | 0.43 |
| 740 | 5.25 | P^2ir: Universal Deep Node Representation Via Partial Permutation Invariant Set Functions | 4, 7, 5, 5 | 1.09 |
| 741 | 5.25 | Improving Generative Adversarial Imitation Learning With Non-expert Demonstrations | 5, 5, 7, 4 | 1.09 |
| 742 | 5.25 | Convergent Reinforcement Learning With Function Approximation: A Bilevel Optimization Perspective | 4, 6, 5, 6 | 0.83 |
| 743 | 5.00 | Quantization For Rapid Deployment Of Deep Neural Networks | 5, 5 | 0.00 |
| 744 | 5.00 | Characterizing The Accuracy/complexity Landscape Of Explanations Of Deep Networks Through Knowledge Extraction | 5 | 0.00 |
| 745 | 5.00 | Improved Robustness To Adversarial Examples Using Lipschitz Regularization Of The Loss | 6, 6, 3 | 1.41 |
| 746 | 5.00 | Implicit Autoencoders | 2, 7, 6 | 2.16 |
| 747 | 5.00 | Downsampling Leads To Image Memorization In Convolutional Autoencoders | 5 | 0.00 |
| 748 | 5.00 | The Gan Landscape: Losses, Architectures, Regularization, And Normalization | 4, 4, 7 | 1.41 |
| 749 | 5.00 | Canonical Correlation Analysis With Implicit Distributions | 5, 6, 4 | 0.82 |
| 750 | 5.00 | K-nearest Neighbors By Means Of Sequence To Sequence Deep Neural Networks And Memory Networks | 7, 4, 4 | 1.41 |
| 751 | 5.00 | Stcn: Stochastic Temporal Convolutional Networks | 6, 5, 4 | 0.82 |
| 752 | 5.00 | Actrce: Augmenting Experience Via Teacher’s Advice | 5, 7, 3 | 1.63 |
| 753 | 5.00 | Towards Gan Benchmarks Which Require Generalization | 6, 6, 3 | 1.41 |
| 754 | 5.00 | Metropolis-hastings View On Variational Inference And Adversarial Training | 5, 5, 5 | 0.00 |
| 755 | 5.00 | Learning Discriminators As Energy Networks In Adversarial Learning | 5, 5, 5 | 0.00 |
| 756 | 5.00 | Multi-agent Dual Learning | 5, 6, 4 | 0.82 |
| 757 | 5.00 | Optimistic Acceleration For Optimization | 6, 5, 4 | 0.82 |
| 758 | 5.00 | High Resolution And Fast Face Completion Via Progressively Attentive Gans | 5, 5, 5 | 0.00 |
| 759 | 5.00 | Ad-vat: An Asymmetric Dueling Mechanism For Learning Visual Active Tracking | 5, 4, 6 | 0.82 |
| 760 | 5.00 | Approximation Capability Of Neural Networks On Sets Of Probability Measures And Tree-structured Data | 6, 5, 4 | 0.82 |
| 761 | 5.00 | Likelihood-based Permutation Invariant Loss Function For Probability Distributions | 5, 6, 4 | 0.82 |
| 762 | 5.00 | An Efficient And Margin-approaching Zero-confidence Adversarial Attack | 5, 5 | 0.00 |
| 763 | 5.00 | S3ta: A Soft, Spatial, Sequential, Top-down Attention Model | 5, 5, 5 | 0.00 |
| 764 | 5.00 | Unsupervised Document Representation Using Partition Word-vectors Averaging | 6, 5, 4 | 0.82 |
| 765 | 5.00 | Systematic Generalization: What Is Required And Can It Be Learned? | 4, 5, 6 | 0.82 |
| 766 | 5.00 | On Learning Heteroscedastic Noise Models Within Differentiable Bayes Filters | 4, 4, 7 | 1.41 |
| 767 | 5.00 | Neural Program Repair By Jointly Learning To Localize And Repair | 5 | 0.00 |
| 768 | 5.00 | Learning To Remember: Dynamic Generative Memory For Continual Learning | 4, 3, 8 | 2.16 |
| 769 | 5.00 | On Meaning-preserving Adversarial Perturbations For Sequence-to-sequence Models | 4, 3, 8 | 2.16 |
| 770 | 5.00 | Variation Network: Learning High-level Attributes For Controlled Input Manipulation | 6, 4 | 1.00 |
| 771 | 5.00 | Unsupervised Learning Of The Set Of Local Maxima | 6, 7, 2 | 2.16 |
| 772 | 5.00 | Transfer Learning For Sequences Via Learning To Collocate | 6, 4, 5 | 0.82 |
| 773 | 5.00 | Excitation Dropout: Encouraging Plasticity In Deep Neural Networks | 5, 5, 5 | 0.00 |
| 774 | 5.00 | Nattack: A Strong And Universal Gaussian Black-box Adversarial Attack | 7, 4, 4 | 1.41 |
| 775 | 5.00 | Representation-constrained Autoencoders And An Application To Wireless Positioning | 4, 6 | 1.00 |
| 776 | 5.00 | Learning To Understand Goal Specifications By Modelling Reward | 5, 4, 6 | 0.82 |
| 777 | 5.00 | Link Prediction In Hypergraphs Using Graph Convolutional Networks | 6, 5, 4 | 0.82 |
| 778 | 5.00 | Is Wasserstein All You Need? | 4, 6, 5 | 0.82 |
| 779 | 5.00 | Generative Ensembles For Robust Anomaly Detection | 4, 4, 7 | 1.41 |
| 780 | 5.00 | The Importance Of Norm Regularization In Linear Graph Embedding: Theoretical Analysis And Empirical Demonstration | 7, 4, 4 | 1.41 |
| 781 | 5.00 | Nesterov's Method Is The Discretization Of A Differential Equation With Hessian Damping | 4, 5, 6 | 0.82 |
| 782 | 5.00 | The Effectiveness Of Pre-trained Code Embeddings | 6, 4, 5 | 0.82 |
| 783 | 5.00 | Understand The Dynamics Of Gans Via Primal-dual Optimization | 4, 5, 6 | 0.82 |
| 784 | 5.00 | Assessing Generalization In Deep Reinforcement Learning | 3, 5, 7 | 1.63 |
| 785 | 5.00 | Neural Persistence: A Complexity Measure For Deep Neural Networks Using Algebraic Topology | 6, 4, 5 | 0.82 |
| 786 | 5.00 | Transfer Learning For Estimating Causal Effects Using Neural Networks | 7, 5, 3 | 1.63 |
| 787 | 5.00 | Successor Options : An Option Discovery Algorithm For Reinforcement Learning | 6, 4 | 1.00 |
| 788 | 5.00 | Improving Gaussian Mixture Latent Variable Model Convergence With Optimal Transport | 5, 5, 5 | 0.00 |
| 789 | 5.00 | The Anisotropic Noise In Stochastic Gradient Descent: Its Behavior Of Escaping From Minima And Regularization Effects | 4, 6, 5 | 0.82 |
| 790 | 5.00 | Unicorn: Continual Learning With A Universal, Off-policy Agent | 4, 5, 6 | 0.82 |
| 791 | 5.00 | A Main/subsidiary Network Framework For Simplifying Binary Neural Networks | 5 | 0.00 |
| 792 | 5.00 | Adversarial Audio Super-resolution With Unsupervised Feature Losses | 4, 5, 6 | 0.82 |
| 793 | 5.00 | Point Cloud Gan | 4, 5, 6 | 0.82 |
| 794 | 5.00 | Unsupervised Meta-learning For Reinforcement Learning | 6, 4 | 1.00 |
| 795 | 5.00 | Initialized Equilibrium Propagation For Backprop-free Training | 3, 8, 4 | 2.16 |
| 796 | 5.00 | Stop Memorizing: A Data-dependent Regularization Framework For Intrinsic Pattern Learning | 7, 4, 4 | 1.41 |
| 797 | 5.00 | Geometry Aware Convolutional Filters For Omnidirectional Images Representation | 6, 4 | 1.00 |
| 798 | 5.00 | Cumulative Saliency Based Globally Balanced Filter Pruning For Efficient Convolutional Neural Networks | 6, 5, 4 | 0.82 |
| 799 | 5.00 | On-policy Trust Region Policy Optimisation With Replay Buffers | 4, 6, 5 | 0.82 |
| 800 | 5.00 | Teacher Guided Architecture Search | 6, 5, 4 | 0.82 |
| 801 | 5.00 | One-shot High-fidelity Imitation: Training Large-scale Deep Nets With Rl | 5 | 0.00 |
| 802 | 5.00 | Collaborative Multiagent Reinforcement Learning In Homogeneous Swarms | 6, 4, 5 | 0.82 |
| 803 | 5.00 | Engaging Image Captioning Via Personality | 5, 5, 5 | 0.00 |
| 804 | 5.00 | Strength In Numbers: Trading-off Robustness And Computation Via Adversarially-trained Ensembles | 5, 6, 4 | 0.82 |
| 805 | 5.00 | Reinforced Imitation Learning From Observations | 6, 5, 4 | 0.82 |
| 806 | 5.00 | Snapquant: A Probabilistic And Nested Parameterization For Binary Networks | 4, 6, 5 | 0.82 |
| 807 | 5.00 | Network Compression Using Correlation Analysis Of Layer Responses | 5, 6, 4 | 0.82 |
| 808 | 5.00 | Self-binarizing Networks | 5, 5, 5 | 0.00 |
| 809 | 5.00 | Characterizing Malicious Edges Targeting On Graph Neural Networks | 5, 5, 5 | 0.00 |
| 810 | 5.00 | Inferring Reward Functions From Demonstrators With Unknown Biases | 5, 5, 5 | 0.00 |
| 811 | 5.00 | Zero-shot Dual Machine Translation | 4, 6, 5 | 0.82 |
| 812 | 5.00 | Learning To Refer To 3d Objects With Natural Language | 5, 6, 4 | 0.82 |
| 813 | 5.00 | Multi-modal Generative Adversarial Networks For Diverse Datasets | 4, 6 | 1.00 |
| 814 | 5.00 | Pyramid Recurrent Neural Networks For Multi-scale Change-point Detection | 4, 6, 5 | 0.82 |
| 815 | 5.00 | On Regularization And Robustness Of Deep Neural Networks | 5, 4, 6 | 0.82 |
| 816 | 5.00 | Transferring Slu Models In Novel Domains | 6, 5, 4 | 0.82 |
| 817 | 5.00 | Directional Analysis Of Stochastic Gradient Descent Via Von Mises-fisher Distributions In Deep Learning | 6, 5, 4 | 0.82 |
| 818 | 5.00 | The Conditional Entropy Bottleneck | 2, 8, 5 | 2.45 |
| 819 | 5.00 | Learning To Simulate | 4, 5, 6 | 0.82 |
| 820 | 5.00 | Robustness Certification With Refinement | 5, 6, 4 | 0.82 |
| 821 | 5.00 | A Privacy-preserving Image Classification Framework With A Learnable Obfuscator | 5, 5 | 0.00 |
| 822 | 5.00 | Noisy Information Bottlenecks For Generalization | 7, 5, 3 | 1.63 |
| 823 | 5.00 | Convolutional Neural Networks Combined With Runge-kutta Methods | 4, 5, 6 | 0.82 |
| 824 | 5.00 | Generative Adversarial Self-imitation Learning | 5, 6, 4 | 0.82 |
| 825 | 5.00 | Capsules Graph Neural Network | 6, 4, 5 | 0.82 |
| 826 | 5.00 | Human-guided Column Networks: Augmenting Deep Learning With Advice | 6, 4, 5 | 0.82 |
| 827 | 5.00 | Rethinking Learning Rate Schedules For Stochastic Optimization | 4, 6 | 1.00 |
| 828 | 5.00 | Therml: The Thermodynamics Of Machine Learning | 7, 3, 5 | 1.63 |
| 829 | 5.00 | Geneval: A Benchmark Suite For Evaluating Generative Models | 4, 5, 6 | 0.82 |
| 830 | 5.00 | Segen: Sample-ensemble Genetic Evolutionary Network Model | 5, 5, 5 | 0.00 |
| 831 | 5.00 | Incremental Few-shot Learning With Attention Attractor Networks | 5, 5, 5 | 0.00 |
| 832 | 5.00 | Interactive Parallel Exploration For Reinforcement Learning In Continuous Action Spaces | 5, 4, 6 | 0.82 |
| 833 | 5.00 | On The Learning Dynamics Of Deep Neural Networks | 5, 5, 5 | 0.00 |
| 834 | 5.00 | Redsync : Reducing Synchronization Traffic For Distributed Deep Learning | 5, 5, 5 | 0.00 |
| 835 | 5.00 | Spread Divergences | 5, 4, 6 | 0.82 |
| 836 | 5.00 | Local Image-to-image Translation Via Pixel-wise Highway Adaptive Instance Normalization | 6, 4, 5 | 0.82 |
| 837 | 5.00 | Causal Importance Of Orientation Selectivity For Generalization In Image Recognition | 6, 4 | 1.00 |
| 838 | 5.00 | Tts-gan: A Generative Adversarial Network For Style Modeling In A Text-to-speech System | 5, 4, 6, 5 | 0.71 |
| 839 | 5.00 | Connecting The Dots Between Mle And Rl For Sequence Generation | 4, 6, 5 | 0.82 |
| 840 | 5.00 | Denoise While Aggregating: Collaborative Learning In Open-domain Question Answering | 4, 6, 5 | 0.82 |
| 841 | 5.00 | Learning Neuron Non-linearities With Kernel-based Deep Neural Networks | 5, 4, 6 | 0.82 |
| 842 | 5.00 | A Recurrent Neural Cascade-based Model For Continuous-time Diffusion Process | 7, 4, 4 | 1.41 |
| 843 | 5.00 | Learning And Planning With A Semantic Model | 4, 7, 4 | 1.41 |
| 844 | 5.00 | On The Effectiveness Of Task Granularity For Transfer Learning | 5, 5, 5 | 0.00 |
| 845 | 5.00 | Towards Language Agnostic Universal Representations | 5, 4, 6 | 0.82 |
| 846 | 5.00 | Guided Exploration In Deep Reinforcement Learning | 7, 5, 3 | 1.63 |
| 847 | 5.00 | Variational Autoencoders With Jointly Optimized Latent Dependency Structure | 6, 4, 5 | 0.82 |
| 848 | 5.00 | Metric-optimized Example Weights | 4, 4, 7 | 1.41 |
| 849 | 5.00 | Bias Also Matters: Bias Attribution For Deep Neural Network Explanation | 5, 5 | 0.00 |
| 850 | 5.00 | Discrete Flow Posteriors For Variational Inference In Discrete Dynamical Systems | 4, 4, 7 | 1.41 |
| 851 | 5.00 | Revisiting Reweighted Wake-sleep | 5, 5, 5 | 0.00 |
| 852 | 5.00 | Physiological Signal Embeddings (phase) Via Interpretable Stacked Models | 6, 5, 4 | 0.82 |
| 853 | 5.00 | Evading Defenses To Transferable Adversarial Examples By Mitigating Attention Shift | 4, 4, 7 | 1.41 |
| 854 | 5.00 | Transferrable End-to-end Learning For Protein Interface Prediction | 5, 5, 5 | 0.00 |
| 855 | 5.00 | From Language To Goals: Inverse Reinforcement Learning For Vision-based Instruction Following | 5, 5, 5 | 0.00 |
| 856 | 5.00 | Importance Resampling For Off-policy Policy Evaluation | 5, 5 | 0.00 |
| 857 | 5.00 | Phrase-based Attentions | 5, 5, 5 | 0.00 |
| 858 | 5.00 | Riemannian Transe: Multi-relational Graph Embedding In Non-euclidean Space | 5, 5, 5 | 0.00 |
| 859 | 5.00 | Graph2graph Networks For Multi-label Classification | 4, 6, 5 | 0.82 |
| 860 | 5.00 | Dynamic Graph Representation Learning Via Self-attention Networks | 4, 6, 5 | 0.82 |
| 861 | 5.00 | Pointgrow: Autoregressively Learned Point Cloud Generation With Self-attention | 3, 6, 6 | 1.41 |
| 862 | 5.00 | Learning Representations Of Categorical Feature Combinations Via Self-attention | 5, 5, 5 | 0.00 |
| 863 | 5.00 | Graph2seq: Scalable Learning Dynamics For Graphs | 6, 5, 4 | 0.82 |
| 864 | 5.00 | Selective Convolutional Units: Improving Cnns Via Channel Selectivity | 5, 5 | 0.00 |
| 865 | 5.00 | Probabilistic Semantic Embedding | 7, 4, 4 | 1.41 |
| 866 | 5.00 | Double Neural Counterfactual Regret Minimization | 5, 6, 4 | 0.82 |
| 867 | 5.00 | Prototypical Examples In Deep Learning: Metrics, Characteristics, And Utility | 5 | 0.00 |
| 868 | 5.00 | Experience Replay For Continual Learning | 5, 5, 5 | 0.00 |
| 869 | 5.00 | Learning With Random Learning Rates. | 5, 5, 5 | 0.00 |
| 870 | 5.00 | End-to-end Learning Of A Convolutional Neural Network Via Deep Tensor Decomposition | 5, 5 | 0.00 |
| 871 | 5.00 | Mlprune: Multi-layer Pruning For Automated Neural Network Compression | 5, 6, 4 | 0.82 |
| 872 | 5.00 | Graph Transformation Policy Network For Chemical Reaction Prediction | 5, 5, 5 | 0.00 |
| 873 | 5.00 | Adversarial Reprogramming Of Neural Networks | 4, 4, 7 | 1.41 |
| 874 | 5.00 | Exploiting Cross-lingual Subword Similarities In Low-resource Document Classification | 4, 6, 5 | 0.82 |
| 875 | 5.00 | Using Ontologies To Improve Performance In Massively Multi-label Prediction | 6, 5, 4 | 0.82 |
| 876 | 5.00 | Vhegan: Variational Hetero-encoder Randomized Gan For Zero-short Learning | 5, 5, 5 | 0.00 |
| 877 | 5.00 | Analyzing Federated Learning Through An Adversarial Lens | 5, 4, 6 | 0.82 |
| 878 | 5.00 | Information Regularized Neural Networks | 5, 6, 4 | 0.82 |
| 879 | 5.00 | Ace: Artificial Checkerboard Enhancer To Induce And Evade Adversarial Attacks | 4, 6 | 1.00 |
| 880 | 5.00 | Cutting Down Training Memory By Re-fowarding | 4, 6 | 1.00 |
| 881 | 5.00 | Accelerated Value Iteration Via Anderson Mixing | 7, 4, 4 | 1.41 |
| 882 | 5.00 | Cohen Welling Bases & So(2)-equivariant Classifiers Using Tensor Nonlinearity. | 3, 7, 5 | 1.63 |
| 883 | 5.00 | Cautious Deep Learning | 4, 7, 4 | 1.41 |
| 884 | 5.00 | Distributionally Robust Optimization Leads To Better Generalization: On Sgd And Beyond | 5 | 0.00 |
| 885 | 5.00 | Variational Smoothing In Recurrent Neural Network Language Models | 7, 6, 2 | 2.16 |
| 886 | 5.00 | Morph-net: An Universal Function Approximator | 5, 5 | 0.00 |
| 887 | 5.00 | Excessive Invariance Causes Adversarial Vulnerability | 6, 4, 5 | 0.82 |
| 888 | 5.00 | Zero-shot Learning For Speech Recognition With Universal Phonetic Model | 7, 4, 4 | 1.41 |
| 889 | 5.00 | On The Relationship Between Neural Machine Translation And Word Alignment | 4, 5, 6 | 0.82 |
| 890 | 5.00 | Choicenet: Robust Learning By Revealing Output Correlations | 4, 6, 5 | 0.82 |
| 891 | 5.00 | Multi-objective Training Of Generative Adversarial Networks With Multiple Discriminators | 6, 5, 4 | 0.82 |
| 892 | 5.00 | Massively Parallel Hyperparameter Tuning | 6, 4, 5 | 0.82 |
| 893 | 5.00 | Label Propagation Networks | 5, 4, 6 | 0.82 |
| 894 | 5.00 | A Variational Autoencoder For Probabilistic Non-negative Matrix Factorisation | 4, 4, 7 | 1.41 |
| 895 | 5.00 | Volumetric Convolution: Automatic Representation Learning In Unit Ball | 5, 5, 5 | 0.00 |
| 896 | 5.00 | Data Interpretation And Reasoning Over Scientific Plots | 6, 6, 3 | 1.41 |
| 897 | 5.00 | Weakly-supervised Knowledge Graph Alignment With Adversarial Learning | 5, 5, 5 | 0.00 |
| 898 | 5.00 | Multi-agent Deep Reinforcement Learning With Extremely Noisy Observations | 7, 3, 5 | 1.63 |
| 899 | 5.00 | Guided Evolutionary Strategies: Escaping The Curse Of Dimensionality In Random Search | 5, 4, 6 | 0.82 |
| 900 | 5.00 | Improved Language Modeling By Decoding The Past | 6, 7, 2 | 2.16 |
| 901 | 5.00 | A Comprehensive, Application-oriented Study Of Catastrophic Forgetting In Dnns | 5, 5, 5 | 0.00 |
| 902 | 5.00 | On Accurate Evaluation Of Gans For Language Generation | 5, 3, 7 | 1.63 |
| 903 | 5.00 | Adversarial Sampling For Active Learning | 5, 5 | 0.00 |
| 904 | 5.00 | A Case For Object Compositionality In Deep Generative Models Of Images | 5, 4, 6 | 0.82 |
| 905 | 5.00 | Inducing Cooperation Via Learning To Reshape Rewards In Semi-cooperative Multi-agent Reinforcement Learning | 5, 5, 5 | 0.00 |
| 906 | 5.00 | Déjà Vu: An Empirical Evaluation Of The Memorization Properties Of Convnets | 4, 5, 6 | 0.82 |
| 907 | 5.00 | Deep Clustering Based On A Mixture Of Autoencoders | 6, 4, 5 | 0.82 |
| 908 | 5.00 | Graph2seq: Graph To Sequence Learning With Attention-based Neural Networks | 6, 6, 3 | 1.41 |
| 909 | 5.00 | Simple Black-box Adversarial Attacks | 6, 5, 4 | 0.82 |
| 910 | 5.00 | The Expressive Power Of Deep Neural Networks With Circulant Matrices | 7, 3 | 2.00 |
| 911 | 5.00 | A Frank-wolfe Framework For Efficient And Effective Adversarial Attacks | 5, 4, 6 | 0.82 |
| 912 | 5.00 | Arm: Augment-reinforce-merge Gradient For Stochastic Binary Networks | 3, 7 | 2.00 |
| 913 | 5.00 | Invase: Instance-wise Variable Selection Using Neural Networks | 4, 5, 6 | 0.82 |
| 914 | 5.00 | Quality Evaluation Of Gans Using Cross Local Intrinsic Dimensionality | 4, 6, 5 | 0.82 |
| 915 | 5.00 | Measuring And Regularizing Networks In Function Space | 5, 5, 5 | 0.00 |
| 916 | 5.00 | Unsupervised Multi-target Domain Adaptation: An Information Theoretic Approach | 6, 4, 5 | 0.82 |
| 917 | 5.00 | Learnable Embedding Space For Efficient Neural Architecture Compression | 5, 5, 5 | 0.00 |
| 918 | 5.00 | Learning Diverse Generations Using Determinantal Point Processes | 5, 5 | 0.00 |
| 919 | 5.00 | Towards Resisting Large Data Variations Via Introspective Learning | 4, 5, 6 | 0.82 |
| 920 | 5.00 | Collapse Of Deep And Narrow Neural Nets | 6, 4, 5 | 0.82 |
| 921 | 5.00 | Dataset Distillation | 5, 5, 5 | 0.00 |
| 922 | 5.00 | Random Mask: Towards Robust Convolutional Neural Networks | 6, 7, 2 | 2.16 |
| 923 | 5.00 | Making Convolutional Networks Shift-invariant Again | 5, 5, 5 | 0.00 |
| 924 | 5.00 | Intrinsic Social Motivation Via Causal Influence In Multi-agent Rl | 4, 6 | 1.00 |
| 925 | 5.00 | Poincare Glove: Hyperbolic Word Embeddings | 6, 4, 5 | 0.82 |
| 926 | 5.00 | Information Maximization Auto-encoding | 5, 6, 4 | 0.82 |
| 927 | 5.00 | Deep Reinforcement Learning Of Universal Policies With Diverse Environment Summaries | 4, 6, 5 | 0.82 |
| 928 | 5.00 | What Is In A Translation Unit? Comparing Character And Subword Representations Beyond Translation | 5, 5, 5 | 0.00 |
| 929 | 5.00 | Multi-turn Dialogue Response Generation In An Adversarial Learning Framework | 4, 6, 5 | 0.82 |
| 930 | 5.00 | Live Face De-identification In Video | 4, 6 | 1.00 |
| 931 | 5.00 | Capacity Of Deep Neural Networks Under Parameter Quantization | 5, 5, 5 | 0.00 |
| 932 | 5.00 | Learning To Plan | 4, 6, 5 | 0.82 |
| 933 | 5.00 | Ada-boundary: Accelerating The Dnn Training Via Adaptive Boundary Batch Selection | 5, 5, 5 | 0.00 |
| 934 | 5.00 | Empirical Observations On The Instability Of Aligning Word Vector Spaces With Gans | 4, 6, 5 | 0.82 |
| 935 | 5.00 | Coco-gan: Conditional Coordinate Generative Adversarial Network | 5, 6, 4 | 0.82 |
| 936 | 5.00 | Shrinkage-based Bias-variance Trade-off For Deep Reinforcement Learning | 4, 4, 7 | 1.41 |
| 937 | 5.00 | The Nonlinearity Coefficient - Predicting Generalization In Deep Neural Networks | 5, 7, 3 | 1.63 |
| 938 | 5.00 | Dissecting An Adversarial Framework For Information Retrieval | 6, 5, 4 | 0.82 |
| 939 | 5.00 | Favae: Sequence Disentanglement Using In- Formation Bottleneck Principle | 5, 6, 4 | 0.82 |
| 940 | 5.00 | An Automatic Operation Batching Strategy For The Backward Propagation Of Neural Networks Having Dynamic Computation Graphs | 5, 6, 4 | 0.82 |
| 941 | 5.00 | Adversarial Information Factorization | 3, 6, 6 | 1.41 |
| 942 | 5.00 | Learning To Progressively Plan | 5, 5 | 0.00 |
| 943 | 5.00 | Learning From The Experience Of Others: Approximate Empirical Bayes In Neural Networks | 3, 7 | 2.00 |
| 944 | 5.00 | Structured Content Preservation For Unsupervised Text Style Transfer | 6, 4 | 1.00 |
| 945 | 5.00 | Optimal Margin Distribution Network | 5, 6, 4 | 0.82 |
| 946 | 5.00 | Spatial-winograd Pruning Enabling Sparse Winograd Convolution | 5, 4, 6 | 0.82 |
| 947 | 5.00 | Bayesian Deep Learning Via Stochastic Gradient Mcmc With A Stochastic Approximation Adaptation | 5, 4, 6 | 0.82 |
| 948 | 4.75 | Geomstats: A Python Package For Riemannian Geometry In Machine Learning | 4, 4, 3, 8 | 1.92 |
| 949 | 4.67 | Boosting Trust Region Policy Optimization By Normalizing Flows Policy | 6, 4, 4 | 0.94 |
| 950 | 4.67 | Inference Of Unobserved Event Streams With Neural Hawkes Particle Smoothing | 5, 4, 5 | 0.47 |
| 951 | 4.67 | Selective Self-training For Semi-supervised Learning | 4, 6, 4 | 0.94 |
| 952 | 4.67 | Visualizing And Discovering Behavioural Weaknesses In Deep Reinforcement Learning | 5, 5, 4 | 0.47 |
| 953 | 4.67 | Differentiable Expected Bleu For Text Generation | 4, 4, 6 | 0.94 |
| 954 | 4.67 | Stochastic Learning Of Additive Second-order Penalties With Applications To Fairness | 5, 5, 4 | 0.47 |
| 955 | 4.67 | Learning To Attend On Essential Terms: An Enhanced Retriever-reader Model For Open-domain Question Answering | 4, 5, 5 | 0.47 |
| 956 | 4.67 | Computation-efficient Quantization Method For Deep Neural Networks | 4, 5, 5 | 0.47 |
| 957 | 4.67 | Parameter Efficient Training Of Deep Convolutional Neural Networks By Dynamic Sparse Reparameterization | 4, 4, 6 | 0.94 |
| 958 | 4.67 | Improving Latent Variable Descriptiveness By Modelling Rather Than Ad-hoc Factors | 4, 4, 6 | 0.94 |
| 959 | 4.67 | Diagnosing Language Inconsistency In Cross-lingual Word Embeddings | 6, 4, 4 | 0.94 |
| 960 | 4.67 | Learning Joint Wasserstein Auto-encoders For Joint Distribution Matching | 5, 5, 4 | 0.47 |
| 961 | 4.67 | Pruning In Training: Learning And Ranking Sparse Connections In Deep Convolutional Networks | 5, 5, 4 | 0.47 |
| 962 | 4.67 | Learning Graph Representations By Dendrograms | 4, 5, 5 | 0.47 |
| 963 | 4.67 | Maximum A Posteriori On A Submanifold: A General Image Restoration Method With Gan | 4, 4, 6 | 0.94 |
| 964 | 4.67 | Dual Skew Divergence Loss For Neural Machine Translation | 3, 6, 5 | 1.25 |
| 965 | 4.67 | Pix2scene: Learning Implicit 3d Representations From Images | 5, 6, 3 | 1.25 |
| 966 | 4.67 | Few-shot Learning By Exploiting Object Relation | 6, 4, 4 | 0.94 |
| 967 | 4.67 | Model Compression With Generative Adversarial Networks | 4, 5, 5 | 0.47 |
| 968 | 4.67 | Explicit Recall For Efficient Exploration | 7, 4, 3 | 1.70 |
| 969 | 4.67 | On The Geometry Of Adversarial Examples | 5, 3, 6 | 1.25 |
| 970 | 4.67 | The Expressive Power Of Gated Recurrent Units As A Continuous Dynamical System | 5, 4, 5 | 0.47 |
| 971 | 4.67 | Siamese Capsule Networks | 5, 6, 3 | 1.25 |
| 972 | 4.67 | A Theoretical Framework For Deep Locally Connected Relu Network | 3, 7, 4 | 1.70 |
| 973 | 4.67 | Unsupervised Graph Embedding Using Dynamic Routing Between Capsules | 4, 5, 5 | 0.47 |
| 974 | 4.67 | Intriguing Properties Of Learned Representations | 3, 6, 5 | 1.25 |
| 975 | 4.67 | Unsupervised Emergence Of Spatial Structure From Sensorimotor Prediction | 4, 4, 6 | 0.94 |
| 976 | 4.67 | 3d-relnet: Joint Object And Relational Network For 3d Prediction | 6, 3, 5 | 1.25 |
| 977 | 4.67 | Area Attention | 5, 4, 5 | 0.47 |
| 978 | 4.67 | Manifold Alignment Via Feature Correspondence | 5, 5, 4 | 0.47 |
| 979 | 4.67 | Learning To Drive By Observing The Best And Synthesizing The Worst | 3, 6, 5 | 1.25 |
| 980 | 4.67 | Analysis Of Memory Organization For Dynamic Neural Networks | 8, 4, 2 | 2.49 |
| 981 | 4.67 | Learning Gibbs-regularized Gans With Variational Discriminator Reparameterization | 5, 5, 4 | 0.47 |
| 982 | 4.67 | Online Bellman Residue Minimization Via Saddle Point Optimization | 5, 5, 4 | 0.47 |
| 983 | 4.67 | Variational Sparse Coding | 4, 5, 5 | 0.47 |
| 984 | 4.67 | A Study Of Robustness Of Neural Nets Using Approximate Feature Collisions | 6, 4, 4 | 0.94 |
| 985 | 4.67 | Learning With Little Data: Evaluation Of Deep Learning Algorithms | 6, 4, 4 | 0.94 |
| 986 | 4.67 | Crystalgan: Learning To Discover Crystallographic Structures With Generative Adversarial Networks | 3, 7, 4 | 1.70 |
| 987 | 4.67 | Discriminative Out-of-distribution Detection For Semantic Segmentation | 4, 7, 3 | 1.70 |
| 988 | 4.67 | Characterizing Vulnerabilities Of Deep Reinforcement Learning | 5, 4, 5 | 0.47 |
| 989 | 4.67 | Traditional And Heavy Tailed Self Regularization In Neural Network Models | 4, 4, 6 | 0.94 |
| 990 | 4.67 | Cgnf: Conditional Graph Neural Fields | 5, 4, 5 | 0.47 |
| 991 | 4.67 | Novel Positional Encodings To Enable Tree-structured Transformers | 5, 3, 6 | 1.25 |
| 992 | 4.67 | Feature Prioritization And Regularization Improve Standard Accuracy And Adversarial Robustness | 4, 4, 6 | 0.94 |
| 993 | 4.67 | Neural Variational Inference For Embedding Knowledge Graphs | 5, 5, 4 | 0.47 |
| 994 | 4.67 | Neural Separation Of Observed And Unobserved Distributions | 5, 4, 5 | 0.47 |
| 995 | 4.67 | Neural Malware Control With Deep Reinforcement Learning | 5, 4, 5 | 0.47 |
| 996 | 4.67 | Exploiting Environmental Variation To Improve Policy Robustness In Reinforcement Learning | 5, 3, 6 | 1.25 |
| 997 | 4.67 | Learning Programmatically Structured Representations With Perceptor Gradients | 6, 6, 2 | 1.89 |
| 998 | 4.67 | Logically-constrained Neural Fitted Q-iteration | 5, 4, 5 | 0.47 |
| 999 | 4.67 | Noise-tempered Generative Adversarial Networks | 4, 5, 5 | 0.47 |
| 1000 | 4.67 | Low-rank Matrix Factorization Of Lstm As Effective Model Compression | 5, 5, 4 | 0.47 |
| 1001 | 4.67 | Safe Policy Learning From Observations | 6, 5, 3 | 1.25 |
| 1002 | 4.67 | Finding Mixed Nash Equilibria Of Generative Adversarial Networks | 4, 5, 5 | 0.47 |
| 1003 | 4.67 | Tfgan: Improving Conditioning For Text-to-video Synthesis | 6, 3, 5 | 1.25 |
| 1004 | 4.67 | An Efficient Network For Predicting Time-varying Distributions | 5, 4, 5 | 0.47 |
| 1005 | 4.67 | Investigating Cnns' Learning Representation Under Label Noise | 5, 4, 5 | 0.47 |
| 1006 | 4.67 | Backplay: 'man Muss Immer Umkehren' | 5, 5, 4 | 0.47 |
| 1007 | 4.67 | Expanding The Reach Of Federated Learning By Reducing Client Resource Requirements | 4, 5, 5 | 0.47 |
| 1008 | 4.67 | Sliced Wasserstein Auto-encoders | 4, 4, 6 | 0.94 |
| 1009 | 4.67 | Conditional Network Embeddings | 4, 5, 5 | 0.47 |
| 1010 | 4.67 | Gradient-based Learning For F-measure And Other Performance Metrics | 5, 4, 5 | 0.47 |
| 1011 | 4.67 | Neural Networks With Structural Resistance To Adversarial Attacks | 5, 5, 4 | 0.47 |
| 1012 | 4.67 | Stability Of Stochastic Gradient Method With Momentum For Strongly Convex Loss Functions | 4, 6, 4 | 0.94 |
| 1013 | 4.67 | Context-aware Forecasting For Multivariate Stationary Time-series | 5, 5, 4 | 0.47 |
| 1014 | 4.67 | Ssoc: Learning Spontaneous And Self-organizing Communication For Multi-agent Collaboration | 4, 5, 5 | 0.47 |
| 1015 | 4.67 | Unifying Bilateral Filtering And Adversarial Training For Robust Neural Networks | 4, 5, 5 | 0.47 |
| 1016 | 4.67 | When Will Gradient Methods Converge To Max-margin Classifier Under Relu Models? | 5, 4, 5 | 0.47 |
| 1017 | 4.67 | Skim-pixelcnn | 4, 4, 6 | 0.94 |
| 1018 | 4.67 | Nsga-net: A Multi-objective Genetic Algorithm For Neural Architecture Search | 6, 5, 3 | 1.25 |
| 1019 | 4.67 | Sentence Encoding With Tree-constrained Relation Networks | 3, 5, 6 | 1.25 |
| 1020 | 4.67 | Text Infilling | 3, 5, 6 | 1.25 |
| 1021 | 4.67 | Accelerated Sparse Recovery Under Structured Measurements | 4, 5, 5 | 0.47 |
| 1022 | 4.67 | Tequilagan: How To Easily Identify Gan Samples | 4, 6, 4 | 0.94 |
| 1023 | 4.67 | Differential Equation Networks | 5, 4, 5 | 0.47 |
| 1024 | 4.67 | Predicting The Present And Future States Of Multi-agent Systems From Partially-observed Visual Data | 5, 4, 5 | 0.47 |
| 1025 | 4.67 | What Information Does A Resnet Compress? | 4, 4, 6 | 0.94 |
| 1026 | 4.67 | Security Analysis Of Deep Neural Networks Operating In The Presence Of Cache Side-channel Attacks | 4, 6, 4 | 0.94 |
| 1027 | 4.67 | A Proposed Hierarchy Of Deep Learning Tasks | 6, 4, 4 | 0.94 |
| 1028 | 4.67 | Probabilistic Binary Neural Networks | 6, 5, 3 | 1.25 |
| 1029 | 4.67 | Unsupervised Word Discovery With Segmental Neural Language Models | 4, 3, 7 | 1.70 |
| 1030 | 4.67 | Double Viterbi: Weight Encoding For High Compression Ratio And Fast On-chip Reconstruction For Deep Neural Network | 3, 5, 6 | 1.25 |
| 1031 | 4.67 | Consistency-based Anomaly Detection With Adaptive Multiple-hypotheses Predictions | 4, 5, 5 | 0.47 |
| 1032 | 4.67 | Efficient Dictionary Learning With Gradient Descent | 5, 4, 5 | 0.47 |
| 1033 | 4.67 | Meta-learning With Differentiable Closed-form Solvers | 5, 2, 7 | 2.05 |
| 1034 | 4.67 | Learning Shared Manifold Representation Of Images And Attributes For Generalized Zero-shot Learning | 4, 5, 5 | 0.47 |
| 1035 | 4.67 | Rectified Gradient: Layer-wise Thresholding For Sharp And Coherent Attribution Maps | 5, 5, 4 | 0.47 |
| 1036 | 4.67 | Holographic And Other Point Set Distances For Machine Learning | 4, 3, 7 | 1.70 |
| 1037 | 4.67 | Generative Replay With Feedback Connections As A General Strategy For Continual Learning | 4, 4, 6 | 0.94 |
| 1038 | 4.67 | Multi-grained Entity Proposal Network For Named Entity Recognition | 5, 5, 4 | 0.47 |
| 1039 | 4.67 | Unsupervised Disentangling Structure And Appearance | 6, 5, 3 | 1.25 |
| 1040 | 4.67 | Mode Normalization | 5, 6, 3 | 1.25 |
| 1041 | 4.67 | Measuring Density And Similarity Of Task Relevant Information In Neural Representations | 4, 5, 5 | 0.47 |
| 1042 | 4.67 | Chemical Names Standardization Using Neural Sequence To Sequence Model | 4, 3, 7 | 1.70 |
| 1043 | 4.67 | Predictive Uncertainty Through Quantization | 5, 4, 5 | 0.47 |
| 1044 | 4.67 | Learning Information Propagation In The Dynamical Systems Via Information Bottleneck Hierarchy | 5, 4, 5 | 0.47 |
| 1045 | 4.67 | Like What You Like: Knowledge Distill Via Neuron Selectivity Transfer | 4, 4, 6 | 0.94 |
| 1046 | 4.67 | Pumpout: A Meta Approach For Robustly Training Deep Neural Networks With Noisy Labels | 6, 5, 3 | 1.25 |
| 1047 | 4.67 | Integral Pruning On Activations And Weights For Efficient Neural Networks | 4, 5, 5 | 0.47 |
| 1048 | 4.67 | Object-oriented Model Learning Through Multi-level Abstraction | 4, 4, 6 | 0.94 |
| 1049 | 4.67 | Zero-training Sentence Embedding Via Orthogonal Basis | 5, 4, 5 | 0.47 |
| 1050 | 4.67 | Effective Path: Know The Unknowns Of Neural Network | 4, 4, 6 | 0.94 |
| 1051 | 4.67 | Marginalized Average Attentional Network For Weakly-supervised Learning | 5, 6, 3 | 1.25 |
| 1052 | 4.67 | An Investigation Of Model-free Planning | 5, 5, 4 | 0.47 |
| 1053 | 4.67 | Learning To Control Self-assembling Morphologies: A Study Of Generalization Via Modularity | 4, 7, 3 | 1.70 |
| 1054 | 4.67 | Success At Any Cost: Value Constrained Model-free Continuous Control | 5, 5, 4 | 0.47 |
| 1055 | 4.67 | $a^*$ Sampling With Probability Matching | 5, 6, 3 | 1.25 |
| 1056 | 4.67 | Outlier Detection From Image Data | 4, 5, 5 | 0.47 |
| 1057 | 4.67 | Penetrating The Fog: The Path To Efficient Cnn Models | 5, 4, 5 | 0.47 |
| 1058 | 4.67 | Conscious Inference For Object Detection | 4, 6, 4 | 0.94 |
| 1059 | 4.67 | Reduced-gate Convolutional Lstm Design Using Predictive Coding For Next-frame Video Prediction | 5, 2, 7 | 2.05 |
| 1060 | 4.67 | Simile: Introducing Sequential Information Towards More Effective Imitation Learning | 6, 4, 4 | 0.94 |
| 1061 | 4.67 | Approximation And Non-parametric Estimation Of Resnet-type Convolutional Neural Networks Via Block-sparse Fully-connected Neural Networks | 4, 6, 4 | 0.94 |
| 1062 | 4.67 | Unsupervised Image To Sequence Translation With Canvas-drawer Networks | 4, 6, 4 | 0.94 |
| 1063 | 4.67 | Domain Adaptive Transfer Learning | 3, 4, 7 | 1.70 |
| 1064 | 4.67 | An Energy-based Framework For Arbitrary Label Noise Correction | 5, 4, 5 | 0.47 |
| 1065 | 4.67 | Expressiveness In Deep Reinforcement Learning | 6, 4, 4 | 0.94 |
| 1066 | 4.67 | Generalized Adaptive Moment Estimation | 3, 4, 7 | 1.70 |
| 1067 | 4.67 | Lit: Block-wise Intermediate Representation Training For Model Compression | 4, 5, 5 | 0.47 |
| 1068 | 4.67 | Ergodic Measure Preserving Flows | 5, 5, 4 | 0.47 |
| 1069 | 4.67 | End-to-end Learning Of Pharmacological Assays From High-resolution Microscopy Images | 6, 3, 5 | 1.25 |
| 1070 | 4.67 | Theoretical And Empirical Study Of Adversarial Examples | 5, 5, 4 | 0.47 |
| 1071 | 4.67 | Sparse Binary Compression: Towards Distributed Deep Learning With Minimal Communication | 6, 3, 5 | 1.25 |
| 1072 | 4.67 | Generating Realistic Stock Market Order Streams | 5, 5, 4 | 0.47 |
| 1073 | 4.67 | A Unified View Of Deep Metric Learning Via Gradient Analysis | 3, 6, 5 | 1.25 |
| 1074 | 4.67 | Self-supervised Generalisation With Meta Auxiliary Learning | 4, 4, 6 | 0.94 |
| 1075 | 4.67 | Improved Resistance Of Neural Networks To Adversarial Images Through Generative Pre-training | 4, 4, 6 | 0.94 |
| 1076 | 4.67 | Count-based Exploration With The Successor Representation | 5, 5, 4 | 0.47 |
| 1077 | 4.67 | Tabnn: A Universal Neural Network Solution For Tabular Data | 5, 4, 5 | 0.47 |
| 1078 | 4.67 | Evolving Intrinsic Motivations For Altruistic Behavior | 5, 6, 3 | 1.25 |
| 1079 | 4.67 | Integrated Steganography And Steganalysis With Generative Adversarial Networks | 3, 6, 5 | 1.25 |
| 1080 | 4.67 | Coupled Recurrent Models For Polyphonic Music Composition | 7, 3, 4 | 1.70 |
| 1081 | 4.67 | Deep Curiosity Search: Intra-life Exploration Can Improve Performance On Challenging Deep Reinforcement Learning Problems | 5, 5, 4 | 0.47 |
| 1082 | 4.67 | Supportnet: Solving Catastrophic Forgetting In Class Incremental Learning With Support Data | 6, 4, 4 | 0.94 |
| 1083 | 4.67 | Using Gans For Generation Of Realistic City-scale Ride Sharing/hailing Data Sets | 4, 5, 5 | 0.47 |
| 1084 | 4.67 | Entropic Gans Meet Vaes: A Statistical Approach To Compute Sample Likelihoods In Gans | 5, 5, 4 | 0.47 |
| 1085 | 4.67 | Deep-trim: Revisiting L1 Regularization For Connection Pruning Of Deep Network | 4, 6, 4 | 0.94 |
| 1086 | 4.67 | Highly Efficient 8-bit Low Precision Inference Of Convolutional Neural Networks | 6, 4, 4 | 0.94 |
| 1087 | 4.67 | Towards A Better Understanding Of Vector Quantized Autoencoders | 7, 3, 4 | 1.70 |
| 1088 | 4.67 | What A Difference A Pixel Makes: An Empirical Examination Of Features Used By Cnns For Categorisation | 4, 3, 7 | 1.70 |
| 1089 | 4.67 | Unsupervised Expectation Learning For Multisensory Binding | 4, 5, 5 | 0.47 |
| 1090 | 4.67 | Pairwise Augmented Gans With Adversarial Reconstruction Loss | 4, 5, 5 | 0.47 |
| 1091 | 4.67 | Gradient Descent Happens In A Tiny Subspace | 4, 4, 6 | 0.94 |
| 1092 | 4.67 | Answer-based Adversarial Training For Generating Clarification Questions | 4, 6, 4 | 0.94 |
| 1093 | 4.67 | Convergence Guarantees For Rmsprop And Adam In Non-convex Optimization And An Empirical Comparison To Nesterov Acceleration | 5, 4, 5 | 0.47 |
| 1094 | 4.67 | What Would Pi* Do?: Imitation Learning Via Off-policy Reinforcement Learning | 5, 4, 5 | 0.47 |
| 1095 | 4.67 | Strokenet: A Neural Painting Environment | 4, 6, 4 | 0.94 |
| 1096 | 4.67 | Learned Optimizers That Outperform On Wall-clock And Validation Loss | 4, 5, 5 | 0.47 |
| 1097 | 4.67 | Estimating Heterogeneous Treatment Effects Using Neural Networks With The Y-learner | 5, 5, 4 | 0.47 |
| 1098 | 4.67 | Batch Normalization Sampling | 4, 5, 5 | 0.47 |
| 1099 | 4.67 | Selectivity Metrics Can Overestimate The Selectivity Of Units: A Case Study On Alexnet | 5, 6, 3 | 1.25 |
| 1100 | 4.50 | Transfer And Exploration Via The Information Bottleneck | 6, 3 | 1.50 |
| 1101 | 4.50 | Improving On-policy Learning With Statistical Reward Accumulation | 4, 5 | 0.50 |
| 1102 | 4.50 | Salsa-text : Self Attentive Latent Space Based Adversarial Text Generation | 4, 5 | 0.50 |
| 1103 | 4.50 | Pooling Is Neither Necessary Nor Sufficient For Appropriate Deformation Stability In Cnns | 4, 5 | 0.50 |
| 1104 | 4.50 | Look Ma, No Gans! Image Transformation With Modifae | 4, 5 | 0.50 |
| 1105 | 4.50 | A Priori Estimates Of The Generalization Error For Two-layer Neural Networks | 4, 5 | 0.50 |
| 1106 | 4.50 | Pushing The Bounds Of Dropout | 5, 4 | 0.50 |
| 1107 | 4.50 | Deep Neuroevolution: Genetic Algorithms Are A Competitive Alternative For Training Deep Neural Networks For Reinforcement Learning | 3, 6 | 1.50 |
| 1108 | 4.50 | Pruning With Hints: An Efficient Framework For Model Acceleration | 5, 4 | 0.50 |
| 1109 | 4.50 | Partially Mutual Exclusive Softmax For Positive And Unlabeled Data | 4, 5 | 0.50 |
| 1110 | 4.50 | Countdown Regression: Sharp And Calibrated Survival Predictions | 4, 5 | 0.50 |
| 1111 | 4.50 | Context Dependent Modulation Of Activation Function | 4, 4, 4, 6 | 0.87 |
| 1112 | 4.50 | Solar: Deep Structured Representations For Model-based Reinforcement Learning | 5, 4 | 0.50 |
| 1113 | 4.50 | Transfer Value Or Policy? A Value-centric Framework Towards Transferrable Continuous Reinforcement Learning | 4, 5 | 0.50 |
| 1114 | 4.50 | Sufficient Conditions For Robustness To Adversarial Examples: A Theoretical And Empirical Study With Bayesian Neural Networks | 5, 4 | 0.50 |
| 1115 | 4.50 | Relwalk -- A Latent Variable Model Approach To Knowledge Graph Embedding | 5, 4 | 0.50 |
| 1116 | 4.50 | Consistent Jumpy Predictions For Videos And Scenes | 4, 5 | 0.50 |
| 1117 | 4.50 | Shaping Representations Through Communication | 4, 5 | 0.50 |
| 1118 | 4.50 | A Better Baseline For Second Order Gradient Estimation In Stochastic Computation Graphs | 6, 3 | 1.50 |
| 1119 | 4.50 | Iteratively Learning From The Best | 3, 6 | 1.50 |
| 1120 | 4.50 | Teaching To Teach By Structured Dark Knowledge | 3, 6 | 1.50 |
| 1121 | 4.50 | How Training Data Affect The Accuracy And Robustness Of Neural Networks For Image Classification | 5, 4 | 0.50 |
| 1122 | 4.50 | Aciq: Analytical Clipping For Integer Quantization Of Neural Networks | 4, 5 | 0.50 |
| 1123 | 4.50 | Synthetic Datasets For Neural Program Synthesis | 5, 4 | 0.50 |
| 1124 | 4.50 | Fast Exploration With Simplified Models And Approximately Optimistic Planning In Model Based Reinforcement Learning | 5, 4 | 0.50 |
| 1125 | 4.50 | Online Abstraction With Mdp Homomorphisms For Deep Learning | 4, 5 | 0.50 |
| 1126 | 4.50 | Visual Imitation With A Minimal Adversary | 3, 6 | 1.50 |
| 1127 | 4.50 | Unsupervised Classification Into Unknown K Classes | 4, 5 | 0.50 |
| 1128 | 4.33 | A Model Cortical Network For Spatiotemporal Sequence Learning And Prediction | 3, 7, 3 | 1.89 |
| 1129 | 4.33 | Neural Probabilistic Motor Primitives For Humanoid Control | 3, 6, 4 | 1.25 |
| 1130 | 4.33 | Augmented Cyclic Adversarial Learning For Domain Adaptation | 6, 3, 4 | 1.25 |
| 1131 | 4.33 | Locally Linear Unsupervised Feature Selection | 4, 6, 3 | 1.25 |
| 1132 | 4.33 | Mental Fatigue Monitoring Using Brain Dynamics Preferences | 7, 4, 2 | 2.05 |
| 1133 | 4.33 | Discovering Low-precision Networks Close To Full-precision Networks For Efficient Embedded Inference | 5, 3, 5 | 0.94 |
| 1134 | 4.33 | Combining Global Sparse Gradients With Local Gradients | 5, 5, 3 | 0.94 |
| 1135 | 4.33 | Shamann: Shared Memory Augmented Neural Networks | 4, 5, 4 | 0.47 |
| 1136 | 4.33 | Learning Grounded Sentence Representations By Jointly Using Video And Text Information | 4, 3, 6 | 1.25 |
| 1137 | 4.33 | Where And When To Look? Spatial-temporal Attention For Action Recognition In Videos | 6, 3, 4 | 1.25 |
| 1138 | 4.33 | Plan Online, Learn Offline: Efficient Learning And Exploration Via Model-based Control | 6, 3, 4 | 1.25 |
| 1139 | 4.33 | Dvolver: Efficient Pareto-optimal Neural Network Architecture Search | 4, 5, 4 | 0.47 |
| 1140 | 4.33 | Dppnet: Approximating Determinantal Point Processes With Deep Networks | 3, 5, 5 | 0.94 |
| 1141 | 4.33 | Adaptive Convolutional Neural Networks | 5, 4, 4 | 0.47 |
| 1142 | 4.33 | Total Style Transfer With A Single Feed-forward Network | 4, 5, 4 | 0.47 |
| 1143 | 4.33 | Variational Domain Adaptation | 4, 4, 5 | 0.47 |
| 1144 | 4.33 | Modular Deep Probabilistic Programming | 3, 4, 6 | 1.25 |
| 1145 | 4.33 | Wasserstein Proximal Of Gans | 3, 6, 4 | 1.25 |
| 1146 | 4.33 | Successor Uncertainties: Exploration And Uncertainty In Temporal Difference Learning | 4, 5, 4 | 0.47 |
| 1147 | 4.33 | Sense: Semantically Enhanced Node Sequence Embedding | 4, 4, 5 | 0.47 |
| 1148 | 4.33 | How To Learn (and How Not To Learn) Multi-hop Reasoning With Memory Networks | 3, 5, 5 | 0.94 |
| 1149 | 4.33 | Backdrop: Stochastic Backpropagation | 5, 3, 5 | 0.94 |
| 1150 | 4.33 | Incsql: Training Incremental Text-to-sql Parsers With Non-deterministic Oracles | 4, 6, 3 | 1.25 |
| 1151 | 4.33 | Architecture Compression | 6, 3, 4 | 1.25 |
| 1152 | 4.33 | W2gan: Recovering An Optimal Transportmap With A Gan | 6, 3, 4 | 1.25 |
| 1153 | 4.33 | A Guider Network For Multi-dual Learning | 4, 5, 4 | 0.47 |
| 1154 | 4.33 | Online Learning For Supervised Dimension Reduction | 2, 5, 6 | 1.70 |
| 1155 | 4.33 | Universal Attacks On Equivariant Networks | 4, 4, 5 | 0.47 |
| 1156 | 4.33 | Learning What To Remember: Long-term Episodic Memory Networks For Learning From Streaming Data | 5, 4, 4 | 0.47 |
| 1157 | 4.33 | Mean Replacement Pruning | 5, 4, 4 | 0.47 |
| 1158 | 4.33 | Deep Ensemble Bayesian Active Learning : Adressing The Mode Collapse Issue In Monte Carlo Dropout Via Ensembles | 4, 4, 5 | 0.47 |
| 1159 | 4.33 | Feed: Feature-level Ensemble Effect For Knowledge Distillation | 5, 4, 4 | 0.47 |
| 1160 | 4.33 | Classifier-agnostic Saliency Map Extraction | 4, 5, 4 | 0.47 |
| 1161 | 4.33 | Stacked U-nets: A No-frills Approach To Natural Image Segmentation | 5, 3, 5 | 0.94 |
| 1162 | 4.33 | Boltzmann Weighting Done Right In Reinforcement Learning | 4, 4, 5 | 0.47 |
| 1163 | 4.33 | Looking Inside The Black Box: Assessing The Modular Structure Of Deep Generative Models With Counterfactuals | 6, 4, 3 | 1.25 |
| 1164 | 4.33 | On Breiman’s Dilemma In Neural Networks: Success And Failure Of Normalized Margins | 3, 5, 5 | 0.94 |
| 1165 | 4.33 | Isolating Effects Of Age With Fair Representation Learning When Assessing Dementia | 4, 4, 5 | 0.47 |
| 1166 | 4.33 | Efficient Convolutional Neural Network Training With Direct Feedback Alignment | 4, 4, 5 | 0.47 |
| 1167 | 4.33 | Pseudosaccades: A Simple Ensemble Scheme For Improving Classification Performance Of Deep Nets | 5, 4, 4 | 0.47 |
| 1168 | 4.33 | Meta-learning With Individualized Feature Space For Few-shot Classification | 5, 5, 3 | 0.94 |
| 1169 | 4.33 | Inter-bmv: Interpolation With Block Motion Vectors For Fast Semantic Segmentation On Video | 5, 3, 5 | 0.94 |
| 1170 | 4.33 | Select Via Proxy: Efficient Data Selection For Training Deep Networks | 4, 4, 5 | 0.47 |
| 1171 | 4.33 | Open Vocabulary Learning On Source Code With A Graph-structured Cache | 3, 4, 6 | 1.25 |
| 1172 | 4.33 | Targeted Adversarial Examples For Black Box Audio Systems | 4, 6, 3 | 1.25 |
| 1173 | 4.33 | Topicgan: Unsupervised Text Generation From Explainable Latent Topics | 4, 4, 5 | 0.47 |
| 1174 | 4.33 | Visual Imitation Learning With Recurrent Siamese Networks | 4, 4, 5 | 0.47 |
| 1175 | 4.33 | Rating Continuous Actions In Spatial Multi-agent Problems | 5, 4, 4 | 0.47 |
| 1176 | 4.33 | Exploration By Uncertainty In Reward Space | 5, 5, 3 | 0.94 |
| 1177 | 4.33 | Mixfeat: Mix Feature In Latent Space Learns Discriminative Space | 5, 4, 4 | 0.47 |
| 1178 | 4.33 | Nice: Noise Injection And Clamping Estimation For Neural Network Quantization | 4, 5, 4 | 0.47 |
| 1179 | 4.33 | Bridging Hmms And Rnns Through Architectural Transformations | 5, 3, 5 | 0.94 |
| 1180 | 4.33 | Generalized Label Propagation Methods For Semi-supervised Learning | 4, 3, 6 | 1.25 |
| 1181 | 4.33 | Learning Actionable Representations With Goal Conditioned Policies | 4, 5, 4 | 0.47 |
| 1182 | 4.33 | Manifoldnet: A Deep Neural Network For Manifold-valued Data | 5, 4, 4 | 0.47 |
| 1183 | 4.33 | Dual Learning: Theoretical Study And Algorithmic Extensions | 6, 2, 5 | 1.70 |
| 1184 | 4.33 | Confidence Regularized Self-training | 3, 5, 5 | 0.94 |
| 1185 | 4.33 | Representation Flow For Action Recognition | 3, 5, 5 | 0.94 |
| 1186 | 4.33 | Learning A Neural-network-based Representation For Open Set Recognition | 4, 4, 5 | 0.47 |
| 1187 | 4.33 | Feature Matters: A Stage-by-stage Approach For Task Independent Knowledge Transfer | 5, 4, 4 | 0.47 |
| 1188 | 4.33 | Over-parameterization Improves Generalization In The Xor Detection Problem | 4, 4, 5 | 0.47 |
| 1189 | 4.33 | Mumomaml: Model-agnostic Meta-learning For Multimodal Task Distributions | 3, 5, 5 | 0.94 |
| 1190 | 4.33 | Variational Recurrent Models For Representation Learning | 5, 3, 5 | 0.94 |
| 1191 | 4.33 | The Cakewalk Method | 5, 4, 4 | 0.47 |
| 1192 | 4.33 | Pixel Redrawn For A Robust Adversarial Defense | 4, 6, 3 | 1.25 |
| 1193 | 4.33 | Jumpout: Improved Dropout For Deep Neural Networks With Rectified Linear Units | 5, 4, 4 | 0.47 |
| 1194 | 4.33 | A Fast Quasi-newton-type Method For Large-scale Stochastic Optimisation | 5, 5, 3 | 0.94 |
| 1195 | 4.33 | Meta-learning Neural Bloom Filters | 3, 5, 5 | 0.94 |
| 1196 | 4.33 | Provable Defenses Against Spatially Transformed Adversarial Inputs: Impossibility And Possibility Results | 5, 3, 5 | 0.94 |
| 1197 | 4.33 | Q-neurons: Neuron Activations Based On Stochastic Jackson's Derivative Operators | 6, 2, 5 | 1.70 |
| 1198 | 4.33 | Correction Networks: Meta-learning For Zero-shot Learning | 4, 4, 5 | 0.47 |
| 1199 | 4.33 | Deep Perm-set Net: Learn To Predict Sets With Unknown Permutation And Cardinality Using Deep Neural Networks | 7, 3, 3 | 1.89 |
| 1200 | 4.33 | Music Transformer | 4, 4, 5 | 0.47 |
| 1201 | 4.33 | No Pressure! Addressing Problem Of Local Minima In Manifold Learning | 5, 4, 4 | 0.47 |
| 1202 | 4.33 | Hierarchical Reinforcement Learning Via Advantage-weighted Information Maximization | 5, 3, 5 | 0.94 |
| 1203 | 4.33 | Learning Physics Priors For Deep Reinforcement Learing | 5, 3, 5 | 0.94 |
| 1204 | 4.33 | Pie: Pseudo-invertible Encoder | 3, 5, 5 | 0.94 |
| 1205 | 4.33 | Unification Of Recurrent Neural Network Architectures And Quantum Inspired Stable Design | 4, 4, 5 | 0.47 |
| 1206 | 4.33 | Adversarial Examples Are A Natural Consequence Of Test Error In Noise | 4, 5, 4 | 0.47 |
| 1207 | 4.33 | Explaining Neural Networks Semantically And Quantitatively | 5, 4, 4 | 0.47 |
| 1208 | 4.33 | Realistic Adversarial Examples In 3d Meshes | 5, 5, 3 | 0.94 |
| 1209 | 4.33 | On The Convergence And Robustness Of Batch Normalization | 6, 4, 3 | 1.25 |
| 1210 | 4.33 | From Nodes To Networks: Evolving Recurrent Neural Networks | 5, 4, 4 | 0.47 |
| 1211 | 4.33 | A Single Shot Pca-driven Analysis Of Network Structure To Remove Redundancy | 4, 4, 5 | 0.47 |
| 1212 | 4.33 | Structured Prediction Using Cgans With Fusion Discriminator | 7, 3, 3 | 1.89 |
| 1213 | 4.33 | Modeling Dynamics Of Biological Systems With Deep Generative Neural Networks | 6, 4, 3 | 1.25 |
| 1214 | 4.33 | Latent Transformations For Object View Points Synthesis | 3, 5, 5 | 0.94 |
| 1215 | 4.33 | Generative Adversarial Network Training Is A Continual Learning Problem | 5, 3, 5 | 0.94 |
| 1216 | 4.33 | Adversarial Decomposition Of Text Representation | 3, 6, 4 | 1.25 |
| 1217 | 4.33 | End-to-end Hierarchical Text Classification With Label Assignment Policy | 5, 4, 4 | 0.47 |
| 1218 | 4.33 | Opportunistic Learning: Budgeted Cost-sensitive Learning From Data Streams | 4, 5, 4 | 0.47 |
| 1219 | 4.33 | Beyond Winning And Losing: Modeling Human Motivations And Behaviors With Vector-valued Inverse Reinforcement Learning | 5, 4, 4 | 0.47 |
| 1220 | 4.33 | Generative Models From The Perspective Of Continual Learning | 4, 5, 4 | 0.47 |
| 1221 | 4.33 | Evolutionary-neural Hybrid Agents For Architecture Search | 5, 4, 4 | 0.47 |
| 1222 | 4.33 | Blackmarks: Black-box Multi-bit Watermarking For Deep Neural Networks | 5, 4, 4 | 0.47 |
| 1223 | 4.33 | Harmonic Unpaired Image-to-image Translation | 4, 5, 4 | 0.47 |
| 1224 | 4.33 | Representation Compression And Generalization In Deep Neural Networks | 6, 3, 4 | 1.25 |
| 1225 | 4.33 | Combining Learned Representations For Combinatorial Optimization | 4, 4, 5 | 0.47 |
| 1226 | 4.33 | On Generalization Bounds Of A Family Of Recurrent Neural Networks | 4, 6, 3 | 1.25 |
| 1227 | 4.33 | Adaptive Convolutional Relus | 5, 4, 4 | 0.47 |
| 1228 | 4.33 | Modulating Transfer Between Tasks In Gradient-based Meta-learning | 5, 4, 4 | 0.47 |
| 1229 | 4.33 | Sample Efficient Deep Neuroevolution In Low Dimensional Latent Space | 4, 5, 4 | 0.47 |
| 1230 | 4.33 | Context-adaptive Entropy Model For End-to-end Optimized Image Compression | 4, 3, 6 | 1.25 |
| 1231 | 4.33 | Task-gan For Improved Gan Based Image Restoration | 4, 5, 4 | 0.47 |
| 1232 | 4.33 | Confidence Calibration In Deep Neural Networks Through Stochastic Inferences | 5, 3, 5 | 0.94 |
| 1233 | 4.33 | Recycling The Discriminator For Improving The Inference Mapping Of Gan | 3, 3, 7 | 1.89 |
| 1234 | 4.33 | Network Reparameterization For Unseen Class Categorization | 5, 3, 5 | 0.94 |
| 1235 | 4.33 | Odin: Outlier Detection In Neural Networks | 5, 4, 4 | 0.47 |
| 1236 | 4.33 | Unsupervised Latent Tree Induction With Deep Inside-outside Recursive Auto-encoders | 5, 6, 2 | 1.70 |
| 1237 | 4.33 | Meta-learning To Guide Segmentation | 7, 3, 3 | 1.89 |
| 1238 | 4.33 | An Adversarial Learning Framework For A Persona-based Multi-turn Dialogue Model | 6, 3, 4 | 1.25 |
| 1239 | 4.33 | On The Effect Of The Activation Function On The Distribution Of Hidden Nodes In A Deep Network | 4, 5, 4 | 0.47 |
| 1240 | 4.33 | Do Language Models Have Common Sense? | 5, 4, 4 | 0.47 |
| 1241 | 4.33 | Log Hyperbolic Cosine Loss Improves Variational Auto-encoder | 4, 4, 5 | 0.47 |
| 1242 | 4.33 | Compound Density Networks | 3, 5, 5 | 0.94 |
| 1243 | 4.33 | Optimal Attacks Against Multiple Classifiers | 4, 6, 3 | 1.25 |
| 1244 | 4.33 | Learning Corresponded Rationales For Text Matching | 6, 4, 3 | 1.25 |
| 1245 | 4.33 | Robust Determinantal Generative Classifier For Noisy Labels And Adversarial Attacks | 3, 6, 4 | 1.25 |
| 1246 | 4.33 | Auto-encoding Knockoff Generator For Fdr Controlled Variable Selection | 3, 4, 6 | 1.25 |
| 1247 | 4.33 | Interpreting Deep Neural Network: Fast Object Localization Via Sensitivity Analysis | 4, 6, 3 | 1.25 |
| 1248 | 4.33 | Robust Text Classifier On Test-time Budgets | 4, 4, 5 | 0.47 |
| 1249 | 4.33 | Asynchronous Sgd Without Gradient Delay For Efficient Distributed Training | 5, 4, 4 | 0.47 |
| 1250 | 4.33 | Large Batch Size Training Of Neural Networks With Adversarial Training And Second-order Information | 4, 5, 4 | 0.47 |
| 1251 | 4.33 | Compositional Gan: Learning Conditional Image Composition | 4, 4, 5 | 0.47 |
| 1252 | 4.33 | Progressive Weight Pruning Of Deep Neural Networks Using Admm | 5, 4, 4 | 0.47 |
| 1253 | 4.33 | Implicit Maximum Likelihood Estimation | 4, 4, 5 | 0.47 |
| 1254 | 4.33 | Learning Adversarial Examples With Riemannian Geometry | 6, 4, 3 | 1.25 |
| 1255 | 4.33 | Neuron Hierarchical Networks | 5, 4, 4 | 0.47 |
| 1256 | 4.33 | Contextualized Role Interaction For Neural Machine Translation | 4, 5, 4 | 0.47 |
| 1257 | 4.33 | In Your Pace: Learning The Right Example At The Right Time | 5, 4, 4 | 0.47 |
| 1258 | 4.33 | Local Stability And Performance Of Simple Gradient Penalty $\mu$-wasserstein Gan | 5, 4, 4 | 0.47 |
| 1259 | 4.33 | Magic Tunnels | 5, 4, 4 | 0.47 |
| 1260 | 4.33 | Stochastic Quantized Activation: To Prevent Overfitting In Fast Adversarial Training | 4, 5, 4 | 0.47 |
| 1261 | 4.33 | Low-cost Parameterizations Of Deep Convolutional Neural Networks | 4, 4, 5 | 0.47 |
| 1262 | 4.33 | From Adversarial Training To Generative Adversarial Networks | 3, 6, 4 | 1.25 |
| 1263 | 4.33 | Efficient Sequence Labeling With Actor-critic Training | 5, 4, 4 | 0.47 |
| 1264 | 4.33 | Hiding Objects From Detectors: Exploring Transferrable Adversarial Patterns | 6, 4, 3 | 1.25 |
| 1265 | 4.33 | Explainable Adversarial Learning: Implicit Generative Modeling Of Random Noise During Training For Adversarial Robustness | 3, 5, 5 | 0.94 |
| 1266 | 4.33 | Variadic Learning By Bayesian Nonparametric Deep Embedding | 5, 4, 4 | 0.47 |
| 1267 | 4.33 | Learning Hash Codes Via Hamming Distance Targets | 4, 5, 4 | 0.47 |
| 1268 | 4.33 | Composition And Decomposition Of Gans | 4, 5, 4 | 0.47 |
| 1269 | 4.33 | A Preconditioned Accelerated Stochastic Gradient Descent Algorithm | 4, 4, 5 | 0.47 |
| 1270 | 4.33 | Modulated Variational Auto-encoders For Many-to-many Musical Timbre Transfer | 5, 5, 3 | 0.94 |
| 1271 | 4.33 | On Inductive Biases In Deep Reinforcement Learning | 3, 3, 7 | 1.89 |
| 1272 | 4.33 | Improving Sample-based Evaluation For Generative Adversarial Networks | 5, 5, 3 | 0.94 |
| 1273 | 4.00 | Universal Discriminative Quantum Neural Networks | 5, 5, 2 | 1.41 |
| 1274 | 4.00 | Unsupervised Exploration With Deep Model-based Reinforcement Learning | 4, 4, 4 | 0.00 |
| 1275 | 4.00 | Improving Machine Classification Using Human Uncertainty Measurements | 6, 3, 3 | 1.41 |
| 1276 | 4.00 | Deep Generative Models For Learning Coherent Latent Representations From Multi-modal Data | 4, 4, 4 | 0.00 |
| 1277 | 4.00 | Robustness And Equivariance Of Neural Networks | 3, 4, 5 | 0.82 |
| 1278 | 4.00 | Learning Deep Embeddings In Krein Spaces | 4, 5, 3 | 0.82 |
| 1279 | 4.00 | Morpho-mnist: Quantitative Assessment And Diagnostics For Representation Learning | 3, 5, 4 | 0.82 |
| 1280 | 4.00 | Fatty And Skinny: A Joint Training Method Of Watermark Encoder And Decoder | 4, 4 | 0.00 |
| 1281 | 4.00 | Differentially Private Federated Learning: A Client Level Perspective | 4, 4, 4 | 0.00 |
| 1282 | 4.00 | On The Statistical And Information Theoretical Characteristics Of Dnn Representations | 5, 4, 3 | 0.82 |
| 1283 | 4.00 | Distilled Agent Dqn For Provable Adversarial Robustness | 5, 3, 4 | 0.82 |
| 1284 | 4.00 | Rnns With Private And Shared Representations For Semi-supervised Sequence Learning | 3, 5, 4 | 0.82 |
| 1285 | 4.00 | In Search Of Theoretically Grounded Pruning | 4, 3, 5 | 0.82 |
| 1286 | 4.00 | Label Smoothing And Logit Squeezing: A Replacement For Adversarial Training? | 6, 4, 2 | 1.63 |
| 1287 | 4.00 | Reinforcement Learning: From Temporal To Spatial Value Decomposition | 6, 2, 4 | 1.63 |
| 1288 | 4.00 | Hc-net: Memory-based Incremental Dual-network System For Continual Learning | 4, 4, 4 | 0.00 |
| 1289 | 4.00 | Deep Adversarial Forward Model | 4, 4, 4 | 0.00 |
| 1290 | 4.00 | Q-map: A Convolutional Approach For Goal-oriented Reinforcement Learning | 5, 4, 3 | 0.82 |
| 1291 | 4.00 | Hypergan: Exploring The Manifold Of Neural Networks | 4, 4, 4 | 0.00 |
| 1292 | 4.00 | Latent Domain Transfer: Crossing Modalities With Bridging Autoencoders | 4, 4, 4 | 0.00 |
| 1293 | 4.00 | Understanding The Effectiveness Of Lipschitz-continuity In Generative Adversarial Nets | 6, 1, 5 | 2.16 |
| 1294 | 4.00 | Functional Bayesian Neural Networks For Model Uncertainty Quantification | 3, 4, 5 | 0.82 |
| 1295 | 4.00 | Prob2vec: Mathematical Semantic Embedding For Problem Retrieval In Adaptive Tutoring | 3, 5, 4 | 0.82 |
| 1296 | 4.00 | Few-shot Classification On Graphs With Structural Regularized Gcns | 3, 5, 4 | 0.82 |
| 1297 | 4.00 | Continual Learning Via Explicit Structure Learning | 4, 4, 4 | 0.00 |
| 1298 | 4.00 | Layerwise Recurrent Autoencoder For General Real-world Traffic Flow Forecasting | 4, 5, 3 | 0.82 |
| 1299 | 4.00 | Overfitting Detection Of Deep Neural Networks Without A Hold Out Set | 4, 5, 3 | 0.82 |
| 1300 | 4.00 | Neural Rendering Model: Joint Generation And Prediction For Semi-supervised Learning | 5, 3 | 1.00 |
| 1301 | 4.00 | Exploration Of Efficient On-device Acoustic Modeling With Neural Networks | 4, 4, 4 | 0.00 |
| 1302 | 4.00 | Reinforced Pipeline Optimization: Behaving Optimally With Non-differentiabilities | 4, 5, 3 | 0.82 |
| 1303 | 4.00 | Two-timescale Networks For Nonlinear Value Function Approximation | 4 | 0.00 |
| 1304 | 4.00 | Nuts: Network For Unsupervised Telegraphic Summarization | 4, 4, 4 | 0.00 |
| 1305 | 4.00 | Towards More Theoretically-grounded Particle Optimization Sampling For Deep Learning | 5, 4, 3 | 0.82 |
| 1306 | 4.00 | Understanding The Asymptotic Performance Of Model-based Rl Methods | 6, 4, 2 | 1.63 |
| 1307 | 4.00 | Training Hard-threshold Networks With Combinatorial Search In A Discrete Target Propagation Setting | 3, 4, 5 | 0.82 |
| 1308 | 4.00 | The Effectiveness Of Layer-by-layer Training Using The Information Bottleneck Principle | 5, 2, 5 | 1.41 |
| 1309 | 4.00 | Better Accuracy With Quantified Privacy: Representations Learned Via Reconstructive Adversarial Network | 4, 5, 3 | 0.82 |
| 1310 | 4.00 | Semantic Parsing Via Cross-domain Schema | 3, 4, 5 | 0.82 |
| 1311 | 4.00 | Neural Network Cost Landscapes As Quantum States | 5, 3, 4 | 0.82 |
| 1312 | 4.00 | Second-order Adversarial Attack And Certifiable Robustness | 4, 5, 3 | 0.82 |
| 1313 | 4.00 | Learning To Search Efficient Densenet With Layer-wise Pruning | 4, 4, 4 | 0.00 |
| 1314 | 4.00 | Polar Prototype Networks | 5, 3, 4 | 0.82 |
| 1315 | 4.00 | Ain't Nobody Got Time For Coding: Structure-aware Program Synthesis From Natural Language | 4, 4, 4 | 0.00 |
| 1316 | 4.00 | Sample-efficient Policy Learning In Multi-agent Reinforcement Learning Via Meta-learning | 4 | 0.00 |
| 1317 | 4.00 | Reconciling Feature-reuse And Overfitting In Densenet With Specialized Dropout | 5, 3, 4 | 0.82 |
| 1318 | 4.00 | On The Use Of Convolutional Auto-encoder For Incremental Classifier Learning In Context Aware Advertisement | 5, 4, 3 | 0.82 |
| 1319 | 4.00 | Dual Importance Weight Gan | 4, 3, 5 | 0.82 |
| 1320 | 4.00 | Few-shot Intent Inference Via Meta-inverse Reinforcement Learning | 4, 4 | 0.00 |
| 1321 | 4.00 | Exploiting Invariant Structures For Compression In Neural Networks | 4, 4, 4 | 0.00 |
| 1322 | 4.00 | Chaingan: A Sequential Approach To Gans | 4, 4, 4 | 0.00 |
| 1323 | 4.00 | Distinguishability Of Adversarial Examples | 4, 4 | 0.00 |
| 1324 | 4.00 | Learning From Noisy Demonstration Sets Via Meta-learned Suitability Assessor | 4 | 0.00 |
| 1325 | 4.00 | Uncertainty-guided Lifelong Learning In Bayesian Networks | 4, 4, 4 | 0.00 |
| 1326 | 4.00 | Applications Of Gaussian Processes In Finance | 4, 5, 3 | 0.82 |
| 1327 | 4.00 | Merci: A New Metric To Evaluate The Correlation Between Predictive Uncertainty And True Error | 4, 5, 3 | 0.82 |
| 1328 | 4.00 | Adversarial Attacks For Optical Flow-based Action Recognition Classifiers | 4, 3, 5 | 0.82 |
| 1329 | 4.00 | Efficient Exploration Through Bayesian Deep Q-networks | 6, 4, 4, 2 | 1.41 |
| 1330 | 4.00 | Overcoming Catastrophic Forgetting Through Weight Consolidation And Long-term Memory | 4, 4, 4 | 0.00 |
| 1331 | 4.00 | Deepström Networks | 4, 5, 3 | 0.82 |
| 1332 | 4.00 | Assumption Questioning: Latent Copying And Reward Exploitation In Question Generation | 4, 3, 5 | 0.82 |
| 1333 | 4.00 | Sequenced-replacement Sampling For Deep Learning | 3, 5, 4 | 0.82 |
| 1334 | 4.00 | Evaluating Gans Via Duality | 4, 3, 5 | 0.82 |
| 1335 | 4.00 | A Teacher Student Network For Faster Video Classification | 4, 4, 4 | 0.00 |
| 1336 | 4.00 | Co-manifold Learning With Missing Data | 4, 4, 4 | 0.00 |
| 1337 | 4.00 | Deep Processing Of Structured Data | 4, 4 | 0.00 |
| 1338 | 4.00 | Decoupling Feature Extraction From Policy Learning: Assessing Benefits Of State Representation Learning In Goal Based Robotics | 5, 3, 4 | 0.82 |
| 1339 | 4.00 | Overlapping Community Detection With Graph Neural Networks | 5, 3, 4 | 0.82 |
| 1340 | 4.00 | Unsupervised Convolutional Neural Networks For Accurate Video Frame Interpolation With Integration Of Motion Components | 3, 5, 4 | 0.82 |
| 1341 | 4.00 | Classification Of Building Noise Type/position Via Supervised Learning | 4, 4 | 0.00 |
| 1342 | 4.00 | Sequence Modelling With Memory-augmented Recurrent Neural Networks | 4, 4, 4 | 0.00 |
| 1343 | 4.00 | Complexity Of Training Relu Neural Networks | 3, 5, 4 | 0.82 |
| 1344 | 4.00 | Relational Graph Attention Networks | 4, 4, 4 | 0.00 |
| 1345 | 4.00 | Incremental Hierarchical Reinforcement Learning With Multitask Lmdps | 3, 4, 5 | 0.82 |
| 1346 | 4.00 | Multi-objective Value Iteration With Parameterized Threshold-based Safety Constraints | 5, 3 | 1.00 |
| 1347 | 4.00 | Learning Latent Semantic Representation From Pre-defined Generative Model | 5, 3, 4 | 0.82 |
| 1348 | 4.00 | Microgan: Promoting Variety Through Microbatch Discrimination | 3, 3, 6 | 1.41 |
| 1349 | 4.00 | Mol-cyclegan - A Generative Model For Molecular Optimization | 4, 4, 4 | 0.00 |
| 1350 | 4.00 | Large-scale Classification Of Structured Objects Using A Crf With Deep Class Embedding | 3, 3, 6 | 1.41 |
| 1351 | 4.00 | Generalized Capsule Networks With Trainable Routing Procedure | 5, 3, 4 | 0.82 |
| 1352 | 4.00 | D2ke: From Distance To Kernel And Embedding Via Random Features For Structured Inputs | 4, 3, 5 | 0.82 |
| 1353 | 4.00 | Conditional Inference In Pre-trained Variational Autoencoders Via Cross-coding | 4, 4, 4 | 0.00 |
| 1354 | 4.00 | Graph Generation Via Scattering | 4, 4, 4 | 0.00 |
| 1355 | 4.00 | The Missing Ingredient In Zero-shot Neural Machine Translation | 5, 4, 3 | 0.82 |
| 1356 | 4.00 | Pa-gan: Improving Gan Training By Progressive Augmentation | 4, 3, 5 | 0.82 |
| 1357 | 4.00 | Understanding Opportunities For Efficiency In Single-image Super Resolution Networks | 4, 5, 3 | 0.82 |
| 1358 | 4.00 | The Wisdom Of The Crowd: Reliable Deep Reinforcement Learning Through Ensembles Of Q-functions | 4, 5, 3 | 0.82 |
| 1359 | 4.00 | Trajectory Vae For Multi-modal Imitation | 4, 4, 4 | 0.00 |
| 1360 | 4.00 | Dynamic Pricing On E-commerce Platform With Deep Reinforcement Learning | 4, 4, 4 | 0.00 |
| 1361 | 4.00 | Constrained Bayesian Optimization For Automatic Chemical Design | 3, 4, 5 | 0.82 |
| 1362 | 4.00 | Found By Nemo: Unsupervised Object Detection From Negative Examples And Motion | 5, 3, 4 | 0.82 |
| 1363 | 4.00 | Defactor: Differentiable Edge Factorization-based Probabilistic Graph Generation | 3, 5, 4 | 0.82 |
| 1364 | 4.00 | Mixture Of Pre-processing Experts Model For Noise Robust Deep Learning On Resource Constrained Platforms | 4, 4 | 0.00 |
| 1365 | 4.00 | A Generalized Active Learning Approach For Unsupervised Anomaly Detection | 4, 5, 3 | 0.82 |
| 1366 | 4.00 | Revisting Negative Transfer Using Adversarial Learning | 4, 2, 6 | 1.63 |
| 1367 | 4.00 | A Multi-modal One-class Generative Adversarial Network For Anomaly Detection In Manufacturing | 3, 4, 5 | 0.82 |
| 1368 | 4.00 | Generative Adversarial Interpolative Autoencoding: Adversarial Training On Latent Space Interpolations Encourage Convex Latent Distributions | 4, 4, 4 | 0.00 |
| 1369 | 4.00 | Learning To Control Visual Abstractions For Structured Exploration In Deep Reinforcement Learning | 4, 5, 3 | 0.82 |
| 1370 | 4.00 | Language Modeling With Graph Temporal Convolutional Networks | 4, 4, 4 | 0.00 |
| 1371 | 4.00 | Guaranteed Recovery Of One-hidden-layer Neural Networks Via Cross Entropy | 3, 4, 5 | 0.82 |
| 1372 | 4.00 | Iterative Binary Decisions | 4, 4, 4 | 0.00 |
| 1373 | 4.00 | Discovering General-purpose Active Learning Strategies | 4, 4, 4 | 0.00 |
| 1374 | 4.00 | Constraining Action Sequences With Formal Languages For Deep Reinforcement Learning | 5, 3, 4 | 0.82 |
| 1375 | 4.00 | Deeptwist: Learning Model Compression Via Occasional Weight Distortion | 5, 4, 3 | 0.82 |
| 1376 | 4.00 | S-system, Geometry, Learning, And Optimization: A Theory Of Neural Networks | 4 | 0.00 |
| 1377 | 4.00 | On The Selection Of Initialization And Activation Function For Deep Neural Networks | 3, 4, 5 | 0.82 |
| 1378 | 4.00 | Recovering The Lowest Layer Of Deep Networks With High Threshold Activations | 4 | 0.00 |
| 1379 | 4.00 | Learning To Adapt In Dynamic, Real-world Environments Through Meta-reinforcement Learning | 2, 6 | 2.00 |
| 1380 | 4.00 | Neural Mmo: A Massively Multiplayer Game Environment For Intelligent Agents | 3, 5 | 1.00 |
| 1381 | 4.00 | Fast Binary Functional Search On Graph | 4, 4 | 0.00 |
| 1382 | 4.00 | Activity Regularization For Continual Learning | 4, 4, 4 | 0.00 |
| 1383 | 4.00 | Cosine Similarity-based Adversarial Process | 4, 3, 5 | 0.82 |
| 1384 | 4.00 | The Forward-backward Embedding Of Directed Graphs | 5, 3, 4 | 0.82 |
| 1385 | 4.00 | Learning Representations In Model-free Hierarchical Reinforcement Learning | 5, 4, 3 | 0.82 |
| 1386 | 4.00 | Exploration Using Distributional Rl And Ucb | 4, 4, 4 | 0.00 |
| 1387 | 4.00 | Hyper-regularization: An Adaptive Choice For The Learning Rate In Gradient Descent | 4, 4, 4 | 0.00 |
| 1388 | 4.00 | Data Poisoning Attack Against Node Embedding Methods | 4, 4, 4 | 0.00 |
| 1389 | 3.67 | Interpretable Convolutional Filter Pruning | 4, 4, 3 | 0.47 |
| 1390 | 3.67 | Prior Networks For Detection Of Adversarial Attacks | 3, 4, 4 | 0.47 |
| 1391 | 3.67 | D-gan: Divergent Generative Adversarial Network For Positive Unlabeled Learning And Counter-examples Generation | 3, 5, 3 | 0.94 |
| 1392 | 3.67 | The Natural Language Decathlon: Multitask Learning As Question Answering | 5, 3, 3 | 0.94 |
| 1393 | 3.67 | Graph Spectral Regularization For Neural Network Interpretability | 4, 3, 4 | 0.47 |
| 1394 | 3.67 | Radial Basis Feature Transformation To Arm Cnns Against Adversarial Attacks | 4, 4, 3 | 0.47 |
| 1395 | 3.67 | Explaining Alphago: Interpreting Contextual Effects In Neural Networks | 3, 4, 4 | 0.47 |
| 1396 | 3.67 | Automatic Generation Of Object Shapes With Desired Functionalities | 5, 3, 3 | 0.94 |
| 1397 | 3.67 | An Attention-based Model For Learning Dynamic Interaction Networks | 4, 3, 4 | 0.47 |
| 1398 | 3.67 | Quantile Regression Reinforcement Learning With State Aligned Vector Rewards | 4, 3, 4 | 0.47 |
| 1399 | 3.67 | Generating Images From Sounds Using Multimodal Features And Gans | 3, 4, 4 | 0.47 |
| 1400 | 3.67 | Fake Sentence Detection As A Training Task For Sentence Encoding | 5, 3, 3 | 0.94 |
| 1401 | 3.67 | Improving Adversarial Discriminative Domain Adaptation | 4, 4, 3 | 0.47 |
| 1402 | 3.67 | Filter Training And Maximum Response: Classification Via Discerning | 2, 3, 6 | 1.70 |
| 1403 | 3.67 | Question Generation Using A Scratchpad Encoder | 4, 3, 4 | 0.47 |
| 1404 | 3.67 | R Esidual Networks Classify Inputs Based On Their Neural Transient Dynamics | 4, 2, 5 | 1.25 |
| 1405 | 3.67 | A Walk With Sgd: How Sgd Explores Regions Of Deep Network Loss? | 4, 4, 3 | 0.47 |
| 1406 | 3.67 | Distributed Deep Policy Gradient For Competitive Adversarial Environment | 4, 4, 3 | 0.47 |
| 1407 | 3.67 | Optimizing For Generalization In Machine Learning With Cross-validation Gradients | 5, 2, 4 | 1.25 |
| 1408 | 3.67 | Encoding Category Trees Into Word-embeddings Using Geometric Approach | 4, 4, 3 | 0.47 |
| 1409 | 3.67 | Feature Attribution As Feature Selection | 4, 4, 3 | 0.47 |
| 1410 | 3.67 | Synthnet: Learning Synthesizers End-to-end | 4, 4, 3 | 0.47 |
| 1411 | 3.67 | Gradmix: Multi-source Transfer Across Domains And Tasks | 5, 3, 3 | 0.94 |
| 1412 | 3.67 | Normalization Gradients Are Least-squares Residuals | 4, 4, 3 | 0.47 |
| 1413 | 3.67 | Learning Agents With Prioritization And Parameter Noise In Continuous State And Action Space | 3, 4, 4 | 0.47 |
| 1414 | 3.67 | Kmer2vec: Towards Transcriptomic Representations By Learning Kmer Embeddings | 4, 2, 5 | 1.25 |
| 1415 | 3.67 | Localized Random Projections Challenge Benchmarks For Bio-plausible Deep Learning | 5, 3, 3 | 0.94 |
| 1416 | 3.67 | Discrete Structural Planning For Generating Diverse Translations | 2, 3, 6 | 1.70 |
| 1417 | 3.67 | Image Score: How To Select Useful Samples | 4, 4, 3 | 0.47 |
| 1418 | 3.67 | Using Word Embeddings To Explore The Learned Representations Of Convolutional Neural Networks | 3, 4, 4 | 0.47 |
| 1419 | 3.67 | Riemannian Stochastic Gradient Descent For Tensor-train Recurrent Neural Networks | 4, 4, 3 | 0.47 |
| 1420 | 3.67 | Optimization On Multiple Manifolds | 7, 1, 3 | 2.49 |
| 1421 | 3.67 | Efficient Federated Learning Via Variational Dropout | 4, 4, 3 | 0.47 |
| 1422 | 3.67 | Na | 4, 4, 3 | 0.47 |
| 1423 | 3.67 | Deep Hierarchical Model For Hierarchical Selective Classification And Zero Shot Learning | 4, 5, 2 | 1.25 |
| 1424 | 3.67 | Dynamic Recurrent Language Model | 3, 4, 4 | 0.47 |
| 1425 | 3.67 | Object-contrastive Networks: Unsupervised Object Representations | 3, 3, 5 | 0.94 |
| 1426 | 3.67 | Inhibited Softmax For Uncertainty Estimation In Neural Networks | 4, 4, 3 | 0.47 |
| 1427 | 3.67 | Contextual Recurrent Convolutional Model For Robust Visual Learning | 4, 3, 4 | 0.47 |
| 1428 | 3.67 | Spectral Convolutional Networks On Hierarchical Multigraphs | 4, 3, 4 | 0.47 |
| 1429 | 3.67 | Deep Geometrical Graph Classification With Dynamic Pooling | 4, 3, 4 | 0.47 |
| 1430 | 3.67 | Graph Learning Network: A Structure Learning Algorithm | 4, 3, 4 | 0.47 |
| 1431 | 3.67 | Delibgan: Coarse-to-fine Text Generation Via Adversarial Network | 4, 3, 4 | 0.47 |
| 1432 | 3.67 | Unsupervised Monocular Depth Estimation With Clear Boundaries | 4, 4, 3 | 0.47 |
| 1433 | 3.67 | Visualizing And Understanding The Semantics Of Embedding Spaces Via Algebraic Formulae | 4, 3, 4 | 0.47 |
| 1434 | 3.67 | Learning Robust, Transferable Sentence Representations For Text Classification | 4, 3, 4 | 0.47 |
| 1435 | 3.67 | Optimized Gated Deep Learning Architectures For Sensor Fusion | 4, 4, 3 | 0.47 |
| 1436 | 3.67 | Why Do Neural Response Generation Models Prefer Universal Replies? | 3, 7, 1 | 2.49 |
| 1437 | 3.67 | Polycnn: Learning Seed Convolutional Filters | 3, 4, 4 | 0.47 |
| 1438 | 3.67 | Parametrizing Fully Convolutional Nets With A Single High-order Tensor | 4, 3, 4 | 0.47 |
| 1439 | 3.67 | Bilingual-gan: Neural Text Generation And Neural Machine Translation As Two Sides Of The Same Coin | 3, 4, 4 | 0.47 |
| 1440 | 3.67 | Diminishing Batch Normalization | 4, 3, 4 | 0.47 |
| 1441 | 3.67 | Text Embeddings For Retrieval From A Large Knowledge Base | 3, 3, 5 | 0.94 |
| 1442 | 3.67 | Pcnn: Environment Adaptive Model Without Finetuning | 4, 3, 4 | 0.47 |
| 1443 | 3.67 | Unsupervised One-to-many Image Translation | 3, 4, 4 | 0.47 |
| 1444 | 3.67 | Hierarchical Attention: What Really Counts In Various Nlp Tasks | 4, 3, 4 | 0.47 |
| 1445 | 3.67 | Dyncnn: An Effective Dynamic Architecture On Convolutional Neural Network For Surveillance Videos | 3, 4, 4 | 0.47 |
| 1446 | 3.67 | Feature Transformers: A Unified Representation Learning Framework For Lifelong Learning | 4, 3, 4 | 0.47 |
| 1447 | 3.67 | Adversarially Robust Training Through Structured Gradient Regularization | 4, 4, 3 | 0.47 |
| 1448 | 3.67 | Accelerating First Order Optimization Algorithms | 3, 4, 4 | 0.47 |
| 1449 | 3.67 | Unsupervised Video-to-video Translation | 3, 4, 4 | 0.47 |
| 1450 | 3.50 | Using Deep Siamese Neural Networks To Speed Up Natural Products Research | 3, 4 | 0.50 |
| 1451 | 3.50 | Difference-seeking Generative Adversarial Network | 4, 3 | 0.50 |
| 1452 | 3.50 | Rethinking Self-driving : Multi -task Knowledge For Better Generalization And Accident Explanation Ability | 4, 3 | 0.50 |
| 1453 | 3.50 | Geometric Augmentation For Robust Neural Network Classifiers | 4, 3 | 0.50 |
| 1454 | 3.50 | Neural Regression Tree | 3, 4 | 0.50 |
| 1455 | 3.50 | Controlling Over-generalization And Its Effect On Adversarial Examples Detection And Generation | 4, 3 | 0.50 |
| 1456 | 3.50 | Mctsbug: Generating Adversarial Text Sequences Via Monte Carlo Tree Search And Homoglyph Attack | 3, 4 | 0.50 |
| 1457 | 3.50 | Geometry Of Deep Convolutional Networks | 4, 3 | 0.50 |
| 1458 | 3.50 | Learn From Neighbour: A Curriculum That Train Low Weighted Samples By Imitating | 3, 4 | 0.50 |
| 1459 | 3.50 | The Body Is Not A Given: Joint Agent Policy Learning And Morphology Evolution | 3, 4 | 0.50 |
| 1460 | 3.50 | Learning To Reinforcement Learn By Imitation | 2, 5 | 1.50 |
| 1461 | 3.50 | Pearl: Prototype Learning Via Rule Lists | 3, 4 | 0.50 |
| 1462 | 3.33 | Learning And Data Selection In Big Datasets | 4, 3, 3 | 0.47 |
| 1463 | 3.33 | Deep Models Calibration With Bayesian Neural Networks | 4, 3, 3 | 0.47 |
| 1464 | 3.33 | Geometric Operator Convolutional Neural Network | 2, 5, 3 | 1.25 |
| 1465 | 3.33 | Neural Network Regression With Beta, Dirichlet, And Dirichlet-multinomial Outputs | 3, 3, 4 | 0.47 |
| 1466 | 3.33 | Uncertainty In Multitask Transfer Learning | 3, 2, 5 | 1.25 |
| 1467 | 3.33 | Gradient Acceleration In Activation Functions | 5, 2, 3 | 1.25 |
| 1468 | 3.33 | Linearizing Visual Processes With Deep Generative Models | 3, 3, 4 | 0.47 |
| 1469 | 3.33 | Step-wise Sensitivity Analysis: Identifying Partially Distributed Representations For Interpretable Deep Learning | 3, 4, 3 | 0.47 |
| 1470 | 3.33 | Accidental Explorationa Through Value Predictors | 4, 3, 3 | 0.47 |
| 1471 | 3.33 | Logit Regularization Methods For Adversarial Robustness | 3, 5, 2 | 1.25 |
| 1472 | 3.33 | Generative Model Based On Minimizing Exact Empirical Wasserstein Distance | 5, 2, 3 | 1.25 |
| 1473 | 3.33 | Iea: Inner Ensemble Average Within A Convolutional Neural Network | 4, 2, 4 | 0.94 |
| 1474 | 3.33 | Empirical Study Of Easy And Hard Examples In Cnn Training | 3, 4, 3 | 0.47 |
| 1475 | 3.33 | Attack Graph Convolutional Networks By Adding Fake Nodes | 4, 3, 3 | 0.47 |
| 1476 | 3.33 | Neural Distribution Learning For Generalized Time-to-event Prediction | 4, 3, 3 | 0.47 |
| 1477 | 3.33 | Combining Adaptive Algorithms And Hypergradient Method: A Performance And Robustness Study | 3, 3, 4 | 0.47 |
| 1478 | 3.33 | Beyond Games: Bringing Exploration To Robots In Real-world | 3, 3, 4 | 0.47 |
| 1479 | 3.33 | Detecting Topological Defects In 2d Active Nematics Using Convolutional Neural Networks | 4, 4, 2 | 0.94 |
| 1480 | 3.33 | Deconfounding Reinforcement Learning | 4, 4, 2 | 0.94 |
| 1481 | 3.33 | Neural Random Projections For Language Modelling | 3, 4, 3 | 0.47 |
| 1482 | 3.33 | Learning Spatio-temporal Representations Using Spike-based Backpropagation | 3, 4, 3 | 0.47 |
| 1483 | 3.33 | Human Action Recognition Based On Spatial-temporal Attention | 4, 3, 3 | 0.47 |
| 1484 | 3.33 | Associate Normalization | 3, 5, 2 | 1.25 |
| 1485 | 3.33 | Multi-scale Stacked Hourglass Network For Human Pose Estimation | 3, 4, 3 | 0.47 |
| 1486 | 3.33 | Major-minor Lstms For Word-level Language Model | 3, 4, 3 | 0.47 |
| 1487 | 3.33 | Behavior Module In Neural Networks | 3, 3, 4 | 0.47 |
| 1488 | 3.33 | Bigsage: Unsupervised Inductive Representation Learning Of Graph Via Bi-attended Sampling And Global-biased Aggregating | 2, 4, 4 | 0.94 |
| 1489 | 3.33 | Interpreting Layered Neural Networks Via Hierarchical Modular Representation | 4, 3, 3 | 0.47 |
| 1490 | 3.33 | Non-synergistic Variational Autoencoders | 3, 4, 3 | 0.47 |
| 1491 | 3.33 | Deterministic Policy Gradients With General State Transitions | 4, 5, 1 | 1.70 |
| 1492 | 3.33 | Understanding And Improving Sequence-labeling Ner With Self-attentive Lstms | 3, 3, 4 | 0.47 |
| 1493 | 3.33 | She2: Stochastic Hamiltonian Exploration And Exploitation For Derivative-free Optimization | 4, 3, 3 | 0.47 |
| 1494 | 3.33 | Encoder Discriminator Networks For Unsupervised Representation Learning | 3, 4, 3 | 0.47 |
| 1495 | 3.33 | A Quantifiable Testing Of Global Translational Invariance In Convolutional And Capsule Networks | 3, 4, 3 | 0.47 |
| 1496 | 3.00 | Attentive Explainability For Patient Tempo- Ral Embedding | 4, 3, 2 | 0.82 |
| 1497 | 3.00 | Probabilistic Program Induction For Intuitive Physics Game Play | 3, 4, 2 | 0.82 |
| 1498 | 3.00 | A Fully Automated Periodicity Detection In Time Series | 3 | 0.00 |
| 1499 | 3.00 | Stacking For Transfer Learning | 3, 4, 2 | 0.82 |
| 1500 | 3.00 | Irda Method For Sparse Convolutional Neural Networks | 3, 3 | 0.00 |
| 1501 | 3.00 | Spamhmm: Sparse Mixture Of Hidden Markov Models For Graph Connected Entities | 3, 3, 3 | 0.00 |
| 1502 | 3.00 | Bamboo: Ball-shape Data Augmentation Against Adversarial Attacks From All Directions | 3 | 0.00 |
| 1503 | 3.00 | One Bit Matters: Understanding Adversarial Examples As The Abuse Of Redundancy | 3, 3, 3 | 0.00 |
| 1504 | 3.00 | A Self-supervised Method For Mapping Human Instructions To Robot Policies | 4, 3, 2 | 0.82 |
| 1505 | 3.00 | Lsh Microbatches For Stochastic Gradients: Value In Rearrangement | 3 | 0.00 |
| 1506 | 3.00 | Real-time Neural-based Input Method | 3, 3, 3 | 0.00 |
| 1507 | 3.00 | A Forensic Representation To Detect Non-trivial Image Duplicates, And How It Applies To Semantic Segmentation | 4, 3, 2 | 0.82 |
| 1508 | 3.00 | Calibration Of Neural Network Logit Vectors To Combat Adversarial Attacks | 3, 2, 4 | 0.82 |
| 1509 | 3.00 | Variational Autoencoders For Text Modeling Without Weakening The Decoder | 4, 1, 4 | 1.41 |
| 1510 | 3.00 | Dopamine: A Research Framework For Deep Reinforcement Learning | 3, 3, 3 | 0.00 |
| 1511 | 3.00 | Predictive Local Smoothness For Stochastic Gradient Methods | 2, 4 | 1.00 |
| 1512 | 3.00 | Evaluation Methodology For Attacks Against Confidence Thresholding Models | 2, 3, 4 | 0.82 |
| 1513 | 3.00 | Differentiable Greedy Networks | 2, 4 | 1.00 |
| 1514 | 3.00 | An Exhaustive Analysis Of Lazy Vs. Eager Learning Methods For Real-estate Property Investment | 3, 4, 2 | 0.82 |
| 1515 | 3.00 | A Non-linear Theory For Sentence Embedding | 3, 3, 3 | 0.00 |
| 1516 | 3.00 | A Rate-distortion Theory Of Adversarial Examples | 4, 3, 2 | 0.82 |
| 1517 | 3.00 | Hybrid Policies Using Inverse Rewards For Reinforcement Learning | 2, 4 | 1.00 |
| 1518 | 3.00 | Featurized Bidirectional Gan: Adversarial Defense Via Adversarially Learned Semantic Inference | 3, 3, 3 | 0.00 |
| 1519 | 3.00 | Handling Concept Drift In Wifi-based Indoor Localization Using Representation Learning | 2, 3, 4 | 0.82 |
| 1520 | 3.00 | Mapping The Hyponymy Relation Of Wordnet Onto Vector Spaces | 3, 3, 3 | 0.00 |
| 1521 | 3.00 | An Analysis Of Composite Neural Network Performance From Function Composition Perspective | 3, 3, 3 | 0.00 |
| 1522 | 3.00 | Hr-td: A Regularized Td Method To Avoid Over-generalization | 4, 3, 2 | 0.82 |
| 1523 | 3.00 | End-to-end Multi-lingual Multi-speaker Speech Recognition | 3, 3, 3 | 0.00 |
| 1524 | 3.00 | Reversed Neural Network - Automatically Finding Nash Equilibrium | 2, 4 | 1.00 |
| 1525 | 3.00 | Learning With Reflective Likelihoods | 4, 2, 3 | 0.82 |
| 1526 | 3.00 | Classification In The Dark Using Tactile Exploration | 4, 3, 2 | 0.82 |
| 1527 | 3.00 | Learning Powerful Policies And Better Generative Models By Interaction | 3, 2, 4 | 0.82 |
| 1528 | 3.00 | Knowledge Distill Via Learning Neuron Manifold | 5, 1, 3 | 1.63 |
| 1529 | 3.00 | Attention Incorporate Network: A Network Can Adapt Various Data Size | 3, 4, 2 | 0.82 |
| 1530 | 3.00 | Generative Model For Material Irradiation Experiments Based On Prior Knowledge And Attention Mechanism | 3, 3 | 0.00 |
| 1531 | 3.00 | Reneg And Backseat Driver: Learning From Demonstration With Continuous Human Feedback | 3, 4, 2 | 0.82 |
| 1532 | 3.00 | Learning Of Sophisticated Curriculums By Viewing Them As Graphs Over Tasks | 3, 2, 4 | 0.82 |
| 1533 | 3.00 | Feature Quantization For Parsimonious And Meaningful Predictive Models | 2, 3, 4 | 0.82 |
| 1534 | 3.00 | From Amortised To Memoised Inference: Combining Wake-sleep And Variational-bayes For Unsupervised Few-shot Program Learning | 3, 3, 3 | 0.00 |
| 1535 | 3.00 | Nonlinear Channels Aggregation Networks For Deep Action Recognition | 3, 3, 3 | 0.00 |
| 1536 | 2.67 | Decoupling Gating From Linearity | 3, 2, 3 | 0.47 |
| 1537 | 2.67 | A Bird's Eye View On Coherence, And A Worm's Eye View On Cohesion | 2, 2, 4 | 0.94 |
| 1538 | 2.67 | Happier: Hierarchical Polyphonic Music Generative Rnn | 2, 3, 3 | 0.47 |
| 1539 | 2.67 | End-to-end Learning Of Video Compression Using Spatio-temporal Autoencoders | 3, 3, 2 | 0.47 |
| 1540 | 2.67 | Faster Training By Selecting Samples Using Embeddings | 3, 3, 2 | 0.47 |
| 1541 | 2.67 | Exponentially Decaying Flows For Optimization In Deep Learning | 3, 3, 2 | 0.47 |
| 1542 | 2.67 | Variational Sgd: Dropout , Generalization And Critical Point At The End Of Convexity | 4, 2, 2 | 0.94 |
| 1543 | 2.67 | Multiple Encoder-decoders Net For Lane Detection | 2, 2, 4 | 0.94 |
| 1544 | 2.67 | Explaining Adversarial Examples With Knowledge Representation | 3, 3, 2 | 0.47 |
| 1545 | 2.67 | Learning Goal-conditioned Value Functions With One-step Path Rewards Rather Than Goal-rewards | 4, 1, 3 | 1.25 |
| 1546 | 2.50 | Weak Contraction Mapping And Optimization | 1, 4 | 1.50 |
| 1547 | 2.50 | Deli-fisher Gan: Stable And Efficient Image Generation With Structured Latent Generative Space | 2, 3 | 0.50 |
| 1548 | 2.50 | Hierarchical Bayesian Modeling For Clustering Sparse Sequences In The Context Of Group Profiling | 3, 2 | 0.50 |
| 1549 | 2.50 | A Solution To China Competitive Poker Using Deep Learning | 3, 2 | 0.50 |
| 1550 | 2.33 | Advanced Neuroevolution: A Gradient-free Algorithm To Train Deep Neural Networks | 1, 1, 5 | 1.89 |
| 1551 | 2.33 | Vectorization Methods In Recommender System | 2, 2, 3 | 0.47 |
| 1552 | 2.33 | Pixel Chem: A Representation For Predicting Material Properties With Neural Network | 3, 1, 3 | 0.94 |
| 1553 | 2.33 | Hierarchical Deep Reinforcement Learning Agent With Counter Self-play On Competitive Games | 3, 2, 2 | 0.47 |
| 1554 | 2.33 | Training Variational Auto Encoders With Discrete Latent Representations Using Importance Sampling | 3, 1, 3 | 0.94 |
| 1555 | 2.33 | Generating Text Through Adversarial Training Using Skip-thought Vectors | 3, 2, 2 | 0.47 |
| 1556 | 2.25 | A Synaptic Neural Network And Synapse Learning | 2, 3, 2, 2 | 0.43 |
| 1557 | 2.00 | Object Detection Deep Learning Networks For Optical Character Recognition | 1, 2, 1, 4 | 1.22 |
| 1558 | 2.00 | A Case Study On Optimal Deep Learning Model For Uavs | 2 | 0.00 |
| 1559 | 2.00 | Psychophysical Vs. Learnt Texture Representations In Novelty Detection | 3, 1 | 1.00 |
| 1560 | 1.67 | Hierarchical Bayesian Modeling For Clustering Sparse Sequences In The Context Of Group Profiling | 2, 2, 1 | 0.47 |
| 1561 | nan | Exploration In Policy Mirror Descent |  | nan |
| 1562 | nan | Neural Network Bandit Learning By Last Layer Marginalization |  | nan |
| 1563 | nan | Scaling Up Deep Learning For Pde-based Models |  | nan |
| 1564 | nan | Isonetry : Geometry Of Critical Initializations And Training |  | nan |
| 1565 | nan | Confidence-based Graph Convolutional Networks For Semi-supervised Learning |  | nan |
| 1566 | nan | Statistical Characterization Of Deep Neural Networks And Their Sensitivity |  | nan |
| 1567 | nan | Teaching Machine How To Think By Natural Language: A Study On Machine Reading Comprehension |  | nan |
| 1568 | nan | Exploring Deep Learning Using Information Theory Tools And Patch Ordering |  | nan |
| 1569 | nan | Pass: Phased Attentive State Space Modeling Of Disease Progression Trajectories |  | nan |
| 1570 | nan | Program Synthesis With Learned Code Idioms |  | nan |
| 1571 | nan | Is Pgd-adversarial Training Necessary? Alternative Training Via A Soft-quantization Network With Noisy-natural Samples Only |  | nan |
| 1572 | nan | Neural Collobrative Networks |  | nan |
| 1573 | nan | Adversarial Defense Via Data Dependent Activation Function And Total Variation Minimization |  | nan |
| 1574 | nan | Show, Attend And Translate: Unsupervised Image Translation With Self-regularization And Attention |  | nan |
