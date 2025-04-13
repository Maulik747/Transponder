# Lab Course 24-01 'Think, GPT! Think!' - Introducing The Trans-Ponder

This is the repository for the lab course team "'Think, GPT! Think!' - Introducing The Trans-Ponder". Please add a README here.

Using the code contained here:

1. Install the BidirectionalUtils package first:

```
pip3 install notebooks/BidirectionalUtils
```

2. Then install all the dependencies in the requirements.txt file

```
pip3 install -r requirements.txt
```

<h2>Two Main Ideas<h2>

![Alt text](forced_encoder_figure.png?raw=true "Forced Encoder")

<p>In forced encoder,we generate k tokens autoregressively and when an uncertain token is detected, we generate k future tokens. Then, we replace the uncertain token with a custom token called <MASK> . Then we turn on the bidirectional mechanism which makes the uncertain token gather context from past and the future tokens (different from autoregressive) and reclassify by this token using a Neural Network, that was trained on MLM(Masked Lnaguage Modelling). </p>


![Alt text](dencoder_figure.png?raw=true " De-encoder")


The Dencoder architecture differs from the Froced Encoder because we use the entire hidden layer representation of the past and the future tokens to classify the uncertain token rather than just using the Masked Token. So, this classification differs from the simple MLM task.