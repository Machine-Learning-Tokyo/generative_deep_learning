# TL;DR: [An Overview of Normalization Methods in Deep Learning](https://mlexplained.com/2018/11/30/an-overview-of-normalization-methods-in-deep-learning/)

Author: [Emil VATAI](https://github.com/vatai), 2020-04-05.

The bottom line: The blogpost is a nice collection of TL;DRs/summaries
of interesting papers on different normalization methods.

Intro: Normalization is not as trivial as you might think.

1. What is Batch Normalization and Why is it Effective?:

   - Short into to batch normalization.
   - It is not clear why it works but it does (references included).
   - One possible explanation from the perspective of high-order
     interactions (according to the Deep Learning book, link to another blog post).

2. Why not Batch Normalization?:

   - Possible reason 1: Small batch size.
   - Possible reason 2: Recurrent connections.

3. What are the Alternatives?

   1. Weight normalization (with link to paper).  Good results on
      CIFAR-10 classification.
   2. Layer normalization, which normalizes the inputs across the
      features (with link to paper by Geoffery Hinton). Apparently
      this has some fancy math, explaining why this is more
      expressive/powerful than weight normalization. Good results on RNNs.
   3. Instance normalization (this is what we had in the Generative
      Deep Learning book). A little 2D image specific, generally
      useful for style transfer, and GANs.
   4. Group Normalization: half way between layer and instance
      normalization. Layer and instance did good on RNNs and style
      transfer, but did worse than batch normalization with image
      classification problems: group normalization did almost as
      good as batch normalization there, even outperform it when
      batch size was small.
   5. Batch Renormalization: try to use moving average (with link to
      paper).
   6. Batch-Instance Normalization (with link). This an interesting
      one, it has a learnable parameter which dictates if it acts as
      an instance normalization or a batch normalization.
   7. Switchable Normalization: Swiss army-knife which suggests to
      have all the normalizations available.
   8. Spectral Normalization (with link). GAN and Lipschitz constant
      related.
   9. ScaleNorm (with link): applied to the Transformer model.

4. Additional Pitfalls of Batch Normalization

   1. Dependence of the loss between samples in a minibatch: Link to a
      paper describing problems of batch normalization when using
      distributed training with different batch sizes. Ergo batch size
      must be the same across all machines. Batch normalization makes
      the loss function depend on the batch size.
   2. Fine tuning: when fine-tuning a network, should use the mean and
      variance computed on the original dataset or use the mean and
      variance of the mini-batches (link to a forum discussion
      thread).

5. Conclusion: we don't know squat about normalization, and how to do
   it right. There is still a ton of research to be done. There is a
   link to a paper on this topic as well.

# About this file
Mike sent
[this](https://mlexplained.com/2018/11/30/an-overview-of-normalization-methods-in-deep-learning/)
link to the #generativedeeplearning channel on Slack, and I want to
make a tl;dr: version. Since the original post is also a tl;dr, this
document is intended as a motivation and/or "what to expect"

# (Potential) TODOs

- Tag Mike?
- Add this to a review file?
- Add direct links to the papers (instead of just mentioning the links
  in the blog).
