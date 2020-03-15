# Word2Vec 

This repo implements [Word2Vec algorithm](https://en.wikipedia.org/wiki/Word2vec) using the skip-gram architecture. 

## Resources used 
This repo is created by referring to these materials:

* A really good [conceptual overview](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/) of Word2Vec from Chris McCormick 
* [First Word2Vec paper](https://arxiv.org/pdf/1301.3781.pdf) from Mikolov et al.
* [Neural Information Processing Systems, paper](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) with improvements for Word2Vec also from Mikolov et al

## Word2Vec
This implementation used the __skip-gram architecture__. I might implement it using __Continuous Bag-Of-Words__ in future. 

Also, in the second Word2Vec [paper]((http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)), the authors came up with few improvements:

- Subsampling frequent words to decrease the number of training examples
- Modified the optimization objective with something called __"Negative Sampling"__

My implementation hasn't used the "negative sampling" technique yet.