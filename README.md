# KExtor

## Description:
KExtor is a program that extracts domain keywords from a corpus. Given a corpus of text files, the program selects possible candidates and decides whether the term is a domain keyword or not based on domain relevance and domain consensus. 

### Extraction
The first step is to extract all terminology candidates from the corpus. The extractor considers all composite nouns in a paragraph if the paragraph:
* has at least *c* % capitalized words
* has at least *tok_min* tokens
* has no more than *tok_max* tokens
* ends with punctuation

Only after a paragraph was accepted the program will:
* split the paragraph into sentences
* tokenize it
* POS-Tags will be assigned to words

And finally candidates will be extracted with a regex grammar:
```
grammar = "NP:{(<NN.*>|<JJ.*>|<VB(G|D|N)>)<NN.*>}"
```

Once all the candidates have been collected, the program will compute a score for each one in order to create a set of keywords.

### Decision
After the Extractor selects the candidates, the Discriminator decides whether they are a keyword or not based on domain relevance and domain consensus. For this step a second neutral corpus is needed. By default the program will use the brown corpus (from nltk), but the user can uses a custom corpus by passing it as an argument (as a path to the directory containing the text files) to the program.

* First the conditional probability of each candidate *t* to appear in a corpus D is calculated:  
  <p align="center"><img src="img/cond_prob.png" alt="drawing" width="300"/></p>  

  where *cf(t, D)* is the total frequency of a term *t* in a corpus *D* and *T* is the set of all possible candidates *t*.  
<!-- $$ P(t|D) = \frac{cf(t, D)}{\sum_{t' \in T}(t', D)} $$ -->

* Domain Relevance will then be calculated with the following formula:  
   <p align="center"><img src="img/dom_rel.png" alt="drawing" width="350"/></p>  

  * DR = 1 the candidate only appear in the domain corpus  
  * DR < 0.5 the candidate appears more often in the reference corpus
<!-- $$ DR_{t, D_{dom}} = \frac{P(t|D_{dom})}{P(t|D_{dom}) + P(t|D_{ref})} $$ -->

* To calculate domain consensus it is first necessary to calculate the distribution of a candidate *t* in all documents from the domain corpus as:  
   <p align="center"><img src="img/dist.png" alt="drawing" width="300"/></p>   

  where *tf(t, d)* is the frequency of a candidate *t* in the document *d*.
<!-- $$ P_t(d) = \frac{tf(t, d)}{\sum_{d' \in D_{dom}} tf(t, d')} $$ -->


* Domain Consensus is the entropy of this distribution:  
   <p align="center"><img src="img/entropy.png" alt="drawing" width="500"/></p> 

<!-- $$ DC_{t, D_{dom}} = H(P_t(D)) =  \sum_{d' \in D_{dom}} (P_t(d') log\frac{1}{P_t(d')}) $$ -->


* Based on domain relevance and domain consensus a scoring function will decide if a candidate is a keyword. The score is calculated as:  
   <p align="center"><img src="img/score.png" alt="drawing" width="400"/></p> 

<!-- $$ f(t) = \alpha DR_{t, D_{dom}} + (1 - \alpha) DC_{t, D_{dom}} $$ -->

* A candidate will be considered a keyword if its score is higher than theta  
   <p align="center"><img src="img/decision.png" alt="drawing" width="200"/></p>   

  where *K* is the set of keywords
<!-- $$ f(t) > \theta \rightarrow t \in K $$ -->


## Requirements
* install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
* cd into the root directory of this repository
* create a new environment from the environment.yml file
```
conda env create -f environment.yml
```

* activate the new environment
```
conda activate kextractor
```

Download Data:
```
python initial_setup.py
```
The script will download following packages from nltk:
* corpus reuters
* stopwords
* words
* punkt
* avereged multiperceptron tagger


Developed on: Ubuntu 20.04  
Tested on: Ubuntu 20.04, Windows 10

## Synopsis
```
usage: keyword_extractor.py [-h] [--reference REF] [--min_sen N] [--max_cap N] [--min_tok N]
                            [--max_tok N] [--min_freq N] [--alpha N] [--theta N]
                            [--not-paragraphed] [--validation] [--verbose] [--single]
                            DOMAIN

positional arguments:
  DOMAIN             Path to the domain corpus

optional arguments:
  -h, --help         show this help message and exit
  --reference REF    Path to the reference Corpus
  --min_sen N        Minimum ammount of sentences per paragraph (Default: 2)
  --max_cap N        Max ammount of capitalized words per paragraph (Default: 70)
  --min_tok N        Minimum ammount of tokens per paragraph (Default: 5)
  --max_tok N        Maximum ammount of sentences per paragraph (Default: 20)
  --min_freq N       Minimum absolute frequence of a candidate (Default: 25)
  --alpha N          Alpha value for the discriminator (Default: 0.99)
  --theta N          Theta value for the discriminator (Default: 0.6)
  --not-paragraphed  Domain corpus has NOT one paragraph per line (Default: False)
  --validation       Validates candidates with a dictionary (Default: False)
  --verbose          Save domain relevance, consensus and rejected candidates (Default: False)
  --single           disable multiprocessing
```

The output will be saved in the ```output/``` directory.

### Examples:
```
$ python keyword_extractor.py data/acl_texts/ 
$ python keyword_extractor.py data/acl_texts/ --verbose
$ python keyword_extractor.py data/acl_texts/ --min_freq 15 --alpha 0.8 --theta 1.6
```

## Tests:
To run all tests:
```
$ python -m unittest
```

##  Known Bugs
All bugs are unknown.


## Resources
[Extracting domain terminologies from the World Wide Web](https://www.sigwac.org.uk/wiki/WAC5)
