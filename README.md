# Sequence Labelling (seq2seq) Model - News Tags

Matches words, embedded with word2vec to news item tags - in sequence.

By Adam Atkinson.

## Usage

### How to Test

#### Query Loop

    $ python ner_test.py

- Model, wordvec, and tagged news filepaths are hardcoded.
- Uses 2 layer BiLSTM by default.
- Runs an input loop which accepts a query from stdin (terminated with 'enter') then prints the predicted tags. Exits when it receives the single word 'exit'.

Example execution:

    $ python ner_test.py
    Type a query (type "exit" to exit):
    news about Obama

    news    B-NEWSTYPE
    about   O
    Obama   B-KEYWORDS

#### Tagged News Data File (training data)

    $ python my_test_script.py [# samples to read]

- Reads samples from news_tagged_data.txt, processes and predicts just like the 'ner_test.py' script does, and computes the percentage of correctly classified samples and frames.
- When the [# samples to read] argument isn't specified, the script reads the entire news_tagged_data.txt file.
- Also has hardcoded flags to turn on/off extra logging. By default it prints out the sample words, tags, and predicted tags whenever an incorrectly predicted sample is encountered.

Example execution:

    $ python my_test_script.py

    ...

    Bad Prediction!
    Words: ['home', 'trending', 'news']
    Tags : ['B-SECTION', 'B-NEWSTYPE', 'I-NEWSTYPE']
    Preds: ['B-KEYWORDS', 'B-NEWSTYPE', 'I-NEWSTYPE']

    ...

    ~~~ Summary ~~~
    # samples read = 1000
    Correctly classified samples = 0.9740
    Correctly classified frames = 0.9962

### How to Train Your Own

    $ python ner_train.py  your_model.hd5 [# BiLSTM layers] [# epochs]

Wordvec and tagged news filepaths are hardcoded.
Model parameters are hardcoded.

## Program and Script Structure

    data_util.py    // Reads, parses, and embeds data so it can be used by the model.
                    // Defines DataUtil class which owns parsed data and utility functions.

    data_test.py    // Tests data_util.py

    ner_model.py    // Implements the model: definition, training, testing, prediction.
                    // Defines NERModel class which owns model, parameters, and train/test functions.

    ner_test.py     // Reads sequences of words from the command line and prints the predicted tags.

    ner_train.py    // Trains a new model.

    my_test_script.py   // Script that reads news_tagged_data.txt and runs against the model.

    model_blstm_150_ep50.h5     // Single layer BiLSTM model
    model_blstm_150_150_ep50.h5 // Double layer BiLSTM model

## Environment

### Software Requirements

    Python 2.7.6
    Keras==1.0.8
    numpy==1.11.1
    pandas==0.18.1
    scipy==0.18.0
    Theano==0.8.2

### OS

    Linux adama-ideapad 3.19.0-68-generic #76~14.04.1-Ubuntu SMP Fri Aug 12 11:46:25 UTC 2016 x86_64 x86_64 x86_64 GNU/Linux

### Hardware

    Model:  Lenovo IdeaPad U430p Notebook
    CPU:    Intel(R) Core(TM) i5-4210U CPU @ 1.70 GHz , (32KB/256KB/3MB) L1/L2/L3 cache
    RAM:    8 GB (8GiB SODIMM DDR3 Synchronous @ 1600 MHz)
    GPU:    None 

## Performance

One BiLSTM Layer:

Epoch|Training Accuracy|Test Accuracy
-----|-----------------|-------------
10| 0.9513 | 0.9567
25| 0.9903 | 0.9890
50| 0.9988 | 0.9935

- 94% training accuracy by epoch 10
- 97% training accuracy by epoch 15
- 99% training accuracy by epoch 25

Two BiLSTM Layers:

Epoch|Training Accuracy|Test Accuracy
-----|-----------------|-------------
10| 0.9420 | 0.9556
25| 0.9924 | 0.9887
50| 0.9992 | 0.9929

- 94% training accuracy by epoch 10
- 98% training accuracy by epoch 15
- 99% training accuracy by epoch 25

The 2 layer BiLSTM model seems to overfit a very slightly more than the 1-layer model does, but I use the 2-layer one by default because stacked RNNs are cool :)

## Methodology 

### Overview

Network:    150-unit BiLSTM with dropout -> 150-unit BiLSTM with dropout -> Softmax // All stacked

Loss:       categorical cross-entopy

Optimizer:  Adam

### Preprocessing, Features, and Data Format

1. Encode the words (X)
    a. Assign each word a 300-length word2vec representation, from the provided file.
    b. If no vector exists for a word, sample a 300-length vector from a multivariate normal distribution and normalize it.

2. Encode the tags (y)
    a. Read through the tagged news data to get the number of tags (classes).
    b. Add 'nil' class for unknown data.
    c. Assign each class an id.
    d. Assign each class a 1-hot vector, length= # classes, where the tag id is the index of the 1.

3. Assemble the data
    a. Read the tagged data from the provided file.
    b. Determine the maximum sequence length out of all the data.
    c. For each (sentence, tags) example, 
        i. Map each word in sentence to a vector from 1).
        ii. Map each tag to a one-hot vector from 2).
        iii). Pad the sentence 

- # Features in a frame of X = 300
- # Features in a frame of y = 10
- # Total length of a sample X or y = max. seq. length = 29
- all_X_data.shape = [ n_samples, 29, 300 ]
- all_Y_data.shape = [ n_samples, 29, 10 ]

#### NOTE On Prediction

Unknown words encountered during prediction are assigned a unit, 300-length vector from a multivariate normal.

### Model Details and Design Choices

- This problem is known as sequence labelling. Given a (variable) length sequence of X's, for each element X in the sequence, predict a discrete y that is associated with X. This is a many-to-many sequence learning task.

- Missing Words Embedding:
    - A 3rd of all the words were missing vector representations - this is a significant amount of missing data. It is important to include this data in the model, and thus encode them.
    - Word2vec initializes each word vector embedding as a sample from the multivariate normal distribution, I felt this would be adequate for the 

- Sequence Length:
    - Training sequences are padded with 'nil' frames, which are a 300-length zero vector (X) and 'nil' one-hot vector to make input shapes uniform.
    - The network learns to associate nil frames with nil classes

- Recurrent Unit Selection:
    - LSTMs and GRUs prevent vanishing and exploding gradients in RNNs by gating the activation signals. 
    - LSTMs and GRUs also learn long term dependencies, which is helpful
    - I found experimentally that LSTM performed slightly better than GRU given 50 or fewer training epochs.
    - I found that 100-300 units worked well and converged quickly. I decided to stick with 150.

- Bi-Directional:
    - The news tags depend on the words that came before and that come after, so lookahead ability would be nice. 
    - BiDi-RNNS have 2 RNNS, one operating in each time direction and feeding into each out to facilitate this.

- Multi-class, Loss, and Softmax:
    - Categorical cross entropy is the goto loss function for multiclass classification. In this task I minimize the cross entropy between the true and predicted label for a sample. 
    - Multi-class classification in neural nets usually encodes the class as a one hot vector, then has an output unit for each possible class = each elem of class vector. The class is then determined by which output unit / elem of output vector has the highest strength.
    - We softmax the output vector to induce a probability distribution, so PR(y=c|X) = vec[c].
    - The final recurrent layer of my model has 10 outputs (one for each class + nil) which is softmaxed. This layer is present at every time step (last layer in stack) to provide an estimate for every frame.

- Optimizer:
    - Adam and RMSprop are defacto nowadays for deep learning. They accelerate convergence by looking at the history of the gradients (and some other things).
    - Adam worked best for my models here.

- Stacking:
    - Adding another recurrent layer allows the model to learn abstract relationships in the series. 
    - I found 2 recurrent layers was sufficient for this task.

- Batch Size:
    - < 50 made the loss erratic and often settled at a sub-optimal (local) point.
    - > 100 took too long to converge.
    - I found 100 worked well for this task.

- Dropout:
    - Because the layers are dense, and the sequences are longer and have many inter-dependencies, I felt that the model wold benefit from dropout (randomly removing units while training). I felt this make the model more robust to overfitting and help it cope with different word orderings in the training sequence.
    - Dropout of 0.2-0.4 seemed to work well for my model without slowing training time significantly. I chose a probability of 0.2.
    - In my experience and reading, it seems advisable to always include some dropout when training deep neural networks.

- Regularization:
    - I played with regularization and found its impact to be insignificant. After 25-50 training epochs, test error was within 0-2% of training error.

- Train/Test split = 80/20

### Things I Could Work On (TODO)

- Add and experiment with batch normalization to improve accuracy and training time.
- Experiment more with regularization constant (0.0001 -> 0.001).
- Implement a loss function that reduces the number of miss-classified frames in a sample, and see how this performs. The reason I haven't done this is because I don't know enough Theano to do this right now.
- Tinker with LSTM units themselves (dropout and weighting parameters).
- Learn my own embedding for words.
- Evaluate model on an established NER dataset.

## Notes

- This is my first time using Keras! I learned Keras to complete this challenge.
- This is my first time implementing recurrent neural networks. It was especially challenging since the problem involves variable length sequences, many-to-many sequence learning, and forward and backward dependencies - plus I used stacked, bidirectional LSTM layers to approach it.
- I knew about recurrent neural networks, LSTMs, and bidirectional RNNs going into this challenge. I gained more experience implementing and tuning these models.
- Going into this challenge I had some experience implementing feedforward (simple MLP) networks and convolutional neural networks in TensorFlow.

## References
- https://keras.io/
- https://www.cs.toronto.edu/~graves/preprint.pdf
- http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf
- http://cs231n.github.io/neural-networks-2/#reg
- https://www.quora.com/What-are-differences-between-update-rules-like-AdaDelta-RMSProp-AdaGrad-and-AdaM
