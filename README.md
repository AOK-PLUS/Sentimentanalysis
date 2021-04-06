# Sentiment analysis: One language to rule them all

At AOK PLUS, we are always looking at improving our products.
One way we do that is by gathering feedback from our customers.
Especially the negative feedback might help us understand what is still missing or needs improvement.
However, as the largest health insurer in the states of Saxony and Thuringia, it is impossible to look into all feedback manually (we have over 3.4 Million customers).
Therefore, we are looking into prefiltering feedback automatically to help focus on the most important issues.
Sentiment analysis is one of the building blocks to accomplish that.

### Problem
Basic sentiment analysis can use polarity dictionaries, such as [SentiWS](https://www.aclweb.org/anthology/L10-1339/) which map a limited set of words in a dictionary to a polarity score, usually a float in the range [-1, 1].
A sentence is then a (possibly empty) vector of polarity scores on top of which a simple model can be trained to obtain an overall sentiment score for the sentence.
However, such models cannot grasp the context in which those words are embedded, leading to frequent misclassification.
Moreover, the limited vocabulary leads to many sentences with no polarity vector at all.

On the other hand, deep learning models give promising [results](https://www.tensorflow.org/tutorials/text/classify_text_with_bert).
They use state-of-the-art NLP models, such as BERT and append a classifier to train it on the sentiment analysis task.
Unfortunately, their success ultimately depends on large datasets for training which only exist in English.

### Idea
There is a recent family of NLP models that are language agnostic.
One example is the [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-base-br/1).
They generate an embedding for a sentence in any language, i.e. the input language is not a parameter of the model.
Therefore, two sentences with a similar meaning will end up close to each other inside the embedding space, regardless of their language.

### Working Hypothesis
If we fix the weights of the Universal Sentence Encoder and train a model that classifies sentiments, based on the embeddings of the Universal Sentence Encoder, we (1) keep the language agnostic property of the Universal Sentence Encoder and (2) by training the sentiment model on the output of of the Universal Sentence Encoder, we (hopefully) allow it to generalize to any language as the embeddings of a sentence with a similar meaning will be close.

# Implementation
We use the Yelp polarity [dataset](https://www.tensorflow.org/datasets/catalog/yelp_polarity_reviews) for training as it is closest to our use-case.
Using Tensorflow datasets it's as easy as running

``` {.bash}
tfds build yelp_polarity_reviews
```

in a terminal. This will download and prepare the data. After that, we can use it in our Python script:

``` {.python}
train, val, test = tfds.load(name="yelp_polarity_reviews", 
                             split=('train[:80%]', 'train[80%:]', 'test'),
                             as_supervised=True,
                             shuffle_files=True)
```

Next, we need to instantiate the Universal Sentence Encoder.
Keras offers a convenient method to embed models from Tensorflow Hub directly into a model:

``` {.python}
# model
text_input_USE = tf.keras.layers.Input(shape=(), dtype=tf.string, name="USE_input")
preprocessor = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2",
                              trainable=False,
                              name="USE_preprocessor")
encoder_inputs = preprocessor(text_input_USE)
encoder_USE = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-base-br/1",
                             trainable=False,
                             name="USE")
outputs = encoder_USE(encoder_inputs)["default"]
```

The authors also [suggest](https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-base-br/1) to perform a normalization step on the embeddings if we want to compute distances between embeddings of different sentences which is exactly what we want the model to do.
However, we had to adjust it slightly, such that it uses the tensorflow primitives, instead of numpy.
With that, it can be embedded directly into a Lambda Layer which reads the `outputs` from the Universal Sentence Encoder.

``` {.python}
# taken and adapted from:
# https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-base-br/1
def normalization(embeds):
  norms = tf.norm(embeds, 2, axis=1, keepdims=True)
  return embeds/norms

normalized = tf.keras.layers.Lambda(normalization)(outputs)
```

Note, that most online examples directly add the classifier after the NLP Model (be it BERT or Universal Sentence Encoder) but they make them trainable as well.
This is a sensible decision if you are training a sentiment classifier for English input only.
But it also means that the models might specialize on English and loose some of their generality with regards to language.
Originally, they are trained on several different languages simultaneously to force them into generalizing across languages (see [here](https://openreview.net/forum?id=WDVD4lUCTzU) for details).
We do not want to loose that property which is why we fix their weights (see the `trainable=False` property of the output above).
In order to give the model a little "brain muscle" to learn to understand the embedding, we add a dense layer before the classifier:

``` {.python}
sentiment_mdl_USE = tf.keras.layers.Dense(128, name="sentiment_mdl")(normalized)
classifier = tf.keras.layers.Dense(1, name="classifier")(sentiment_mdl_USE)

model = tf.keras.Model(text_input_USE, classifier)
model.compile(
  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
  optimizer=tf.keras.optimizers.Adam(),
  metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy')]
)
model.summary()
```

To recap, the complete model code looks like this now:
```{.python}
# taken and adapted from:
# https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-base-br/1
def normalization(embeds):
  norms = tf.norm(embeds, 2, axis=1, keepdims=True)
  return embeds/norms

text_input_USE = tf.keras.layers.Input(shape=(), dtype=tf.string, name="USE_input")
preprocessor = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2",
                              trainable=False,
                              name="USE_preprocessor")
encoder_inputs = preprocessor(text_input_USE)
encoder_USE = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-base-br/1",
                             trainable=False,
                             name="USE")
outputs = encoder_USE(encoder_inputs)["default"]
normalized = tf.keras.layers.Lambda(normalization)(outputs)
sentiment_mdl_USE = tf.keras.layers.Dense(128, name="sentiment_mdl")(normalized)
classifier = tf.keras.layers.Dense(1, name="classifier")(sentiment_mdl_USE)

model = tf.keras.Model(text_input_USE, classifier)

model.compile(
  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
  optimizer=tf.keras.optimizers.Adam(),
  metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy')]
)
model.summary()
```

The summary generates the following output:

>     Model: "model"
>     __________________________________________________________________________________________________
>     Layer (type)                    Output Shape         Param #     Connected to                     
>     ==================================================================================================
>     USE_input (InputLayer)          [(None,)]            0                                            
>     __________________________________________________________________________________________________
>     USE_preprocessor (KerasLayer)   {'input_word_ids': ( 0           USE_input[0][0]                  
>     __________________________________________________________________________________________________
>     USE (KerasLayer)                {'sequence_output':  470926849   USE_preprocessor[0][0]           
>                                                                      USE_preprocessor[0][1]           
>                                                                      USE_preprocessor[0][2]           
>     __________________________________________________________________________________________________
>     lambda (Lambda)                 (None, 768)          0           USE[0][0]                        
>     __________________________________________________________________________________________________
>     sentiment_mdl (Dense)           (None, 128)          98432       lambda[0][0]                     
>     __________________________________________________________________________________________________
>     classifier (Dense)              (None, 1)            129         sentiment_mdl[0][0]              
>     ==================================================================================================
>     Total params: 471,025,410
>     Trainable params: 98,561
>     Non-trainable params: 470,926,849
>     __________________________________________________________________________________________________

Now, we are ready for training:

``` {.python}
# train on yelp data
history = model.fit(
  train.shuffle(10000).batch(128).prefetch(tf.data.AUTOTUNE),
  validation_data=val.batch(128).prefetch(tf.data.AUTOTUNE),
  epochs=3,
  verbose=1
)
```
>     Epoch 1/3
>     3500/3500 [==============================] - 3164s 901ms/step - loss: 0.2803 - accuracy: 0.8722 - val_loss: 0.2399 - val_accuracy: 0.8938
>     Epoch 2/3
>     3500/3500 [==============================] - 3001s 858ms/step - loss: 0.2414 - accuracy: 0.8985 - val_loss: 0.2340 - val_accuracy: 0.9011
>     Epoch 3/3
>     3500/3500 [==============================] - 3000s 857ms/step - loss: 0.2388 - accuracy: 0.8998 - val_loss: 0.2343 - val_accuracy: 0.9053

**~90%** - with a very simple model.
That's something!
By looking at the accuracy in each epoch, one can see that after the second epoch, the model doesn't get much better anymore.
This is not surprising as we gave it just a small dense layer and the final classifier to train.
However, this simplicity also has the side effect that we can't really overfit: The model is too small to remember the entire training set.
Therefore, the validation accuracy closely matches the training accuracy (if it would overfit, validation accuracy would be considerably smaller than the training accuracy).
Nice!

Let's see what the evaluation on the test set yields:
``` {.python}
model.evaluate(test.batch(128))
```

>     297/297 [==============================] - 204s 687ms/step - loss: 0.2354 - accuracy: 0.9032

Good, similar values to what we saw during training!

# Results and Conclusion
Of course, there is a million things one could do to improve the model (label smoothing, larger models, other activations, learning rate schedule, ...) but the real question here is: Does it generalize to other languages well enough, such that we can classify the sentiment of non-English sentences?
Since the last layer of our model has no activation we apply a sigmoid function on its output.
This could also be done directly in the model, as in [this](https://www.tensorflow.org/tutorials/keras/text_classification#export_the_model) example but for the sake of brevity we just apply it inside a function:
```{.python}
def get_sentiment(text):
  res = tf.sigmoid(model.predict([text]))[0][0].numpy()
  return res, "positive" if res > 0.5 else "negative"
```

Let's test some phrases (German, French, and English phrases are grouped together):

#### True negatives
``` {.python}
get_sentiment("This really was a lousy restaurant.")
# (0.00044415292, 'negative')
get_sentiment("Das war wirklich ein mieses Restaurant.")
# (0.0001060787, 'negative')
get_sentiment("C'était vraiment un mauvais restaurant.")
# (8.612012e-06, 'negative')

get_sentiment("Extremely unfriendly. I'll never go again.")
# (0.00050263654, 'negative')
get_sentiment("Total unfreundlich. Gehe nie wider dahin.")
# (1.6988986e-05, 'negative')
get_sentiment("Très hostile. Je n'y retournerai plus jamais.")
# (0.00018812467, 'negative')

get_sentiment("Although the service was quite alright waiting for so long annoyed us quite a lot.")
# (0.030132184, 'negative')
get_sentiment("Obwohl die Bedienung eigentlich ganz nett war hat uns die lange Wartezeit sehr gestört.")
# (0.04019558, 'negative')
get_sentiment("Même si le service était tout à fait correct, attendre si longtemps nous a énervés beaucoup.")
# (0.014694035, 'negative')
```

#### True positives
``` {.python}
get_sentiment("Great service. I can recommend this place!")
# (0.9997924, 'positive')
get_sentiment("Großartiger service. Kann ich empfehlen!")
# (0.99997807, 'positive')
get_sentiment("Très bon service. Je peux recommander ce place!")
# (0.99982685, 'positive')

get_sentiment("The new branch is easy to reach.")
# (0.9349099, 'positive')
get_sentiment("Die neue Filiale ist gut zu erreichen.")
# (0.9374107, 'positive')
get_sentiment("La nouvelle succursale est facilement accessible.")
# (0.9890525, 'positive')
```

#### False negatives
``` {.python}
get_sentiment("We were given very good advice. We couldn't wish for more.")
# (0.23018356, 'negative')
get_sentiment("Wir wurden sehr gut beraten. Das hätten wir uns nicht besser wünschen können.")
# (0.19589308, 'negative')
get_sentiment("Nous avons été bien conseillés. Nous ne pouvions pas souhaiter plus.")
# (0.07804157, 'negative')

get_sentiment("We didn't have to wait long and the employee was able to help us.")
# (0.11634488, 'negative')
get_sentiment("Wir mussten nicht lange warten und der Mitarbeiter hat uns gleich weiterhelfen können.")
# (0.19271322, 'negative')
get_sentiment("Nous n'avons pas eu à attendre longtemps et l'employé a pu nous aider.")
# (0.10524392, 'negative')
```

### Discussion
As you can see, it works surprisingly well - not just for English but also for German and French (and presumably for many other languages as well).
Of course, it's not always right: The model seems to have issues with some positive sentences which it misclassifies as negative.
However, in both cases the English translation is a false negative as well.
This means that this misclassification is not a result of training the model with English sentences and then using it to classify non-English sentences.
Rather, this is more likely an issue either in the training data (we didn't see enough diversity during training) or in the model architecture: It may just be too small to grasp all the nuances.
In other words: This is good news because if we make the model perform better on English phrases, we might also solve those false negatives for other languages.

For a real evaluation, though, we'd need a large labeled non-English dataset whose non-existence is what made us use a language agnostic model in the first place.
But telling from our internal results, the model works indeed decently well for German, outperforming sentiment classification using simple polarity dictionaries.

# Next Steps
* Use Google Translate to translate an English sentiment dataset into German (at least this was done in the original [paper](https://arxiv.org/abs/1907.04307) for the first Universal Sentence Encoder model to augment the training data to different languages).
* Use a larger Model
* Train with multi-language [BERT](https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3) model, instead of Universal Sentence Encoder
* This method might also be applicable to other text classification domains but we haven't checked that so far
* ...
