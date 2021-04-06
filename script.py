import os
import datetime
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

import tensorflow_text
import tensorflow_hub as hub
import tensorflow_datasets as tfds


train, val, test = tfds.load(name="yelp_polarity_reviews",
                             split=('train[:80%]', 'train[80%:]', 'test'),
                             as_supervised=True,
                             shuffle_files=True)

train = train.cache()
val = val.cache()
test = test.cache()

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

# train on yelp data
history = model.fit(
  train.shuffle(10000).batch(128).prefetch(tf.data.AUTOTUNE),
  validation_data=val.batch(128).prefetch(tf.data.AUTOTUNE),
  epochs=3,
  verbose=1
)
model.save("sentiments", overwrite=True)

model.evaluate(test.batch(128))

def get_sentiment(text):
  res = tf.sigmoid(model.predict([text]))[0][0].numpy()
  return res, "positive" if res > 0.5 else "negative"


# True negatives
get_sentiment("This really was a lousy restaurant.")
get_sentiment("Das war wirklich ein mieses Restaurant.")
get_sentiment("C'était vraiment un mauvais restaurant.")

get_sentiment("Extremely unfriendly. I'll never go again.")
get_sentiment("Total unfreundlich. Gehe nie wider dahin.")
get_sentiment("Très hostile. Je n'y retournerai plus jamais.")

get_sentiment("Although the service was quite alright waiting for so long annoyed us quite a lot.")
get_sentiment("Obwohl die Bedienung eigentlich ganz nett war hat uns die lange Wartezeit sehr gestört.")
get_sentiment("Même si le service était tout à fait correct, attendre si longtemps nous a énervés beaucoup.")


# True positives
get_sentiment("Great service. I can recommend this place!")
get_sentiment("Großartiger service. Kann ich empfehlen!")
get_sentiment("Très bon service. Je peux recommander ce place!")

get_sentiment("The new branch is easy to reach.")
get_sentiment("Die neue Filiale ist gut zu erreichen.")
get_sentiment("La nouvelle succursale est facilement accessible.")


# False negatives
get_sentiment("We were given very good advice. We couldn't wish for more.")
get_sentiment("Wir wurden sehr gut beraten. Das hätten wir uns nicht besser wünschen können.")
get_sentiment("Nous avons été bien conseillés. Nous ne pouvions pas souhaiter plus.")

get_sentiment("We didn't have to wait long and the employee was able to help us.")
get_sentiment("Wir mussten nicht lange warten und der Mitarbeiter hat uns gleich weiterhelfen können.")
get_sentiment("Nous n'avons pas eu à attendre longtemps et l'employé a pu nous aider.")
