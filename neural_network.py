from __future__ import print_function
from love_letter import generateTrainingData
import numpy as np
import os
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

input_size = 79
num_labels = 51

# With gradient descent training, even this much data is prohibitive.
# Subset the training data for faster turnaround.
train_subset = 1024


class Network :
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
          #initializes weights   
          self.weights = tf.Variable(tf.truncated_normal([input_size, num_labels]))
          self.biases = tf.Variable(tf.zeros([num_labels]))

    def train(self , train_dataset , train_labels , valid_dataset , valid_labels , test_dataset , test_labels) :
        batch_size = 64

        with self.graph.as_default():
            
          # Input data.
          # Load the training, validation and test data into constants that are
          # attached to the graph.


          tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, input_size))
          tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
          #tf_train_dataset = tf.constant(train_dataset)
          #tf_train_labels = tf.constant(train_labels)
          tf_valid_dataset = tf.constant(valid_dataset)
          tf_test_dataset = tf.constant(test_dataset)
          
          
          # Training computation.
          # We multiply the inputs with the weight matrix, and add biases. We compute
          # the softmax and cross-entropy (it's one operation in TensorFlow, because
          # it's very common, and it can be optimized). We take the average of this
          # cross-entropy across all training examples: that's our loss.
          logits = tf.matmul(tf_train_dataset, self.weights) + self.biases
          loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
          
          # Optimizer.
          # We are going to find the minimum of this loss using gradient descent.
          optimizer = tf.train.GradientDescentOptimizer(0.4).minimize(loss)
          
          # Predictions for the training, validation, and test data.
          # These are not part of training, but merely here so that we can report
          # accuracy figures as we train.
          train_prediction = tf.nn.softmax(logits)
          valid_prediction = tf.nn.softmax(
              tf.matmul(tf_valid_dataset, self.weights) + self.biases)
          test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, self.weights) + self.biases)
            
        num_steps = 3001
        with tf.Session(graph=self.graph) as session:
          # This is a one-time operation which ensures the parameters get initialized as
          # we described in the graph: random weights for the matrix, zeros for the
          # biases.
          opt = None
          l = None
          predictions = None
          tf.global_variables_initializer().run()
          print('Initialized')
          for step in range(num_steps):
            # Run the computations. We tell .run() that we want to run the optimizer,
            # and get the loss value and the training predictions returned as numpy
            # arrays.
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
            
            opt, l, predictions = session.run([optimizer, loss, train_prediction],feed_dict=feed_dict)
            if (step % 100 == 0):
              print('Minibatch Loss at step %d: %f' % (step, l))
              print('Minibatch accuracy: %.1f%%' % self.accuracy(
                predictions, batch_labels))
              print('Validation accuracy: %.1f%%' % self.accuracy(
                valid_prediction.eval(), valid_labels))
          print('Test accuracy: %.1f%%' % self.accuracy(test_prediction.eval(), test_labels))
          return opt , l , predictions
          


    def accuracy(sefl , predictions, labels):
      return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
              / predictions.shape[0])

    def reformat(dataset) :
        dataset = np.array(dataset)
        dataset = dataset.reshape((-1 , input_size)).astype(np.float32)
        return dataset

    def getLogits(self, inp) :
        inp = Network.reformat(inp)
        
        with self.graph.as_default():
            tf_data = tf.constant(inp)
            logit = tf.matmul(tf_data, self.weights) + self.biases

        with tf.Session(graph=self.graph) as session:
            tf.global_variables_initializer().run()
            array = session.run(logit)
        return array.tolist()[0]





train_dataset = None
train_labels = None
valid_dataset = None
valid_labels = None
test_dataset = None
test_labels = None

network = Network()
i = str(0)
data_root = '.' # Change me to store data elsewhere
pickle_file = os.path.join(data_root, 'weights_'+ i +'.pickle')
#repeat this process while the accuracy impoves
number_of_players = 4
try :
    with open(pickle_file, 'rb') as f:
      save = pickle.load(f)
      train_dataset = save['train_dataset']
      train_labels = save['train_labels']
      valid_dataset = save['valid_dataset']
      valid_labels = save['valid_labels']
      test_dataset = save['test_dataset']
      test_labels = save['test_labels']
      #network.weights = save['weights']
      #network.biases = save['biases']
      del save  # hint to help gc free up memory
#ele ainda não recupera os pesos, mas precisa recuperar      
except Exception as e :
    #tenta recuperar os dados, se não conseguir, então gera novos dados
    train_dataset , train_labels , valid_dataset ,valid_labels , test_dataset, test_labels = generateTrainingData(number_of_players , train_subset , network)

    train_dataset = Network.reformat(train_dataset)
    valid_dataset = Network.reformat(valid_dataset)
    test_dataset = Network.reformat(test_dataset)
    try:
        f = open(pickle_file, 'wb')
        save = {
            'train_dataset': train_dataset,
            'train_labels': train_labels,
            'valid_dataset': valid_dataset,
            'valid_labels': valid_labels,
            'test_dataset': test_dataset,
            'test_labels': test_labels,
            #'weights': network.weights,
            #'biases' : network.biases,
            }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise
train_labels = np.array(train_labels)
valid_labels = np.array(valid_labels)
test_labels = np.array(test_labels)

print(network.weights)
opt , l , predictions = network.train(train_dataset , train_labels , valid_dataset ,valid_labels , test_dataset, test_labels)
print(network.weights)
'''
train_file = os.path.join(data_root, 'train_'+ i +'.pickle')

try:
    f = open(train_file, 'wb')
      save = {
        'optimizer': opt,
        'loss': l,
        'predictions': predictions,    
        }
      pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
      f.close()

#construir uma mediada real para o quão bom o classificador é ( jogos contra oponente que joga aleatóriamente
'''







