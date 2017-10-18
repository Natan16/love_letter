from __future__ import print_function
from love_letter import generateTrainingData
import numpy as np
import os
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
from random import shuffle 


data_root = '.' # Change me to store data elsewhere
parameters_root = '.' # Change me to store data elsewhere


pickle_file = os.path.join(data_root, 'data.pickle')
par_file = os.path.join(parameters_root, 'par_before_training.pickle' )
par_file_out = os.path.join(parameters_root, 'par_after_training.pickle' )

input_size = 79
num_labels = 51

# With gradient descent training, even this much data is prohibitive.
# Subset the training data for faster turnaround.
train_subset = 1024



class Network :
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            
            fc1_size = 4096 
            fc2_size = 2048 
            fc3_size = 128

            #initializes weights   
            self.W1 = tf.Variable(tf.truncated_normal([input_size, fc1_size]) , stddev=np.sqrt(2.0 / (input_size))))
            self.b1 = tf.Variable(tf.zeros([fc1_size]))

            self.W2 = tf.Variable(tf.truncated_normal([input_size, fc2_size])stddev=np.sqrt(2.0 / (fc1_size))))
            self.b2 = tf.Variable(tf.zeros([fc2_size]))

            self.W3 = tf.Variable(tf.truncated_normal([input_size, fc3_size])stddev=np.sqrt(2.0 / (fc2_size))))
            self.b3 = tf.Variable(tf.zeros([fc3_size]))

            self.W4 = tf.Variable(tf.truncated_normal([input_size, num_labels])stddev=np.sqrt(2.0 / (fc3_size))))
            self.b4 = tf.Variable(tf.zeros([num_labels]))
            
    #4 LAYERS WITH DROUPOUT
    def train(self , train_dataset , train_labels , valid_dataset , valid_labels , test_dataset , test_labels) :
        batch_size = 128
 
        W1 = self.W1
        b1 = self.b1

        W2 = self.W2
        b2 = self.b2

        W3 = self.W3
        b3 = self.b3
        
        W4 = self.W4
        b4 =self.b4
        
        graph = self.graph()
        with graph.as_default():
            # Input data. For the training data, we use a placeholder that will be fed
            # at run time with a training minibatch.
            tf_train_dataset = tf.placeholder(tf.float32,
                                              shape=(batch_size, image_size * image_size))
            tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
            tf_valid_dataset = tf.constant(valid_dataset)
            tf_test_dataset = tf.constant(test_dataset)
            tf_beta = tf.placeholder(tf.float32)
            global_step = tf.Variable(0)  # count the number of steps taken.

            # Training computation.
            y1 = tf.nn.relu(tf.matmul(tf_train_dataset, W1) + b1)
            y1 = tf.nn.dropout(y1, 0.5)

            y2 = tf.nn.relu(tf.matmul(y1, W2) + b2)
            y2 = tf.nn.dropout(y2, 0.5)

            y3 = tf.nn.relu(tf.matmul(y2, W3) + b3)
            y3 = tf.nn.dropout(y3, 0.5)

            logits = tf.matmul(y3, W4) + b4

            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

            loss = loss + tf_beta * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(b1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(b2) +
                                     tf.nn.l2_loss(W3) + tf.nn.l2_loss(b3) + tf.nn.l2_loss(W4) + tf.nn.l2_loss(b4))

            # Optimizer
            learning_rate = tf.train.exponential_decay(0.5, global_step, 1000, 0.7, staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

            # Predictions for the training, validation, and test data.
            train_prediction = tf.nn.softmax(logits)

            y1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, W1) + b1)
            y2_valid = tf.nn.relu(tf.matmul(y1_valid, W2) + b2)
            y3_valid = tf.nn.relu(tf.matmul(y2_valid, W3) + b3)
            valid_logits = tf.matmul(y3_valid, W4) + b4
            valid_prediction = tf.nn.softmax(valid_logits)

            y1_test = tf.nn.relu(tf.matmul(tf_test_dataset, W1) + b1)
            y2_test = tf.nn.relu(tf.matmul(y1_test, W2) + b2)
            y3_test = tf.nn.relu(tf.matmul(y2_test, W3) + b3)
            test_logits = tf.matmul(y3_test, W4) + b4
            test_prediction = tf.nn.softmax(test_logits)

        # Let's run it:
        num_steps = 12001

        with tf.Session(graph=graph) as session:

            tf.initialize_all_variables().run()

            w1 , v1 , w2 , v2 , w3 , v3 , w4 ,v4 = session.run(
                [self.W1, self.b1, self.W2 , self.b2 , self.W3 , self.b3 , self.W4 , self.b4])      
            try:
                f = open(par_file, 'wb')
                save = {
                    'W1': w1,
                    'b1': v1,
                    'W1': w2,
                    'b1': v2,
                    'W1': w3,
                    'b1': v3,
                    'W1': w4,
                    'b1': v4,
                    }
                pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
                f.close()

            
            print("Initialized")
            for step in range(num_steps):
                # Pick an offset within the training data, which has been randomized.
                # Note: we could use better randomization across epochs.
                offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
                # Generate a minibatch.
                batch_data = train_dataset[offset:(offset + batch_size), :]
                batch_labels = train_labels[offset:(offset + batch_size), :]
                # Prepare a dictionary telling the session where to feed the minibatch.
                # The key of the dictionary is the placeholder node of the graph to be fed,
                # and the value is the numpy array to feed to it.
                feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, tf_beta: 0.001438}
                _, l, predictions = session.run(
                    [optimizer, loss, train_prediction], feed_dict=feed_dict)
                if (step % 500 == 0):
                    print("Minibatch loss at step %d: %f" % (step, l))
                    print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                    print("Validation accuracy: %.1f%%" % accuracy(
                        valid_prediction.eval(), valid_labels))
            #print("Final Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
            w1 , v1 , w2 , v2 , w3 , v3 , w4 ,v4 = session.run(
                [self.W1, self.b1, self.W2 , self.b2 , self.W3 , self.b3 , self.W4 , self.b4])      
            try:
                f = open(par_file_out, 'wb')
                save = {
                    'W1': w1,
                    'b1': v1,
                    'W1': w2,
                    'b1': v2,
                    'W1': w3,
                    'b1': v3,
                    'W1': w4,
                    'b1': v4,
                    }
                pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
                f.close()
                
            except Exception as e:
                print('Unable to save data to', par_file, ':', e)
                raise

'''
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
          return opt , l , predictions'''
          


    def accuracy(sefl , predictions, labels):
      return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
              / predictions.shape[0])

    def reformat(dataset) :
        dataset = np.array(dataset)
        dataset = dataset.reshape((-1 , input_size)).astype(np.float32)
        return dataset
    #this method will be essencial List of inputs -> List of outputs
    #given paramenters of neural network
    def getLogits(self, inp) :
        inp = Network.reformat(inp)
        
        with self.graph.as_default():
            tf_data = tf.constant(inp)
            logit = tf.matmul(tf_data, self.weights) + self.biases

        with tf.Session(graph=self.graph) as session:
            tf.global_variables_initializer().run()
            array = session.run(logit)
        return array.tolist()[0]

    #def saveParameters(self):
            
    #def loadParameters(fileName):


train_dataset = []
train_labels = []
valid_dataset = []
valid_labels = []
test_dataset = []
test_labels = []

network = Network()
#recover the weights and biases
try :
    with open(par_file, 'rb') as f :
        save = pickle.load(f)
        with self.graph.as_default(save):
            self.W1 = tf.Variable(save['W1'])
            self.b1 = tf.Variable(save['b1'])

            self.W2 = tf.Variable(save['W2'])
            self.b2 = tf.Variable(save['b2'])

            self.W3 = tf.Variable(save['W3'])
            self.b3 = tf.Variable(save['b3'])

            self.W4 = tf.Variable(save['W4'])
            self.b4 = tf.Variable(save['b4'])
        del save
        
except Exception as e :
    print("random weights and biases generated")


    
try :
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save  # hint to help gc free up memory

#ele ainda não recupera os pesos, mas precisa recuperar      
except Exception as e :
    #tenta recuperar os dados, se não conseguir, então gera novos dados
    for number_of_players in [2,3,4]:
        train_dataset_aux , train_labels_aux , valid_dataset_aux ,valid_labels_aux , test_dataset_aux, test_labels_aux =
        generateTrainingData(number_of_players , train_subset , network)
        train_dataset += train_dataset_aux
        train_labels += train_labels_aux
        valid_dataset += valid_dataset_aux
        valid_labels += valid_labels_aux
        test_dataset += test_dataset_aux
        test_labels += test_labels_aux

    shuffle(train_dataset)
    shuffle(valid_dataset)
    shuffle(test_dataset)
    
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
            }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()

        
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise
    
#train_labels = np.array(train_labels)
#valid_labels = np.array(valid_labels)
#test_labels = np.array(test_labels)

network.train(train_dataset , train_labels , valid_dataset ,valid_labels , test_dataset, test_labels)








