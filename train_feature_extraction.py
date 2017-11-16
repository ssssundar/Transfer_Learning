import pickle
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from alexnet import AlexNet

# TODO: Load traffic signs data.
sign_data_file = './train.p'
with open(sign_data_file, mode='rb') as f:
    sign_data = pickle.load(f)
	
# TODO: Split data into training and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split(sign_data['features'], sign_data['labels'], test_size=0.33, random_state=36)
print("Shapes of X_train, X_valid, y_train, y_valid", X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)

# TODO: Define placeholders and resize operation.
#sign_names = pd.read_csv('signnames.csv')
nb_classes = 43
EPOCHS = 10
BATCH_SIZE = 128
rate = 0.001

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(x, (227, 227))
y = tf.placeholder(tf.int64, None)

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix

#fc8 Fully Connected layer added by Sundar. 11/15/17
#fc(43, relu=false, name='fc8')
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

#prob
#softmax(name='probs')
#probs = tf.nn.softmax(logits)


sess = tf.Session()

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation, var_list=[fc8W, fc8b])
init = tf.global_variables_initializer()

correct_prediction = tf.argmax(logits, 1)
accuracy_operation = tf.reduce_mean(tf.cast(tf.equal(correct_prediction, y), tf.float32))
#saver = tf.train.Saver()

# This function evaulates the acuuracy of Network model.

def evaluate(X_data, y_data, sess):
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss = 0
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        loss, accuracy = sess.run([loss_operation, accuracy_operation], feed_dict={x: batch_x, y: batch_y})
        total_loss += (loss * len(batch_x))
        total_accuracy += (accuracy * len(batch_x))
        return total_loss/num_examples, total_accuracy / num_examples
		
		
# TODO: Train and evaluate the feature extraction model.
with tf.Session() as sess:
    sess.run(init)
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        t0 = time.time()
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        #training_accuracy = evaluate(X_train, y_train)    
        validation_loss, validation_accuracy = evaluate(X_valid, y_valid, sess)
        print("EPOCH {} ...".format(i+1))
        #print("Training Accuracy = {:.3f}".format(training_accuracy))
        print("Time: %.3f seconds" % (time.time() - t0))
        print("Validation Loss = {:.3f}".format(validation_loss))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    #saver.save(sess, './my-model-graph')
    #print("Model saved")




















