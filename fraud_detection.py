import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
import os
from datetime import datetime 
from sklearn.metrics import roc_auc_score as auc 
import seaborn as sns


df = pd.read_csv('creditcard.csv')


#for data visualization of dataset mostly sab kam aayega:
print("\n\n\n\n\n\n\n\Data Visualization begins now for beginners to understand the dataset better\n\n\n\n\n")
print(df.shape)#tocheck the shape of dataset ie. number of rows and columns
print(df.head)#basically prints the whole dataset
print(df.columns)#shows all the names of coulmns
print(df.dtypes)#shows data type (of all entities in columns)
print(df.corr())#shows correlation between data

#using scatter to plot graph of two entities of dataset
#here i plotted graph of column Time and Amount
# create a figure and axis
fig, ax = plt.subplots()
# scatter the sepal_length against the sepal_width
ax.scatter(df['Time'], df['Amount'])
# set a title and labels
ax.set_title('creditcard fraud dataset visualization. Close this graph to proceed forward and do same for all the next vsualizations')
ax.set_xlabel('Time')
ax.set_ylabel('Amount')
plt.show()


# create figure and axis
fig, ax = plt.subplots()
# plot histogram
ax.hist(df['Time'])
# set title and labels
ax.set_title('visualization')
ax.set_xlabel('Time')
ax.set_ylabel('Frequency')
plt.show()

#for histogram
plt.figure(figsize=(12,5*4))
gs = gridspec.GridSpec(5, 1)
for i, cn in enumerate(df.columns[0:1]):#o:1 0th column ka print karega as upperbound is neglected in syntax
    ax = plt.subplot(gs[i])
    sns.distplot(df[cn][df.Class == 1], bins=50)
    sns.distplot(df[cn][df.Class == 0], bins=50)
    ax.set_xlabel('Time')
    ax.set_title('histogram of feature: ' + str(cn))
plt.show()


TEST_RATIO = 0.25
df.sort_values('Time', inplace = True)
TRA_INDEX = int((1-TEST_RATIO) * df.shape[0])#due to this only 75% of the dataset will be used for training and we take 75% of dataset in TRA_INDEX
#input values 1 to 29th column me hai and output 31st me.30th column ka kuch kaam nahi hai and usse overfitting horahai

train_x = df.iloc[:TRA_INDEX, 1:30].values #we write 1:30 to take input from 1st column to 29th column as upperbound is neglected in python syntax.so 30th column is ignored here,or you could also write
#train_x = df.iloc[:TRA_INDEX, 1:-2] #this takes all columns as input except two columns from last ie-it takes 1 to 29 columns numbers
train_y = df.iloc[:TRA_INDEX, -1].values #takes only last column as input ie- 31st column(last column)

test_x = df.iloc[TRA_INDEX:, 1:30].values
test_y = df.iloc[TRA_INDEX:, -1].values  




print("Total train examples: {}, total fraud cases: {}, equal to {:.5f} of total cases. ".format(train_x.shape[0], np.sum(train_y), np.sum(train_y)/train_x.shape[0]))

print("Total test examples: {}, total fraud cases: {}, equal to {:.5f} of total cases. ".format(test_x.shape[0], np.sum(test_y), np.sum(test_y)/test_y.shape[0]))


#normalization for tanh activation
cols_mean = []
cols_std = []
for c in range(train_x.shape[1]):
    cols_mean.append(train_x[:,c].mean())
    cols_std.append(train_x[:,c].std())
    train_x[:, c] = (train_x[:, c] - cols_mean[-1]) / cols_std[-1]
    test_x[:, c] =  (test_x[:, c] - cols_mean[-1]) / cols_std[-1]

    
    
learning_rate = 0.001
training_epochs = 10
batch_size = 256
display_step = 1

# Network Parameters
n_hidden_1 = 15 # 1st layer num features
#n_hidden_2 = 15 # 2nd layer num features
n_input = train_x.shape[1] # MNIST data input (img shape: 28*28)
data_dir = '.'    

X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    #'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
    #'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    #'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_input])),
    #'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    #layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   #biases['encoder_b2']))
    return layer_1


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    #layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                  # biases['decoder_b2']))
    return layer_1

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define batch mse
batch_mse = tf.reduce_mean(tf.pow(y_true - y_pred, 2), 1)

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# TRAIN StARTS
save_model = os.path.join(data_dir, 'temp_saved_model_1layer.ckpt')
saver = tf.train.Saver()

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    now = datetime.now()
    sess.run(init)
    total_batch = int(train_x.shape[0]/batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_idx = np.random.choice(train_x.shape[0], batch_size)
            batch_xs = train_x[batch_idx]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
            
        # Display logs per epoch step
        if epoch % display_step == 0:
            train_batch_mse = sess.run(batch_mse, feed_dict={X: train_x})
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.4f}".format(c), 
                  "Train auc=", "{:.4f}".format(auc(train_y, train_batch_mse)), 
                  "Time elapsed=", "{}".format(datetime.now() - now))

    print("\nOptimization Finished!\n")
    print("FINAL TRAINING ACCURACY ACHIEVED: ", "{:.4f}".format(auc(train_y, train_batch_mse)), )
    
    save_path = saver.save(sess, save_model)
    print("Model saved in file: %s" % save_path)
    


