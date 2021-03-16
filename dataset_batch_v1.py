    
# I was using like this.
# It have a problem
# It doesn't work when inputdata is huge
# with error below
# ValueError: Cannot create a tensor proto whose content is larger than 2GB.

## I need find other way

def dataset_batch(input, output):
    dataset = tf.data.Dataset.from_tensor_slices((input, output))
    dataset = dataset.shuffle(total_size).repeat().batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    input_stacked, output_stacked = iterator.get_next()
    return iterator, input_stacked, output_stacked


with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epoch_size):
            sess.run(iterator.initializer)
            total_batch = int(total_size / batch_size)
            for i in range(total_batch):
                train_input_batch, train_output_batch = sess.run([train_input_stacked, train_output_stacked])
                _, curr_W, curr_b, current_loss = sess.run([train_step, W, b, loss], feed_dict={x:train_input_batch, y:train_output_batch})