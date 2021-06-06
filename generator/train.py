import tensorflow as tf


def train(model, optimizer, loss_fn, loss_metric, X_train, y_train, epochs, batch_size, validation_data, mc=True):
    train_dataset = tf.data.Dataset.from_tensor_slices(X_train)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    X_val, y_val = validation_data
    val_dataset = tf.data.Dataset.from_tensor_slices(X_val)
    val_dataset = val_dataset.shuffle(buffer_size=1024).batch(batch_size)

    for epoch in range(epochs):
        print(f"Start of epoch {(epoch, )}")
        for step, x_batch_train in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                reconstructed = model(x_batch_train)
                # Compute reconstruction loss
                loss = loss_fn(x_batch_train, reconstructed)
                loss += sum(model.losses)  # Add KLD regularization loss

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            loss_metric(loss)
            if step % 100 == 0:
                print("step %d: mean loss = %.4f" %
                      (step, loss_metric.result()))
        for step, x_batch_val in enumerate(val_dataset):
            with tf.GradientTape() as tape:
                reconstructed = model(x_batch_val)
                # Compute reconstruction loss
                val_loss = loss_fn(x_batch_val, reconstructed)
                val_loss += sum(model.losses)  # Add KLD regularization loss

            loss_metric(val_loss)
        print(f"loss: {loss}\t\tval_loss: {val_loss}")
