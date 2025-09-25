from tensorflow import keras

def build_model(image_shape, vocab_size, num_answers):
    L = keras.layers
    Model = keras.Model

    # Image Processing
    # Use CNN for the vision part
    # Process the image input to extract features
    image_input = L.Input(image_shape)

    # Convolutional layers with max pooling and ReLU activation
    x1 = L.Conv2D(8, 3, padding='same')(image_input)
    x1 = L.MaxPooling2D()(x1)
    x1 = L.Activation("relu")(x1)

    x1 = L.Conv2D(16, 3, padding='same')(x1)
    x1 = L.MaxPooling2D()(x1)
    x1 = L.Activation("relu")(x1)

    x1 = L.Dense(32, activation='tanh')(x1)
    x1 = L.Reshape((-1, 64))(x1)

    # NLP Part: Queston Processing
    question_input = L.Input(shape=(vocab_size,))

    # Dense layers with tanh activation
    x2 = L.Dense(64, activation='tanh')(question_input)
    x2 = L.Dense(64, activation='tanh')(x2)
    x2 = L.Reshape((1, 64))(x2)

    # Attention: question attends over image regions
    attended = L.MultiHeadAttention(num_heads=2, key_dim=64)(x2, x1, x1)
    attended = L.Flatten()(attended)

    # Fusion: combine attended vision features + question vector
    fusion = L.Concatenate()([attended, L.Flatten()(x2)])

    # Dense layers to combine features and generate predictions
    out = L.Dense(32, activation='tanh')(fusion)
    out = L.Dense(num_answers, activation='softmax')(out)

    # The model takes two inputs: image features, question features
    return Model(inputs=[image_input,question_input], outputs=out)
