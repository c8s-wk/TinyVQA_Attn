from tensorflow.keras import layers as L
from tensorflow.keras import Model

# Image Processing
# Use CNN for the vision part
# Process the image input to extract features
image_input = L.Input(image_shape)

# Convolutional layers with max pooling and ReLU activation
x1 = L.Conv2D(8, 3, padding='same')(image_input)
x1 = L.MaxPooling2D()(x1)
x1 = L.Activation("relu")(x1)

x1 = L.Conv2D(16, 3, padding='same')(image_input)
x1 = L.MaxPooling2D()(x1)
x1 = L.Activation("relu")(x1)

# Flatten the feature map and apply a Dense layer
x1 = L.Flatten()(x1)
x1 = L.Dense(32, activation='tanh')(x1)

# NLP Part: Queston Processing
question_input = L.Input(shape=(vocab_size,))

# Dense layers with tanh activation
x2 = L.Dense(32, activation='tanh')(question_input)
x2 = L.Dense(32, activation='tanh')(x2)

# Merge Vision and NLP Parts
out = L.Multiply()([x1, x2])

# Dense layers to combine features and generate predictions
out = L.Dense(32, activation='tanh')(out)
out = L.Dense(num_answers, activation='softmax')(out)

# The model takes two inputs: image features, question features
model = Model(inputs=[image_input,question_input], outputs=out)