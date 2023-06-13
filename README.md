# Surface roughness prediction CNN

CNN for predicting surface roughness after milling machining process

## Starting out

ctrl + shift + P -> create env, choose interpreter (latest python version)

## Install libs

py -m pip install numpy

# First moderately successful model

Overfitting is possible, for although val_accuracy=1.00, the model sometimes fails to predict correctly on new images

[INFO] ... [127/127] images were preprocessed successfully
[INFO] Number of generated images: 1016
[INFO] Number of samples used for training ... 812
[INFO] Number of samples used for validation ... 204
25/25 [==============================] - 1364s 51s/step - loss: 1162.7634 - accuracy: 0.3573 - val_loss: 1.0552 - val_accuracy: 0.4706
[INFO] Current validation accuracy: 0.47058823704719543
Epoch 1/2
25/25 [==============================] - 656s 26s/step - loss: 1.1152 - accuracy: 0.4459 - val_loss: 0.9554 - val_accuracy: 0.5392
Epoch 2/2
25/25 [==============================] - 1276s 51s/step - loss: 0.9498 - accuracy: 0.5787 - val_loss: 0.7493 - val_accuracy: 0.8284
[INFO] Current validation accuracy: 0.8284313678741455
Epoch 1/3
25/25 [==============================] - 2660s 106s/step - loss: 0.5641 - accuracy: 0.8159 - val_loss: 0.3101 - val_accuracy: 0.9314
Epoch 2/3
25/25 [==============================] - 2721s 108s/step - loss: 0.1916 - accuracy: 0.9360 - val_loss: 0.0327 - val_accuracy: 1.0000
Epoch 3/3
25/25 [==============================] - 2640s 105s/step - loss: 0.0430 - accuracy: 0.9941 - val_loss: 0.0168 - val_accuracy: 1.0000
[INFO] Current validation accuracy: 1.0
[INFO] ... Saving surface roughness prediction model ...
[INFO] The surface roughness prediction model was built SUCCESSFULLY
[INFO] Run predictRoughness to test the model

# Second moderately successful model (roughness_classifier5)

If both accuracy and validation accuracy are above 80% and the loss and validation loss are below 1, it generally indicates that your model is performing well on the task of predicting surface roughness based on images.

An accuracy above 80% suggests that your model is making correct predictions for a significant portion of the dataset, which is a positive outcome. Similarly, having both loss and validation loss below 1 indicates that your model's predictions are relatively close to the ground truth values, as the loss represents the discrepancy between predicted and actual values.

However, it's important to consider the specific context and requirements of your application. The definition of a "good" accuracy or loss depends on the problem at hand and the level of precision needed in your predictions. In some scenarios, an accuracy of 80% might be acceptable, while in others, higher levels of accuracy may be required.

Since you're predicting surface roughness based on images, it's crucial to evaluate the model's performance on unseen data and consider other metrics or factors that might be relevant for your specific use case. Additionally, you may want to validate the model's performance on a separate test dataset to ensure that it generalizes well and is not overfitting to the training or validation data.

In summary, achieving an accuracy above 80%, having loss and validation loss below 1, and considering the context of surface roughness prediction from images are positive indications, but further evaluation and validation are recommended to ensure the model's effectiveness in practical scenarios.

[INFO] Number of loaded images: 254  
Images saved successfully.
[INFO] Number of generated images: 6096
[INFO] Splitting data into train and validation sets...
[INFO] Number of samples used for training ... 4876
[INFO] Number of samples used for validation ... 1220
[INFO] Initialising the model training...
Epoch 1/100
153/153 [==============================] - 182s 1s/step - loss: 8.7538 - accuracy: 0.4231 - val_loss: 3.6414 - val_accuracy: 0.4033
Epoch 2/100
153/153 [==============================] - 181s 1s/step - loss: 2.5569 - accuracy: 0.4969 - val_loss: 5.3112 - val_accuracy: 0.1811
Epoch 3/100
153/153 [==============================] - 178s 1s/step - loss: 1.8800 - accuracy: 0.5548 - val_loss: 2.5127 - val_accuracy: 0.4951
Epoch 4/100
153/153 [==============================] - 176s 1s/step - loss: 1.3119 - accuracy: 0.6458 - val_loss: 1.1386 - val_accuracy: 0.6516
Epoch 5/100
153/153 [==============================] - 176s 1s/step - loss: 0.9658 - accuracy: 0.6963 - val_loss: 1.7495 - val_accuracy: 0.6402
Epoch 6/100
153/153 [==============================] - 175s 1s/step - loss: 0.7730 - accuracy: 0.7533 - val_loss: 1.3607 - val_accuracy: 0.7164
Epoch 7/100
153/153 [==============================] - 175s 1s/step - loss: 0.6817 - accuracy: 0.7769 - val_loss: 1.0287 - val_accuracy: 0.7402
Epoch 8/100
153/153 [==============================] - 177s 1s/step - loss: 0.4855 - accuracy: 0.8310 - val_loss: 1.0270 - val_accuracy: 0.7869
Epoch 9/100
153/153 [==============================] - 175s 1s/step - loss: 0.4466 - accuracy: 0.8445 - val_loss: 0.8610 - val_accuracy: 0.8041
Epoch 10/100
153/153 [==============================] - 177s 1s/step - loss: 0.3455 - accuracy: 0.8753 - val_loss: 0.9185 - val_accuracy: 0.8254
Epoch 11/100
153/153 [==============================] - 175s 1s/step - loss: 0.3491 - accuracy: 0.8786 - val_loss: 1.0509 - val_accuracy: 0.8041
Epoch 12/100
153/153 [==============================] - 185s 1s/step - loss: 0.2968 - accuracy: 0.8999 - val_loss: 0.7661 - val_accuracy: 0.8369
Epoch 13/100
153/153 [==============================] - 178s 1s/step - loss: 0.2227 - accuracy: 0.9188 - val_loss: 0.8417 - val_accuracy: 0.8525
Epoch 14/100
153/153 [==============================] - 176s 1s/step - loss: 0.1868 - accuracy: 0.9374 - val_loss: 0.9051 - val_accuracy: 0.8533
Epoch 15/100
153/153 [==============================] - ETA: 0s - loss: 0.1814 - accuracy: 0.9354Restoring model weights from the end of the best epoch: 12.
153/153 [==============================] - 176s 1s/step - loss: 0.1814 - accuracy: 0.9354 - val_loss: 0.8084 - val_accuracy: 0.8648
Epoch 15: early stopping
Model: "sequential"

---

# Layer (type) Output Shape Param

conv2d (Conv2D) (None, 339, 339, 16) 160

batch_normalization (BatchN (None, 339, 339, 16) 64
ormalization)

max_pooling2d (MaxPooling2D (None, 169, 169, 16) 0
)

dropout (Dropout) (None, 169, 169, 16) 0

conv2d_1 (Conv2D) (None, 167, 167, 32) 4640

batch_normalization_1 (Batc (None, 167, 167, 32) 128
hNormalization)

max_pooling2d_1 (MaxPooling (None, 83, 83, 32) 0
2D)

dropout_1 (Dropout) (None, 83, 83, 32) 0

conv2d_2 (Conv2D) (None, 81, 81, 64) 18496

batch_normalization_2 (Batc (None, 81, 81, 64) 256
hNormalization)

max_pooling2d_2 (MaxPooling (None, 40, 40, 64) 0
2D)

dropout_2 (Dropout) (None, 40, 40, 64) 0

flatten (Flatten) (None, 102400) 0

dense (Dense) (None, 3) 307203

=================================================================
Total params: 330,947
Trainable params: 330,723
Non-trainable params: 224

---

None
[INFO] ... Saving surface roughness prediction model ...
[INFO] The surface roughness prediction model was built SUCCESSFULLY
[INFO] Plotting training statistics...
[INFO] Plot completed...
[INFO] Run predictRoughness.py to test the model
