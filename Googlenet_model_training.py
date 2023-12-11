from utils import *


def InceptionModule(x, filters):
    conv1x1 = Conv2D(filters[0], (1, 1), padding='same', activation='relu')(x)
    conv3x3 = Conv2D(filters[1], (1, 1), padding='same', activation='relu')(x)
    conv3x3 = Conv2D(filters[2], (3, 3), padding='same', activation='relu')(conv3x3)
    conv5x5 = Conv2D(filters[3], (1, 1), padding='same', activation='relu')(x)
    conv5x5 = Conv2D(filters[4], (5, 5), padding='same', activation='relu')(conv5x5)
    maxpool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    maxpool = Conv2D(filters[5], (1, 1), padding='same', activation='relu')(maxpool)
    inception = Concatenate(axis=-1)([conv1x1, conv3x3, conv5x5, maxpool])
    return inception

input_layer = Input(shape=(232, 232, 1))

x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

x = InceptionModule(x, [64, 128, 128, 32, 32, 32])
x = InceptionModule(x, [128, 192, 192, 96, 96, 96])
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

x = Flatten()(x)
x = Dense(units=128, activation='relu')(x)
output_layer = Dense(units=3, activation='softmax')(x)

Googlenet_model = Model(inputs=input_layer, outputs=output_layer)

Googlenet_model.summary()

Googlenet_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

fle_s=r'TestTraining/Googlenet_Model.h5'
checkpointer = ModelCheckpoint(fle_s, monitor='loss',verbose=1,save_best_only=True,
                               save_weights_only=False, mode='auto',save_freq='epoch')

callback_list=[checkpointer]


historyGooglenet = Googlenet_model.fit(
        train_generator,
    steps_per_epoch=len(train_generator)//8,
    batch_size=128,
    validation_data=validation_generator,
    validation_steps=len(validation_generator.classes),
epochs=epochs,
    shuffle=True,
    callbacks=[callback_list])

test_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(232, 232),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical',
    shuffle=False
)

y_pred = Googlenet_model.predict(test_generator)
y_true = test_generator.classes
y_pred_classes = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(y_true, y_pred_classes)
print(f'Test Accuracy: {accuracy * 100:.2f}%')
Googlenet_model.save('Googlenet_Model.h5')
print("Googlenet Model Training Completed!!!")