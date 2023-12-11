from utils import *


input_layer = Input(shape=(232, 232, 1))
x = input_layer

for _ in range(3):
    conv = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Concatenate()([x, conv])
    x = MaxPooling2D(pool_size=(2, 2))(x)

x = Flatten()(x)

x = Dense(units=128, activation='relu')(x)
output_layer = Dense(units=3, activation='softmax')(x)
Dense_Model = Model(inputs=input_layer, outputs=output_layer)
Dense_Model.summary()

Dense_Model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

file_s = r'TestTraining/Dense_Model.h5'
checkpointer = ModelCheckpoint(file_s, monitor='loss', verbose=1, save_best_only=True, save_weights_only=False, save_freq='epoch')

callback_list=[checkpointer]


historyDense = Dense_Model.fit(
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

y_pred = Dense_Model.predict(test_generator)
y_true = test_generator.classes
y_pred_classes = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(y_true, y_pred_classes)
print(f'Test Accuracy: {accuracy * 100:.2f}%')
Dense_Model.save('Dense_Model.h5')
print("Dense Model Training Completed!!!")