import dataset
import model
import keras

from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('model_name', type=str)
    parser.add_argument('epochs', type=int)
    parser.add_argument('num_neurons', type=int)
    parser.add_argument('seed', type=int)
    args = parser.parse_args()

    keras.utils.set_random_seed(args.seed)

    ts, xs, ys = dataset.mk_txy(dataset.fetch_training_data())


    #ann = model.mk_ann([args.num_neurons])
    ann = model.mk_fancy(input_size=xs.shape[1],layer_sizes=[args.num_neurons for _ in range(3)])

    initial_learning_rate = 0.01
    lr_schedule = keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate, decay_steps=100, end_learning_rate=0.0001
    )

    ann.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=initial_learning_rate),
        #loss='mean_squared_error',
        #loss=model.GaussianProbabalisticLoss(),
        loss=model.MultiLoss(losses=[
            model.GaussianProbabalisticLoss(),
            model.LossOnGaussian(keras.losses.MeanSquaredError()),
            model.LambdaLoss(lambda y_true, y_pred: keras.ops.mean(y_pred[:,1]))
        ], weights=[w/10 for w in [10, 4, 6]]),
        #metrics=[
        #    model.MetricOnGaussian(keras.metrics.MeanSquaredError(), name='MSE'),
        #]
    )

    print(ann.summary())

    ann.fit(
        x=xs,
        y=ys,
        validation_split=0.7,
        epochs=args.epochs,
        sample_weight=(ys[:,0] < 12)*5+1,
        validation_freq=5,
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                start_from_epoch=20,
                restore_best_weights=True,
                verbose=1,
            ),
        ]
    )

    ann.save(args.model_name)

