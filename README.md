If not already done, install the Pixi package manager.
Instructions from https://pixi.sh/latest/installation/:
- Windows: `powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"`
- Linux: `curl -fsSL https://pixi.sh/install.sh | sh`

Once Pixi is installed, open this directory in a terminal, run `pixi run replicate-publication`, watch das blinkenlights, then take a look inside the `demo` directory.

By running Pixi at all, the dependencies will be updated and installed as needed.
This includes installing the correct version of Python and specific dependencies from specific sources depending on the operating system and the environment selected (ie: the `hpc` environment).

The `replicate-publication` task will recursively run the tasks it depends on, which includes `download-data`, `train`, and `eval`.

The end result of all this is the correct python environment being installed, the required data being downloaded, a ridiculously complex water level prediction model being trained with `jax`, and the model being run with `tensorflow` to generate reports (which could include evaluation metrics). Also worth noting that PyTorch was used during development to debug a problem with the model (it seems to be the slowest training backend, but its flexibility allows for easy debugging). This effectively solves my reoccuring pain-point of recreating environments on multiple machines and keeping things up-to-date, as well as Hector and Miranda's difficulties with distributing a codebase associated with a publication (the explicit RNG seed helps reproducibility here as well).

The model and training is quite over-engineered in the context of our lab, and has poor performance, but it is a demonstration that Keras 3 is definitely sufficient for anything we need.
The training process includes learning rate reduction, early stopping, L1 and L2 regularization, and a weighted blend of 4 custom loss functions (including the KL loss a VAE would need and the log-probability loss a PNN needs).
The model starts with a dropout layer, then moves on to several sequential dense layers, then splits into two separate variational sampling layers (which are key for a VAE), which are fed into distinct dense layers, which are joined in a probabilistic output layer.

There are [additional layers](https://keras.io/api/layers/) available in the Keras API that are outside the scope of what Brian frequently uses, and therefore this example.

This means that, in addition to MLPs, VAEs, and PNNs, it should also be possible to implement FogNet, FogNet 2.0, and Marina's seq2seq model.

There is also an adapted version of the preprocessing notebook that allows MLFlow results to be explored with our existing hyperparameter exploration visuals (pulled from [here](https://github.com/conrad-blucher-institute/vae-fog-hrrr/blob/new-vae/report_utils.py)).
This notebook can be accessed by running `pixi shell`, and then `jupyter-lab`.
