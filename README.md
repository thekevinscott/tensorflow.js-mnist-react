A port of the Tensorflow.js MNIST example in React.

A live version of this code is hosted here: https://thekevinscott.github.io/tensorflow.js-mnist-react/

# TensorFlow.js Example: Training MNIST

This example shows you how to train MNIST (using the layers API).

You can check out the tutorial that accompanies this example [here](https://js.tensorflow.org/tutorials/mnist.html).

This model will compute accuracy over 1000 random test set examples every 5
steps, plotting loss and accuracy as the model is training. Training time can
be reduced by computing accuracy over fewer examples less often.

Note: currently the entire dataset of MNIST images is stored in a PNG image we have
sprited, and the code in `data.js` is responsible for converting it into
`Tensor`s. This will become much simpler in the near future.

[See this example live!](https://storage.googleapis.com/tfjs-examples/mnist/dist/index.html)
