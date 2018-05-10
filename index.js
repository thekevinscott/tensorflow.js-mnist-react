/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';
import embed from 'vega-embed';
import React, { Component } from 'react';
import ReactDOM from 'react-dom';
import PropTypes from 'prop-types';

import {MnistData} from './data';

const model = tf.sequential();

model.add(tf.layers.conv2d({
  inputShape: [28, 28, 1],
  kernelSize: 5,
  filters: 8,
  strides: 1,
  activation: 'relu',
  kernelInitializer: 'varianceScaling'
}));
model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
model.add(tf.layers.conv2d({
  kernelSize: 5,
  filters: 16,
  strides: 1,
  activation: 'relu',
  kernelInitializer: 'varianceScaling'
}));
model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
model.add(tf.layers.flatten());
model.add(tf.layers.dense(
    {units: 10, kernelInitializer: 'varianceScaling', activation: 'softmax'}));

const LEARNING_RATE = 0.15;
const optimizer = tf.train.sgd(LEARNING_RATE);
model.compile({
  optimizer: optimizer,
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy'],
});

const BATCH_SIZE = 64;
const TRAIN_BATCHES = 150;

// Every few batches, test accuracy over many examples. Ideally, we'd compute
// accuracy over the whole test set, but for performance we'll use a subset.
const TEST_BATCH_SIZE = 1000;
const TEST_ITERATION_FREQUENCY = 5;

let data;
async function load() {
  data = new MnistData();
  await data.load();
}

const Status = ({ status }) => {
  if (status === "training") {
    return (
      <div>Training...</div>
    );
  } else if (status === "testing") {
    return (
      <div>Training...</div>
    );
  }

  return (
    <div>Loading data...</div>
  );
};

Status.propTypes = {
  status: PropTypes.string.isRequired,
};

class Pred extends Component {
  constructor(props) {
    super(props);

    this.getRef = this.getRef.bind(this);
  }

  getRef(ref) {
    if (!this.canvas) {
      this.canvas = ref;
      const canvas = ReactDOM.findDOMNode(ref);
      const [width, height] = [28, 28];
      const ctx = canvas.getContext('2d');
      const imageData = new ImageData(width, height);
      const data = this.props.image.dataSync();
      for (let i = 0; i < height * width; ++i) {
        const j = i * 4;
        imageData.data[j + 0] = data[i] * 255;
        imageData.data[j + 1] = data[i] * 255;
        imageData.data[j + 2] = data[i] * 255;
        imageData.data[j + 3] = 255;
      }
      ctx.putImageData(imageData, 0, 0);
    }
  }

  render() {
    const {
      className,
      prediction,
    } = this.props;

    return (
      <div className="pred-container">
        <div className={className}>
          pred: {prediction}
        </div>
        <canvas
          className="prediction-canvas"
          ref={this.getRef}
          width={28}
          height={28}
        />
      </div>
    );
  }
}

Pred.propTypes = {
  className: PropTypes.string.isRequired,
  prediction: PropTypes.number.isRequired,
  image: PropTypes.object,
};

const Images = ({
  batch,
  predictions,
  labels,
}) => {
  const testExamples = batch.xs.shape[0];
  return Array(testExamples).fill(0).map((_, i) => {
    const image = batch.xs.slice([i, 0], [1, batch.xs.shape[1]]);

    const prediction = predictions[i];
    const label = labels[i];
    const correct = prediction === label;
    const predClassName = `pred ${(correct ? 'pred-correct' : 'pred-incorrect')}`;

    return (
      <Pred
        key={i}
        className={predClassName}
        prediction={prediction}
        image={image}
      />
    );
  });
};

class Stats extends Component {
  componentWillReceiveProps(nextProps) {
    embed(
      '#lossCanvas', {
        '$schema': 'https://vega.github.io/schema/vega-lite/v2.json',
        'data': {'values': nextProps.lossValues},
        'mark': {'type': 'line'},
        'width': 260,
        'orient': 'vertical',
        'encoding': {
          'x': {'field': 'batch', 'type': 'ordinal'},
          'y': {'field': 'loss', 'type': 'quantitative'},
          'color': {'field': 'set', 'type': 'nominal', 'legend': null},
        }
      },
      {width: 360});
    embed(
      '#accuracyCanvas', {
        '$schema': 'https://vega.github.io/schema/vega-lite/v2.json',
        'data': {'values': nextProps.accuracyValues},
        'width': 260,
        'mark': {'type': 'line', 'legend': null},
        'orient': 'vertical',
        'encoding': {
          'x': {'field': 'batch', 'type': 'ordinal'},
          'y': {'field': 'accuracy', 'type': 'quantitative'},
          'color': {'field': 'set', 'type': 'nominal', 'legend': null},
        }
      },
      {'width': 360});
  }

  shouldComponentUpdate(nextProps) {
    if (this.props.accuracyValues === null && nextProps.accuracyValues) {
      return true;
    }

    return false;
  }

  render() {
    const {
      lossValues,
      accuracyValues,
    } = this.props;

    const lastLoss = lossValues[lossValues.length - 1].loss.toFixed(2);

    return (
      <div id="stats">
        <div className="canvases">
          <div className="label" id="loss-label">
            last loss: {lastLoss}
          </div>
          <div ref={this.getLossRef} id="lossCanvas"></div>
        </div>
        <div className="canvases">
          {accuracyValues && (
            <Accuracy accuracyValues={accuracyValues} getAccuracyRef={this.getAccuracyRef} />
          )}
        </div>
      </div>
    );
  }
}

const Accuracy = ({
  accuracyValues,
  getAccuracyRef,
}) => {
  const lastAccuracy = (accuracyValues[accuracyValues.length - 1].accuracy * 100).toFixed(2);

  return [
    <div key="label" className="label" id="accuracy-label">
      last accuracy: {lastAccuracy}%
    </div>,
    <div key="canvas" ref={getAccuracyRef} id="accuracyCanvas"></div>,
  ];
};

Stats.propTypes = {
  lossValues: PropTypes.arrayOf(PropTypes.shape({
    batch: PropTypes.number.isRequired,
    loss: PropTypes.number.isRequired,
    set: PropTypes.string.isRequired,
  })).isRequired,
  accuracyValues: PropTypes.arrayOf(PropTypes.shape({
    batch: PropTypes.number.isRequired,
    loss: PropTypes.number,
    set: PropTypes.string.isRequired,
  })),
};

class App extends Component {
  constructor(props) {
    super(props);

    this.state = Object.assign({
      status: "loading",
      batch: null,
      predictions: null,
      labels: null,
      lossValues: null,
      accuracyValues: null,
    });
  }

  async componentDidMount() {
    const d = new Date();

    await load();
    await this.train();
    this.showPredictions();
    console.log((new Date()).getTime() - d.getTime());
  }

  async train() {
    this.setState({
      status: "training",
    });

    const lossValues = [];
    const accuracyValues = [];

    for (let i = 0; i < TRAIN_BATCHES; i++) {
      const batch = data.nextTrainBatch(BATCH_SIZE);

      let testBatch;
      let validationData;
      // Every few batches test the accuracy of the mode.
      if (i % TEST_ITERATION_FREQUENCY === 0) {
        testBatch = data.nextTestBatch(TEST_BATCH_SIZE);
        validationData = [
          testBatch.xs.reshape([TEST_BATCH_SIZE, 28, 28, 1]), testBatch.labels
        ];
      }

      // The entire dataset doesn't fit into memory so we call fit repeatedly
      // with batches.
      const history = await model.fit(
        batch.xs.reshape([BATCH_SIZE, 28, 28, 1]), batch.labels,
        {batchSize: BATCH_SIZE, validationData, epochs: 1});

      const loss = history.history.loss[0];
      const accuracy = history.history.acc[0];

      // Plot loss / accuracy.
      lossValues.push({'batch': i, 'loss': loss, 'set': 'train'});
      this.setState({
        lossValues,
      });

      if (testBatch != null) {
        accuracyValues.push({'batch': i, 'accuracy': accuracy, 'set': 'train'});
        this.setState({
          accuracyValues,
        });
      }

      batch.xs.dispose();
      batch.labels.dispose();
      if (testBatch != null) {
        testBatch.xs.dispose();
        testBatch.labels.dispose();
      }

      await tf.nextFrame();
    }
  }


  async showPredictions() {
    this.setState({
      status: "testing",
    });
    const testExamples = 100;
    const batch = data.nextTestBatch(testExamples);

    console.log("Do I need to tidy this?");
    // tf.tidy(() => {
    const output = model.predict(batch.xs.reshape([-1, 28, 28, 1]));

    const axis = 1;
    const labels = Array.from(batch.labels.argMax(axis).dataSync());
    const predictions = Array.from(output.argMax(axis).dataSync());

    this.setState({
      batch,
      predictions,
      labels,
    });

    // });
  }


  render() {
    return (
      <div className="App">
        <h3>TensorFlow.js: Train MNIST with the Layers API</h3>
        <Status status={this.state.status} />
        {this.state.lossValues && (
          <Stats
            lossValues={this.state.lossValues}
            accuracyValues={this.state.accuracyValues}
          />
        )}
        {this.state.batch && (
          <Images
            batch={this.state.batch}
            predictions={this.state.predictions}
            labels={this.state.labels}
          />
        )}
      </div>
    );
  }
}

ReactDOM.render(<App />, document.getElementById('root'));
