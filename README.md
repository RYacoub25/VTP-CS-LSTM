<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>VTP-CS-LSTM: Contextual Social LSTM for Vehicle Trajectory Prediction</title>
<style>
  body {
    font-family: Arial, sans-serif;
    max-width: 900px;
    margin: 40px auto;
    line-height: 1.6;
    padding: 0 15px;
    color: #333;
  }
  h1, h2, h3 {
    color: #222;
  }
  pre {
    background: #f4f4f4;
    padding: 15px;
    border-radius: 5px;
    overflow-x: auto;
  }
  code {
    font-family: Consolas, monospace;
    font-size: 0.95em;
  }
  table {
    border-collapse: collapse;
    width: 100%;
    margin: 15px 0;
  }
  table, th, td {
    border: 1px solid #ccc;
  }
  th, td {
    padding: 8px 12px;
    text-align: left;
  }
  blockquote {
    border-left: 5px solid #ccc;
    margin-left: 0;
    padding-left: 15px;
    color: #666;
    font-style: italic;
  }
  a {
    color: #0366d6;
    text-decoration: none;
  }
  a:hover {
    text-decoration: underline;
  }
</style>
</head>
<body>

<h1>VTP-CS-LSTM: Contextual Social LSTM for Vehicle Trajectory Prediction</h1>

<h2>Overview</h2>
<p>This repository implements a Contextual Social LSTM (CS-LSTM) model to predict future vehicle trajectories in a mixed traffic environment using the Argoverse 2 dataset. The model uses historical trajectories of a target vehicle and its neighbors, combined with contextual lane and map information, to forecast future positions.</p>
<ul>
  <li>Predicts future incremental displacements over a sequence of time steps.</li>
  <li>Considers neighboring vehicles within a configurable radius to capture social interactions.</li>
  <li>Integrates lane boundary and map object features to provide contextual awareness.</li>
  <li>Supports visualization of predicted trajectories alongside ground truth and map elements.</li>
</ul>

<hr />

<h2>Dataset</h2>
<p>The training, validation, and testing datasets are preprocessed from the Argoverse 2 raw data into:</p>
<ul>
  <li><code>ego_vehicle_with_intention.csv</code> — ego vehicle states with intention labels.</li>
  <li><code>social_vehicles_relative.csv</code> — relative positions and states of neighboring vehicles.</li>
  <li><code>contextual_features_merged.npy</code> — contextual lane and map features.</li>
  <li><code>constant_features.csv</code> — lane boundary geometries.</li>
  <li><code>map_objects.csv</code> — traffic signs and map objects.</li>
</ul>

<hr />

<h2>Model Architecture</h2>
<p>The CS-LSTM model consists of:</p>
<ul>
  <li>LSTM encoders for ego vehicle, target vehicle, and neighboring vehicles.</li>
  <li>Spatial-temporal attention mechanisms over neighbor encodings.</li>
  <li>Integration of contextual lane and map features through fully connected layers.</li>
  <li>Decoder LSTM that outputs incremental displacements for future trajectory prediction.</li>
</ul>

<hr />

<h2>Installation and Setup</h2>

<h3>Prerequisites</h3>
<ul>
  <li>Python 3.9+</li>
  <li>Conda for environment management</li>
  <li>PyTorch with CUDA support (optional for GPU acceleration)</li>
</ul>

<h3>Create Environment</h3>
<pre><code>conda env create -f environment.yml
conda activate av2-py39
</code></pre>

<h3>Data Preparation</h3>
<p>Ensure the processed dataset files are placed under <code>data/processed/</code> as described above.</p>

<hr />

<h2>Training</h2>
<p>Run the training script with default or customized parameters:</p>

<pre><code>python CS_LSTM/train.py --use_delta_yaw --use_intention --epochs 20 --batch 64
</code></pre>

<p>Example with customized parameters:</p>

<pre><code>python CS_LSTM/train.py --seq_len 30 --pred_len 30 --use_delta_yaw --batch 64 --epochs 30
</code></pre>

<hr />

<h2>Visualization</h2>
<p>Visualize predictions, ego vehicle, target vehicle, neighbors, lane boundaries, and map objects:</p>

<pre><code>python visualize/visualize_everything.py
</code></pre>

<p>The visualization shows:</p>
<ul>
  <li>Ego vehicle as a car icon.</li>
  <li>Target vehicle as a purple dot with its history, predicted and ground truth trajectories.</li>
  <li>Neighbor vehicles, lane boundaries, and map objects in context.</li>
</ul>

<hr />

<h2>Performance Metrics</h2>
<p>The model predicts 30 future timesteps (3 seconds at 10Hz) with average displacement errors (ADE) and final displacement errors (FDE) computed on the validation set.</p>

<hr />

<h2>Citation</h2>
<p>If you use this repository for your research, please cite:</p>

<pre><code>@article{lin2021vehicle,
  title={Vehicle Trajectory Prediction Using LSTMs With Spatial–Temporal Attention Mechanisms},
  author={Lin, Lei and Li, Weizi and Bi, Huikun and Qin, Lingqiao},
  journal={IEEE Intelligent Transportation Systems Magazine},
  volume={13},
  number={1},
  pages={111--124},
  year={2021},
  publisher={IEEE},
  doi={10.1109/MITS.2021.3049404},
  url={https://github.com/leilin-research/VTP}
}
</code></pre>

<p>Or:</p>

<p><em>Lin, L., Li, W., Bi, H., &amp; Qin, L. (2021). Vehicle Trajectory Prediction Using LSTMs With Spatial–Temporal Attention Mechanisms. <strong>IEEE Intelligent Transportation Systems Magazine</strong>, 13(1), 111-124. <a href="https://doi.org/10.1109/MITS.2021.3049404" target="_blank">https://doi.org/10.1109/MITS.2021.3049404</a></em></p>



<h2>Contact</h2>
<p>For questions or issues, please open an issue on GitHub or contact the author.</p>

</body>
</html>
