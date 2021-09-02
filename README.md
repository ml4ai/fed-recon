# Federated Reconnaissance
Repository containing the Federated Reconnaissance Benchmark with a suite of model implementations for multi-client class incremental continual learning.

## Code organization
The top level package is `fed_recon`.
Inside `fed_recon.benchmark` we define the mini-Imagenet Federated Reconnaissance Benchmark.
Our model implementations are available at `fed_recon.models`.
Single client continual learning evaluation and data sampling code is inside of `fed_recon.single_client`. 

## Benchmark
To evaluate a model on the full multi-client Federated Reconnaissance Benchmark:

```bash
python fed_recon/benchmark/eval_model.py --config /path/to/config.json --use-cuda --output_path /directory/path/to/write/results/
```

See `experiments` directory for scripts for running the benchmark with federated prototypical networks and iCaRL.

## Data
Mini-ImageNet [4] can be downloaded in the expected format from: 

[https://drive.google.com/drive/folders/1gEvVF5LmvSOfEVxfpDHcab1pg8S9udea?usp=sharing](https://drive.google.com/drive/folders/1gEvVF5LmvSOfEVxfpDHcab1pg8S9udea?usp=sharing)

Once downloaded, untar the file and unzip the images inside of it.

Similarly, Omniglot can be downloaded from:
[https://drive.google.com/file/d/1g8luTEJd1IkEi2En7x7Q-GVahQZ0yutC/view?usp=sharing](https://drive.google.com/file/d/1g8luTEJd1IkEi2En7x7Q-GVahQZ0yutC/view?usp=sharing)

## Models

Pretrained models are available at: [https://drive.google.com/file/d/12AN_VcLJQ5rpQz1REpHatFAvFjxjUzlM/view?usp=sharing](https://drive.google.com/file/d/12AN_VcLJQ5rpQz1REpHatFAvFjxjUzlM/view?usp=sharing)

### Lower
`fed_recon/models/gradient_based/lower.py`

This model simply trains on one task using SGD and then trains on the next task.
This method is the lower bound for incremental learning as 
the weights will be optimized for classification of last task and will not perform
well on previous tasks.

### iCaRL
`fed_recon/models/gradient_based/icarl.py`

Extends the iCaRL [3] method for multi-client class-incremental continual learning.

### Federated Prototypical Networks

`fed_recon/models/protonet/model.py`

Extends prototypical networks for multi-client class-incremental continual learning. We implement two prototype algorithms: one which stores all embeddings in memory and the prototypes are computed at inference time and a second more efficient method which updates prototypes estimates online.
Our model implementations for prototypical networks reference the original implementation from [1] and a newer implementation from [6]. 


To train a prototypical network [1] on Omniglot or mini-ImageNet, run:

```bash
python fed_recon/models/protonet/train.py  <args>
```
To evaluate the protonet on the single client class-incremental learning benchmark from [2], run:

```bash
python fed_recon/models/protonet/eval_cil.py <args>
```

See `train.py` and `eval_cil.py` for the available arguments. In particular, you will need to pass in `--dataset {omniglot or mini-imagenet}` and `--data-dir {/path/to/uncompressed/data}`


### Upper
`fed_recon/models/gradient_based/upper.py`

This model trains on all data that has been seen by all clients and represents an empirical upper bound of a neural network architecture given an ever growing dataset that is trained from scratch any time any amount of new data is observed.

## References
1. Snell, J., Swersky, K. and Zemel, R., 2017. Prototypical networks for few-shot learning. In Advances in neural information processing systems (pp. 4077-4087).
2. Javed, K. and White, M., 2019. Meta-learning representations for continual learning.
        In Advances in Neural Information Processing Systems (pp. 1820-1830).
        [https://proceedings.neurips.cc/paper/2019/hash/f4dd765c12f2ef67f98f3558c282a9cd-Abstract.html](https://proceedings.neurips.cc/paper/2019/hash/f4dd765c12f2ef67f98f3558c282a9cd-Abstract.html])
3. Rebuffi, S.A., Kolesnikov, A., Sperl, G. and Lampert, C.H., 2017. iCaRL: Incremental classifier and representation learning. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition (pp. 2001-2010).
4. Vinyals, O., Blundell, C., Lillicrap, T., Kavukcuoglu, K. and Wierstra, D., 2016. Matching networks for one shot learning. arXiv preprint arXiv:1606.04080.
5. Lake, B. M., Salakhutdinov, R., and Tenenbaum, J. B. (2015). Human-level concept learning through probabilistic program induction. Science, 350(6266), 1332-1338.
6. Deleu, T., Würfl, T., Samiei, M., Cohen, J.P. and Bengio, Y., 2019. Torchmeta: A meta-learning library for pytorch. arXiv preprint arXiv:1909.06576.