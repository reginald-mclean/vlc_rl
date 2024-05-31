# Video-Language Critic: Transferable Reward Functions for Language-Conditioned Robotics 

[Minttu Alakuijala](#)<sup>1</sup>, [Reginald McLean](https://www.reggiemclean.ca)<sup>2</sup>, [Isaac Woungang](http://cs.torontomu.ca/~iwoungan)<sup>2</sup>, [Nariman Farsad](http://narimanfarsad.com/)<sup>2</sup>, [Samuel Kaski](#)<sup>1,3</sup>, [Pekka Marttinen](#)<sup>1</sup>, [Kai Yuan](https://www.linkedin.com/in/kai-yuan/) <sup>4</sup>

<sup>1</sup>Department of Computer Science, Aalto University, <sup>2</sup>Department of Computer Science, Toronto Metropolitan University, <sup>3</sup>Department of Computer Science, University of Manchester, <sup>4</sup>Intel Corporation

## This is the official training reinforcement learning training code for VLC, a reward model trained using contrastive learning and a temporal ranking objective for downstream robotics tasks.

<img src='assets/VLC_overview_v2.001.png'>

## Table of Contents  
[Installation](#installation)  

[VLC Experiments](#vlc-experiments)

[License](#license)

[Citation](#citation)

[Acknowledgement](#acknowledgement)

## Installation
If you do not have Video Language Critic (VLC) installed, please install this [repo]() first.

We do recommend creating an environment via [miniconda](https://docs.anaconda.com/free/miniconda/) as it can handle the requirements for headless rendering.

After cloning this repo, in the same environment that VLC is installed in:

```
cd VLC_RL/Metaworld;
pip install -e .;
cd ../Gymnasium;
pip install -e .;
cd ../;
pip install -r requirements.txt;
pip install 'jax[cuda12]==0.4.23'
```

## VLC Experiments

### Replication

### Rendering
placeholder

## License
The source code in this repository is licensed under the **X** License.

## Citation
If you find this repository or paper useful for your research, please cite

@article{alakuijala2024videolanguage,
      title={Video-Language Critic: Transferable Reward Functions for Language-Conditioned Robotics}, 
      author={Minttu Alakuijala and Reginald McLean and Isaac Woungang and Nariman Farsad and Samuel Kaski and Pekka Marttinen and Kai Yuan},
      year={2024},
      eprint={2405.19988},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}

## Acknowledgement
This implementation is based off [CleanRL](https://github.com/vwxyzjn/cleanrl)
