<p align="center">
    <img src="assets/logo.png?raw=true" width="240" style="margin-bottom: 0.1;"/>
<p>
<h3 align="center">
  <a href="https://github.com/SJTU-DENG-Lab/WLA">World-Language-Action Model for Unified World Modeling,<br>Language Reasoning, and Action Synthesis</a>
</h3>
<h4 align="center"> 
  If you find our project helpful, please give us a star ⭐ to support us 🙏🙏
</h4>



<p align="center">
  <a href="https://arxiv.org/pdf/2606.05979"><b>📄 Paper</b></a> |
  <a href="https://huggingface.co/collections/SJTU-DENG-Lab/wla"><b>🤗 Checkpoints</b></a> |
  <a href="./LICENSE"><b>📜 License</b></a>
</p>



<br>

https://github.com/user-attachments/assets/64884200-a8f5-42d6-8ab5-6a15ed73f82f

<br>

## 📘 Contents

- [ToDo](#-todo)
- [Models & Datasets](#-models--datasets)
- [Evaluation](#-evaluation)
- [Training](#-training)
- [Acknowledgements](#-acknowledgements)
- [Citation](#-citation)



## 📌 ToDo

- [x] Training and evaluation code for LIBERO
- [x] Training and evaluation code for RoboTwin 2.0
- [x] Training and evaluation code for RMBench
- [x] Release code for *learning new tasks from videos*
- [ ] Release code for *Efficient Mode*
- [ ] Release code for *TTS Mode*



## 🤗 Models & Datasets

<table style="width:100%;border-collapse:collapse;table-layout: fixed">
  <tr>
    <th style="text-align:center;width:50%;">Model</th>
    <th style="text-align:center;width:50%;">Note</th>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/SJTU-DENG-Lab/wla_libero_all_image_action">wla_libero_all_image_action</a></td>
    <td>Trained on all four LIBERO suites</td></td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/SJTU-DENG-Lab/wla_robotwin_all_image_action">wla_robotwin_all_image_action</a></td>
    <td>Trained across all 50 RoboTwin 2.0 tasks</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/SJTU-DENG-Lab/wla_rmbench_battery_try_image_action_language">wla_rmbench_battery_try_image_action_language</a></td>
    <td>Trained on RMBench Battery Try task</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/SJTU-DENG-Lab/wla_rmbench_blocks_ranking_try_image_action_language">wla_rmbench_blocks_ranking_try_image_action_language</a></td>
    <td>Trained on RMBench Blocks Ranking Try task</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/SJTU-DENG-Lab/wla_rmbench_cover_blocks_image_action_language">wla_rmbench_cover_blocks_image_action_language</a></td>
    <td>Trained on RMBench Cover Blocks task</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/SJTU-DENG-Lab/wla_rmbench_press_button_image_action_language">wla_rmbench_press_button_image_action_language</a></td>
    <td>Trained on RMBench Press Button task</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/SJTU-DENG-Lab/wla_robotwin_same_emb_videos_cotrain_image_action">
wla_robotwin_same_emb_videos_cotrain_image_action</a></td>
    <td>Jointly trained on 45 seen tasks and same-embodiment videos of 5 unseen tasks</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/SJTU-DENG-Lab/wla_robotwin_cross_emb_videos_cotrain_image_action">
wla_robotwin_cross_emb_videos_cotrain_image_action</a></td>
    <td>Jointly trained on 45 seen tasks and cross-embodiment videos of 5 unseen tasks</td>
  </tr>
</table>

<table style="width:100%;border-collapse:collapse;table-layout: fixed">
  <tr>
    <th style="text-align:center;width:40%;">Dataset</th>
    <th style="text-align:center;width:60%;">Note</th>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/datasets/SJTU-DENG-Lab/LIBERO_LeRobot">LIBERO_LeRobot</a></td>
    <td>The LIBERO dataset in LeRobot v3.0 format</td></td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/datasets/SJTU-DENG-Lab/RoboTwin-LeRobot">RoboTwin-LeRobot</a></td>
    <td>The RoboTwin 2.0 dataset in LeRobot v3.0 format</td>
  </tr>
  <tr>
    <td>
    <a href="https://huggingface.co/datasets/SJTU-DENG-Lab/RoboTwin-LeRobot-seen-tasks">RoboTwin-LeRobot-seen-tasks</a>
    </td>
    <td>The 45 seen-task subset of RoboTwin 2.0</td>
  </tr>
    <tr>
    <td>
    <a href="https://huggingface.co/datasets/SJTU-DENG-Lab/RoboTwin-LeRobot-unseen-tasks-same-emb">RoboTwin-Lerobot-unseen-tasks-same-emb</a>
    </td>
    <td>The 5 unseen-task subset of RoboTwin 2.0 under the same-embodiment setting</td>
  </tr>
  </tr>
    <tr>
    <td>
    <a href="https://huggingface.co/datasets/SJTU-DENG-Lab/RoboTwin-LeRobot-unseen-tasks-cross-emb">RoboTwin-Lerobot-unseen-tasks-cross-emb</a>
    </td>
    <td>The 5 unseen-task subset of RoboTwin 2.0 under the cross-embodiment setting</td>
  </tr>
  </tr>
    <tr>
    <td>
    <a href="https://huggingface.co/datasets/SJTU-DENG-Lab/RMBench-LeRobot">RMBench-LeRobot</a>
    </td>
    <td>The four M(n) subset of RMBench in LeRobot v3.0 format</td>
  </tr>
</table>






## 📈 Evaluation

### LIBERO

First, clone the repository and create the conda environment:

```
git clone git@github.com:SJTU-DENG-Lab/WLA.git
cd WLA
conda env create -f configs/environment_libero.yml
conda activate wla_libero
```

Then clone and install the [LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO):

```
git clone git@github.com:Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
```

Install other required packages:

```
cd ..
pip install -r experiments/libero/libero_requirements.txt
```

Evaluate the LIBERO benchmark:

```
bash experiments/libero/run_libero_eval.sh
```

You can modify the `task_suite_name` in the script to evaluate different task suites.

### RoboTwin 2.0

First, create and activate the conda environment:

```
conda create -n wla_robotwin python=3.10 -y
conda activate wla_robotwin
```

Next, clone the [RoboTwin 2.0 repository](https://github.com/robotwin-Platform/robotwin), install its dependencies, and download the required assets:

```
git clone https://github.com/RoboTwin-Platform/RoboTwin.git
cd RoboTwin
bash script/_install.sh
bash script/_download_assets.sh
```

If the installation fails, please follow the [official installation guide](https://robotwin-platform.github.io/doc/usage/robotwin-install.html#5-manual-installation-only-when-step-3-failed) to install RoboTwin manually. 

Then, return to the project root directory and install the remaining required packages:

```
cd ..
pip install -r experiments/robotwin/robotwin_requirements.txt
```

Finally, run the following script to evaluate the model on the RoboTwin 2.0 benchmark:

```
bash experiments/robotwin/run_robotwin_eval.sh
```

Modify `TASK_NAME` to evaluate different tasks. `TASK_CONFIG` determines whether to evaluate the `demo_clean` or `demo_randomized` setting. 

`CONTROL_MODE` specifies the control mode: `eef` for 16-dim end-effector actions / states, or `joint` for 14-dim joint-angle actions / states. `wla_libero_all_image_action` uses `eef`, while the *learning new tasks from videos* experiments use `joint`.

### RMBench

First, create and activate the conda environment:

```
conda create -n wla_rmbench python=3.10 -y
conda activate wla_rmbench
```

Next, clone the [RMBench repository](https://github.com/robotwin-Platform/rmbench), install its dependencies, and download the required assets. Before installation, please replace `script/_install.sh` with the version from [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/script/_install.sh) to avoid some installation issues:

```
git clone git@github.com:RoboTwin-Platform/RMBench.git
cd RMBench
bash script/_install.sh
bash script/_download_assets.sh
```

Then, return to the project root directory and install the remaining required packages:

```
cd ..
pip install -r experiments/rmbench/rmbench_requirements.txt
```

Finally, run the following script to evaluate the model on the RMBench:

```
bash experiments/rmbench/run_rmbench_eval.sh
```

Modify `TASK_NAME` to evaluate different tasks. For different tasks, please use the corresponding `MODEL_ID` and `UNNORM_KEY`.





## 🔧 Training

First, create the training conda environment:

```
conda env create -f configs/environment_lerobot.yml
conda activate wla_lerobot
```

If you want to accelerate training with FlashAttention, run the following command to install it:

```
wget https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.8-cp311-cp311-linux_x86_64.whl
pip install ./flash_attn-2.8.3+cu128torch2.8-cp311-cp311-linux_x86_64.whl
```

Next, modify the `attn_implementation` parameter in `models/model.py`:

```
attn_implementation="eager" => attn_implementation="flash_attention_2"
```

Before training on RoboTwin, please download the `robotwin_seen_instruction.json` file:

```
hf download SJTU-DENG-Lab/robotwin_seen_instruction --local-dir configs --repo-type dataset
```

Then run the training script:

```
sh train.sh
```

You can modify the `TRAINING_SETTING` parameter in the script to train under different settings. The available options are as follows:

- `libero_all_image_action`: Training on all four LIBERO suites
- `libero_all_action`: Training on all four LIBERO suites without the World Expert
- `robotwin_all_image_action`: Training on all 50 RoboTwin 2.0 tasks
- `robotwin_all_action`: Training on all 50 RoboTwin 2.0 tasks without the World Expert
- `robotwin_seen_tasks_image_action`: Training on the 45 seen-task subset of RoboTwin 2.0
- `robotwin_cross_emb_videos_cotrain_image_action`: Joint training on 45 seen tasks and cross-embodiment videos of 5 unseen tasks
- `robotwin_same_emb_videos_cotrain_image_action`: Joint training on 45 seen tasks and same-embodiment videos of 5 unseen tasks
- `rmbench_battery_try_image_action_language.yaml`: Training on the RMBench *Battery Try* task
- `rmbench_blocks_ranking_try_image_action_language.yaml`: Training on the RMBench *Blocks Ranking Try* task
- `rmbench_cover_blocks_image_action_language.yaml`: Training on the RMBench *Cover Blocks* task
- `rmbench_press_button_image_action_language.yaml`: Training on the RMBench *Press Button* task




## ✨ Acknowledgements

Heartfelt thanks to the creators of [StarVLA](https://github.com/starVLA/starVLA) and [LeRobot](https://github.com/huggingface/lerobot) for their open-sourced work!



## 📝 Citation

If you find our code or models useful in your work, please cite [our paper](https://arxiv.org/pdf/2606.05979):

```
@article{yang2026world,
  title={World-Language-Action Model for Unified World Modeling, Language Reasoning, and Action Synthesis},
  author={Yang, Yi and Liu, Zhihong and Kou, Siqi and Chen, Yiyang and Hu, Yanzhe and Zhou, Jianbo and Zhao, Boyuan and Wei, Zhijie and Xia, Xiao and Li, Xueqi and others},
  journal={arXiv preprint arXiv:2606.05979},
  year={2026}
}
```