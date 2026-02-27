# Needle Reach 扩展 — 中文快速指南

此文件旨在帮助你快速理解本仓库的结构、重要文件与如何在本地验证与运行示例环境（基于 Isaac Lab / Omniverse 扩展模板）。

本说明以简洁的方式列出关键文件、它们的作用、如何快速运行与调试，以及常见检查点与建议的下一步操作。

---

## 一、项目概览（一句话）

这是一个基于 Isaac Lab 的扩展模板（包名 `needle_reach`），包含示例任务（environment）、训练脚本（与 RL-Games 集成）、扩展元数据与 UI 示例。用于在独立于核心 Isaac Lab 的环境中开发和测试强化学习场景与扩展。

## 二、重要文件与目录（按重要性排序，路径相对于仓库根目录）

- `source/needle_reach/config/extension.toml`
  - 作用：扩展元数据（名称、版本、依赖、注册的 Python 模块等）。
  - 为什么重要：Isaac Lab 的扩展管理器与 `setup.py` 使用它来识别并加载扩展。

- `source/needle_reach/setup.py`
  - 作用：安装脚本，读取 `extension.toml` 并配置 pip 安装元数据。
  - 为什么重要：用于 `pip install -e source/needle_reach`，本地可编辑安装依赖此文件。

- `source/needle_reach/needle_reach/__init__.py`
  - 作用：包入口，导入 `tasks`（注册环境任务）和 `ui_extension_example`（注册示例 UI）。
  - 为什么重要：导入此模块会触发任务与 UI 的注册，若导入失败则扩展无法被发现。

- `source/needle_reach/needle_reach/tasks/manager_based/needle_reach/needle_reach_env_cfg.py`
  - 作用：核心环境配置（场景、观测、动作、事件、奖励、终止条件与仿真参数）。
  - 为什么重要：定义了环境的行为与 MDP 接口，是实现任务的主要入口。

- `source/needle_reach/needle_reach/tasks/manager_based/needle_reach/mdp/`（例如 `rewards.py`）
  - 作用：MDP 细节实现（奖励函数、观测处理、重置逻辑、终止检测等）。
  - 为什么重要：包含任务的策略性/目标性逻辑。

- `source/needle_reach/needle_reach/tasks/manager_based/needle_reach/agents/rl_games_ppo_cfg.yaml`
  - 作用：RL-Games agent 与训练超参数配置文件。
  - 为什么重要：`scripts/rl_games/train.py` 读取此 YAML 以加载训练相关参数。

- `scripts/rl_games/train.py`
  - 作用：训练入口，结合 AppLauncher、Hydra、RL-Games 与环境注册流程来运行训练。
  - 为什么重要：执行训练/评估的主脚本（需 Isaac Sim 支持）。

- `scripts/zero_agent.py` / `scripts/random_agent.py`
  - 作用：快速的“哑” agent，用来验证环境是否能正确加载与交互（不训练）。
  - 为什么重要：常用于 smoke-test（快速上手检查）。

- `scripts/list_envs.py`
  - 作用：列出当前 Python 环境中可用的任务/环境名（用于确认扩展被注册）。

## 三、如何快速验证（在已安装 Isaac Lab 的 Python 环境中）

1. 用可编辑模式安装扩展（使用安装了 Isaac Lab 的 Python）：

```bash
pip install -e source/needle_reach
```

2. 列出可用任务，确认扩展已被发现：

```bash
python scripts/list_envs.py
```

3. 使用零动作 agent 验证环境是否能被创建并能运行几个步长：

```bash
python scripts/zero_agent.py --task=<TASK_NAME>
```

或使用随机 agent：

```bash
python scripts/random_agent.py --task=<TASK_NAME>
```

4. 启动训练（仅在 Isaac Sim 可用并正确配置 AppLauncher 时）：

```bash
python scripts/rl_games/train.py --task=<TASK_NAME> --num_envs=4096 --device=cuda:0
```

说明：`train.py` 支持额外参数（video, checkpoint, distributed, wandb 等），详见脚本内的 argparse 注释。

## 四、典型工作流程（开发者视角）

1. 修改环境逻辑（比如奖励 / 终止 / 重置）：编辑 `.../mdp/*`（例如 `rewards.py`）和 `needle_reach_env_cfg.py` 中的配置。
2. 修改训练超参数：编辑 `agents/rl_games_ppo_cfg.yaml`。
3. 本地快速验证：先用 `zero_agent.py` 或 `random_agent.py` 做 smoke-test；确认无错误后再运行 `train.py`。
4. 若扩展无法被列出，检查 `source/needle_reach/needle_reach/__init__.py` 的导入是否发生异常（导入时异常会阻止注册）。

## 五、常见问题与排查要点

- 扩展未出现在 `list_envs.py` 的输出：
  - 检查是否在正确的 Python 环境中运行（需要包含 Isaac Lab）；确认 `pip install -e source/needle_reach` 成功且没有导入错误。
  - 在 Python 中尝试导入包并查看错误：

```python
python -c "import needle_reach"
```

- 无法启动训练（AppLauncher/Omniverse/Isaac Sim 相关）：
  - 确认 `isaaclab` 已安装并且 `AppLauncher` 能启动 Omniverse Kit；若在无 GUI 的机器上运行，请检查 headless 支持。

## 六、建议的下一步（按优先级）

1. 使用 `zero_agent.py` 做快速 smoke-test，确保环境能被创建。  
2. 根据需要调整 `agents/rl_games_ppo_cfg.yaml`（尤其是 `num_actors`、`learning_rate`、`max_epochs` 等），再运行小规模训练。  
3. 如果需要 UI 调试，启用 `ui_extension_example` 并在 Omniverse Kit 中加载扩展（在 Extension Manager 中添加 `source` 路径）。

## 七、文件快速树（精简，最多关注点）

```
README.md
README_CN.md  #（本文件）
scripts/
  list_envs.py
  zero_agent.py
  random_agent.py
  rl_games/train.py
source/needle_reach/
  config/extension.toml
  setup.py
  needle_reach/
    __init__.py
    ui_extension_example.py
    tasks/
      manager_based/needle_reach/
        needle_reach_env_cfg.py
        mdp/ (奖励/重置/观测实现)
        agents/rl_games_ppo_cfg.yaml

```

---

如需我把该文件移动到 `source/needle_reach/docs/` 下、生成更详细的文件树（包含每个模块的函数级说明），或在你的环境中实际运行 `python scripts/list_envs.py` 并把输出贴回给你做进一步诊断，我可以继续帮你执行这些步骤。（注意：运行 `list_envs.py` 需要 Isaac Sim/Isaac Lab 在当前 Python 环境中可用。）

完成：已创建 `README_CN.md`，包含关键文件说明、快速上手步骤与常见排查建议。
