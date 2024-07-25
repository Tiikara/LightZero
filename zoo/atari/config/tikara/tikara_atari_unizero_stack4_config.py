from easydict import EasyDict
from zoo.atari.config.atari_env_action_space_map import atari_env_action_space_map
from zoo.atari.config.atari_unizero_config import atari_unizero_config, atari_unizero_create_config

env_id = 'PongNoFrameskip-v4'  # You can specify any Atari game here
action_space_size = atari_env_action_space_map[env_id]

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
update_per_collect = None
replay_ratio = 0.25
reanalyze_ratio = 0
batch_size = 64
# ==============================================================
# Размер контекста при обучении. Определяет размер максимальный размер контекста сети
# ==============================================================
num_unroll_steps = 10
# ==============================================================
# Определяет, сколько модель использует прошлых шагов.
# ==============================================================
infer_context_length = 4
max_env_step = int(5e5)
num_simulations = 50
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

atari_unizero_stack4_config = EasyDict(atari_unizero_config)  # Copy the base config

# Modify specific parts for stack4
atari_unizero_stack4_config.exp_name = f'data_unizero/{env_id[:-14]}_stack4_unizero_upc{update_per_collect}-rr{replay_ratio}_H{num_unroll_steps}_bs{batch_size}_seed0'
atari_unizero_stack4_config.env.update(
    dict(
        observation_shape=(4, 64, 64),
        gray_scale=True,
        collect_max_episode_steps=int(2e4),
        eval_max_episode_steps=int(1e4),
        save_replay=True,
        replay_path='/mnt/d/source/LightZero/data_unizero/replay',
    )
)
atari_unizero_stack4_config.policy.update(
    dict(
        learn=dict(
            learner=dict(
                hook=dict(
                    save_ckpt_after_iter=4000,
                )
            ),
        ),
        model=dict(
            observation_shape=(4, 64, 64),
            image_channel=1,
            frame_stack_num=4,
            gray_scale=True,
            action_space_size=action_space_size,
            world_model_cfg=dict(
                max_blocks=num_unroll_steps,
                max_tokens=2 * num_unroll_steps,
                context_length=2 * infer_context_length,
                device='cuda',
                # device='cpu',
                action_space_size=action_space_size,
                num_layers=4,
                num_heads=8,
                embed_dim=768,
                obs_type='image',
                env_num=max(collector_env_num, evaluator_env_num),
            ),
        ),
        # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
        model_path='/mnt/d/source/LightZero/data_unizero/001_Pong_stack4_unizero_upcNone-rr0.25_H10_bs64_seed0/ckpt/iteration_126123.pth.tar',
        update_per_collect=update_per_collect,
        replay_ratio=replay_ratio,
        batch_size=batch_size,
        optim_type='AdamW',
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    )
)
main_config = atari_unizero_stack4_config
create_config = atari_unizero_create_config

if __name__ == "__main__":
    seeds = [0]  # You can add more seed values here
    for seed in seeds:
        # Update exp_name to include the current seed
        main_config.exp_name = f'data_unizero/{env_id[:-14]}_stack4_unizero_upc{update_per_collect}-rr{replay_ratio}_H{num_unroll_steps}_bs{batch_size}_seed{seed}'
        from lzero.entry import train_unizero
        train_unizero([main_config, create_config], model_path=main_config.policy.model_path, seed=seed, max_env_step=max_env_step)
