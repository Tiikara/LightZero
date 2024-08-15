from easydict import EasyDict
import time
from zoo.atari.config.atari_env_action_space_map import atari_env_action_space_map
import torch

env_id = 'PongNoFrameskip-v4'  # You can specify any Atari game here
action_space_size = atari_env_action_space_map[env_id]

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
update_per_collect = None
replay_ratio = 0.25
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
max_env_step = int(1e6)
reanalyze_ratio = 0.
batch_size = 64
# ==============================================================
# Размер контекста при обучении. Определяет размер максимальный размер контекста сети
# ==============================================================
num_unroll_steps = 10
# ==============================================================
# Определяет, сколько модель использует прошлых шагов.
# ==============================================================
infer_context_length = 4

observation_shape = (3, 64, 64)

# ====== only for debug =====
# collector_env_num = 2
# n_episode = 2
# evaluator_env_num = 2
# num_simulations = 5
# max_env_step = int(5e5)
# reanalyze_ratio = 0.
# batch_size = 2
# num_unroll_steps = 10
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

atari_unizero_config = dict(
    env=dict(
        stop_value=int(1e6),
        env_id=env_id,
        observation_shape=observation_shape,
        gray_scale=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        save_replay=True,
        replay_path='/mnt/d/source/LightZero/data_unizero/replay',
        continous_reward_wrapper=dict(
            enabled=False,
            # Enable only if the game requires a mandatory action. An agent in a game can't just do nothing
            reward=0.001,
            max_reward=0.25
        ),
        # TODO: only for debug
        # collect_max_episode_steps=int(50),
        # eval_max_episode_steps=int(50),
    ),
    policy=dict(
        model=dict(
            observation_shape=observation_shape,
            action_space_size=action_space_size,
            representaion_model=dict(
                type='downsample',
                downsample_network=dict(
                    type='base',
                    res_net=dict(
                        use_coords=False,
                        start_channels=32,
                        channels_scale=1.2,
                        num_blocks=1,
                    )
                ),
                num_capsules=128,
                use_linear_input_for_caps=True,
                double_linear_input_for_caps=False,
                use_routing=False,
                use_squash_in_transformer=True,
                head_type='caps_max_positional',
                head=dict(
                    simnorm_positional=dict(
                        pool_type='max'
                    )
                )
            ),
            use_latent_decoder_espcn=True,  # More accurate model
            world_model_cfg=dict(
                max_blocks=num_unroll_steps,
                max_tokens=2 * num_unroll_steps,  # NOTE: each timestep has 2 tokens: obs and action
                context_length=2 * infer_context_length,
                device='cuda',
                # device='cpu',
                action_space_size=action_space_size,
                num_layers=4,
                num_heads=8,
                embed_dim=768,
                obs_type='image',
                env_num=max(collector_env_num, evaluator_env_num),
                # latent_recon_loss_weight=0.01,
                # perceptual_loss_weight=10.,
                # predict_latent_loss_type='mse'
                predict_latent_loss_type='caps',
                caps_direction_loss_weight=2.,
                value_loss_weight=0.25,  # 0.25 - UniZero
                obs_loss_weight=10.  # 10. - UniZero
            ),
        ),
        learn=dict(
            learner=dict(
                hook=dict(
                    save_ckpt_after_iter=10_000,
                )
            ),
        ),
        use_late_dropout=False,
        # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
        model_path=None,
        num_unroll_steps=num_unroll_steps,
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
    ),
)
atari_unizero_config = EasyDict(atari_unizero_config)
main_config = atari_unizero_config

atari_unizero_create_config = dict(
    env=dict(
        type='atari_lightzero',
        import_names=['zoo.atari.envs.atari_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='unizero',
        import_names=['lzero.policy.unizero'],
    ),
)
atari_unizero_create_config = EasyDict(atari_unizero_create_config)
create_config = atari_unizero_create_config

if __name__ == "__main__":
    # Define a list of seeds for multiple runs
    seeds = [int(time.time())]  # You can add more seed values here
    for seed in seeds:
        # Update exp_name to include the current seed
        main_config.exp_name = f'data_unizero/{env_id[:-14]}_stack1_unizero_upc{update_per_collect}-rr{replay_ratio}_H{num_unroll_steps}_bs{batch_size}_seed{seed}'
        from lzero.entry import train_unizero

        train_unizero([main_config, create_config], seed=seed, model_path=main_config.policy.model_path,
                      max_env_step=max_env_step)
