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
replay_ratio = 0.25 # 0.25 - UniZero
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
        # collect_max_episode_steps=3000, # demon attack: changed
        # eval_max_episode_steps=3000, # demon attack: changed
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
                        channels_scale=1.6,
                        num_blocks=0,
                    ),
                    flat=dict(
                        start_channels=32,
                        channels_scale=1.,
                        num_blocks=3,
                    ),
                ),
                num_capsules=128,
                projection=dict(
                    type=False, # res_feed_forward |
                    num_layers=1,
                    last_norm=None
                ),
                head_type='linear_grouped_instance_norm',
                head=dict(
                    linear=dict(
                        use_coords=False
                    ),
                    caps=dict(
                        use_linear_input_for_caps=False,
                        double_linear_input_for_caps=False,
                        use_routing=True,
                        use_squash_in_transformer=True,
                    ),
                    simnorm_positional=dict(
                        pool_type='max'
                    )
                )
            ),
            use_latent_decoder_espcn=False,  # espcn model
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
                group_size=32,
                obs_type='image',
                transformer_norm_type='LN', # LN - LayerNorm (default: UniZero), RMS - RMSNorm
                transformer_feed_forward_type='base', # base - L->GELU-L (default: UniZero), swiglu - L->SwiGLU-L
                env_num=max(collector_env_num, evaluator_env_num),
                # latent_recon_loss_weight=0.1,
                # perceptual_loss_weight=0.1,
                # predict_latent_loss_type='mse'
                predict_latent_loss_type='log_cosh',
                reg_type=False, # vic |
                use_noisy_aug=True,
                noise_config=dict(
                    use_norm=False,
                    encode_noise_info_to_latent=False,
                    noise_strength_config=dict(
                        mult_random_distributions=[
                            dict(
                                type='sample_seq',
                                sample_seq=dict(
                                    noise_samples_perc=0.75,
                                    seq_length=10
                                )
                            ),
                            dict(
                                type='rand_linear'
                            )
                        ]
                    ),
                    noise_scheduler=dict(
                        initial_noise = 0.25,
                        final_noise = 0.,
                        schedule_length = 1000,
                        decay_type = 'constant'
                    )
                ),
                latent_enc_entropy_loss_weight=None,
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
        # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
        # /mnt/d/source/LightZero/data_unizero/Pong_stack1_unizero_upcNone-rr0.25_H10_bs64_seed1726102429/ckpt/ckpt_best.pth.tar
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
        # grad_clip_value=20., # UniZero - 5
        # model_update_ratio= 0.75 # UniZero - 0.25
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
    # int(time.time())
    seeds = [1726102429]  # You can add more seed values here
    for seed in seeds:
        # Update exp_name to include the current seed
        main_config.exp_name = f'data_unizero/{env_id[:-14]}_stack1_unizero_upc{update_per_collect}-rr{replay_ratio}_H{num_unroll_steps}_bs{batch_size}_seed{seed}'
        from lzero.entry import train_unizero

        train_unizero([main_config, create_config], seed=seed, model_path=main_config.policy.model_path,
                      max_env_step=max_env_step)
