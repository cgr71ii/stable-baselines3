from typing import Any, Dict, List, Optional, Type, Union

import torch as th
from gymnasium import spaces
from torch import nn
import numpy as np

from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule


class Actor(BasePolicy):
    """
    Actor network (policy) for TD3.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn

        #action_dim = get_action_dim(self.action_space)
        #actor_net = create_mlp(features_dim, action_dim, net_arch, activation_fn, squash_output=True)
        self.action_dim = get_action_dim(self.action_space)
        actor_net = create_mlp(features_dim, self.action_dim, net_arch, activation_fn, squash_output=True)
        # Deterministic action
        self.mu = nn.Sequential(*actor_net)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def forward(self, obs: th.Tensor) -> th.Tensor:
        # assert deterministic, 'The TD3 actor only outputs deterministic actions'
        features = self.extract_features(obs, self.features_extractor)
        return self.mu(features)

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        return self(observation)


class TD3Policy(BasePolicy):
    """
    Policy class (with both actor and critic) for TD3.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    actor: Actor
    actor_target: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
            normalize_images=normalize_images,
        )

        # Default network architecture, from the original paper
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = [256, 256]
            else:
                net_arch = [400, 300]

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
            }
        )

        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        # Create actor and target
        # the features extractor should not be shared
        self.actor = self.make_actor(features_extractor=None)
        self.actor_target = self.make_actor(features_extractor=None)
        # Initialize the target to have the same weights as the actor
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            # Critic target should not share the features extractor with critic
            # but it can share it with the actor target as actor and critic are sharing
            # the same features_extractor too
            # NOTE: as a result the effective poliak (soft-copy) coefficient for the features extractor
            # will be 2 * tau instead of tau (updated one time with the actor, a second time with the critic)
            self.critic_target = self.make_critic(features_extractor=self.actor_target.features_extractor)
        else:
            # Create new features extractor for each network
            self.critic = self.make_critic(features_extractor=None)
            self.critic_target = self.make_critic(features_extractor=None)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.optimizer = self.optimizer_class(
            self.critic.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        # Target networks should always be in eval mode
        self.actor_target.set_training_mode(False)
        self.critic_target.set_training_mode(False)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                share_features_extractor=self.share_features_extractor,
            )
        )
        return data

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return Actor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return ContinuousCritic(**critic_kwargs).to(self.device)

    def forward(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self._predict(observation, deterministic=deterministic)

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        return self.actor(observation)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode


MlpPolicy = TD3Policy

class WolpertingerPolicy(TD3Policy):

    #def __init__(self, *args, callback_retrieve_knn, embedding_size, k=0.1, **kwargs):
    def __init__(self, *args, callback_retrieve_knn=None, callback_retrieve_knn_training=None, k=0.1, add_all_knn_to_batch=False,
                 apply_rws_inference=False, **kwargs):
        super().__init__(*args, **kwargs)

        #n_critics = self.critic_kwargs["n_critics"]

        assert callback_retrieve_knn is not None, "callback_retrieve_knn kwarg is mandatory"
        #assert n_critics == 1, f"Expected n_critics was 1, but got {n_critics} (only DDPG is supported)"

        self.actor_input_size, self.actor_output_size = self.actor.features_dim, self.actor.action_dim
        self.critic_input_size, self.critic_output_size = self.critic_kwargs["net_arch"][0], self.critic_kwargs["net_arch"][-1]
        self.callback_retrieve_knn = callback_retrieve_knn # args: embedding, k
        self.callback_retrieve_knn_training = callback_retrieve_knn_training # args: same that self.callback_retrieve_knn
        #self.embedding_size = embedding_size
        self.k = k
        self.knn_percentage = isinstance(self.k, float)
        self.apply_rws_inference = apply_rws_inference

        if self.knn_percentage:
            min_k_percentage = 0.0001 # 0.01%

            assert min_k_percentage <= self.k <= 1.0, f"k={self.k} is not in [{min_k_percentage}, 1.0]"
        else:
            assert self.k >= 1, f"k={self.k} cannot be < 1"

        self.add_all_knn_to_batch = add_all_knn_to_batch # if True, actual batch_size will be retrieved_knn * batch_size

        if self.callback_retrieve_knn_training is None:
            self.callback_retrieve_knn_training = callback_retrieve_knn

    def roulette_wheel_selection(self, population, fitness_scores):
        population = population.detach().cpu().numpy()
        fitness_scores = fitness_scores.detach().cpu().numpy()

        assert len(fitness_scores.shape) == 3
        assert len(population.shape) == 3
        assert fitness_scores.shape[0] == population.shape[0] # batch_size
        assert fitness_scores.shape[1] <= self.k if not self.knn_percentage else True # k
        assert fitness_scores.shape[2] == 1 # Q value
        assert population.shape[1] <= self.k if not self.knn_percentage else True # k
        assert population.shape[2] == self.actor_output_size # action_space

        batch_size = fitness_scores.shape[0]
        result = []

        #total_fitness = sum(fitness_scores)
        total_fitness = np.sum(fitness_scores, axis=1) # (batch_size, 1)
        relative_fitness = np.array([f / t for f, t in zip(fitness_scores, total_fitness)]) # (batch_size, k, 1)
        cumulative_probability = np.array([sum(relative_fitness[i][:j+1]) for i in range(len(relative_fitness)) for j in range(len(relative_fitness[i]))]).reshape(relative_fitness.shape)

        for batch in range(batch_size):
            result.append([])

            rand = np.random.random()

            for i, cp in enumerate(cumulative_probability[batch]):
                cp = cp[0]

                if rand <= cp:
                    result[-1].append(population[batch][i])
                    break

        result = np.squeeze(np.array(result), axis=1)

        assert len(result.shape) == 2
        assert result.shape[0] == batch_size
        assert result.shape[1] == self.actor_output_size

        return result

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self._predict_conf(observation=observation, deterministic=deterministic, actor=self.actor, critic=self.critic, training=False)

    def _predict_conf(self, observation: PyTorchObs, deterministic: bool = False, actor: Actor = None, critic: ContinuousCritic = None, actor_noise: th.Tensor = None, actor_clamp: bool = False, training: bool = False) -> th.Tensor:
        if actor is None:
            actor = self.actor
        if critic is None:
            critic = self.critic

        assert len(observation.shape) == 2, f"Observation shape was expected to contain 2 elements, but got {len(observation.shape)}"
        assert observation.shape[1] == self.actor_input_size, f"Observation shape[1] was expected to be {self.actor_input_size}, but got {observation.shape[1]}"

        batch_size = observation.shape[0]
        proto_action = actor(observation) # (batch_size, action_space)

        if actor_noise is not None:
            # TD3-related
            proto_action = proto_action + actor_noise
        if actor_clamp:
            # TD3-related
            proto_action = proto_action.clamp(-1, 1)

        assert len(proto_action.shape) == 2, f"Proto action shape was expected to contain 2 elements, but got {len(proto_action.shape)}"
        assert proto_action.shape[0] == batch_size, f"Proto action shape[0] was expected to be {batch_size}, but got {proto_action.shape[0]}"
        assert proto_action.shape[1] == self.actor_output_size, f"Proto action shape[1] was expected to be {self.actor_output_size}, but got {proto_action.shape[1]}"

        knn_callback = self.callback_retrieve_knn_training if training else self.callback_retrieve_knn
        knn = knn_callback(proto_action.detach().cpu().numpy(), self.k) # (batch_size, <=k, action_space)

        if not isinstance(knn, th.Tensor):
            knn = th.tensor(knn).to(self.device)

        assert len(knn.shape) == 3, f"kNN shape was expected to contain 3 elements, but got {len(knn.shape)}"
        assert knn.shape[0] == batch_size, f"kNN shape[0] was expected to be {batch_size}, but got {knn.shape[0]}"
        assert knn.shape[1] <= self.k if not self.knn_percentage else True, f"kNN shape[1] was expected to be <={self.k}, but got {knn.shape[1]}"
        assert knn.shape[2] == self.actor_output_size, f"kNN shape[2] was expected to be {self.actor_output_size}, but got {knn.shape[2]}"

        _k = knn.shape[1]
        result = []

        if self.add_all_knn_to_batch:
            # observation: (batch_size, action_space)
            # knn: (batch_size, <=k, action_space)
            #_knn = knn.reshape((_k * batch_size, self.actor_output_size))
            _knn = th.cat(tuple([knn[::,knn_idx] for knn_idx in range(_k)]), dim=0).to(self.device)
            _observation = th.tile(observation, (_k, 1)).to(self.device)
            critic_output = critic(_observation, _knn)
            critic_output = th.cat(critic_output, dim=1)
            critic_output, _= th.min(critic_output, dim=1, keepdim=True)

            assert len(critic_output.shape) == 2
            assert critic_output.shape[0] == _k * batch_size, critic_output.shape
            assert critic_output.shape[1] == 1, critic_output.shape # Q(s,a)

            critic_output = critic_output.reshape((batch_size, _k, 1))
        else:
            # Process each neighbour individually
            partial_result = []

            for knn_idx in range(_k):
                _knn = knn[:,knn_idx] # Get each neighbour (the knn_idx th) for each observation
                critic_output = critic(observation, _knn) # Evaluate observations with each knn
                critic_output = th.cat(critic_output, dim=1)
                critic_output, _= th.min(critic_output, dim=1, keepdim=True)

                assert len(critic_output.shape) == 2, f"critic shape was expected to contain 2 elements, but got {len(critic_output.shape)}"
                assert critic_output.shape[0] == batch_size, f"critic shape[0] was expected to be {batch_size}, but got {critic_output.shape[0]}"
                assert critic_output.shape[1] == 1, f"critic shape[1] was expected to be 1, but got {critic_output.shape[1]}"

                partial_result.append(critic_output.detach().cpu().numpy())

            partial_result = th.tensor(partial_result) # (<=k, batch_size, 1)

            assert len(partial_result.shape) == 3
            assert partial_result.shape[0] == _k # neighbours
            assert partial_result.shape[1] == batch_size
            assert partial_result.shape[2] == 1 # Q(s,a)

            partial_result = partial_result.reshape((batch_size, _k, 1))
            critic_output = partial_result

        if not deterministic and self.apply_rws_inference:
            result = self.roulette_wheel_selection(knn, critic_output)
            result = th.tensor(result).to(self.device)
        else:
            argmax_idx = np.argmax(critic_output.detach().cpu().numpy(), axis=1)

            assert len(argmax_idx.shape) == 2
            assert argmax_idx.shape[0] == batch_size
            assert argmax_idx.shape[1] == 1

            for idx, _argmax_idx in enumerate(argmax_idx):
                _argmax_idx = _argmax_idx[0]

                if isinstance(self.k, int):
                    assert _argmax_idx < self.k

                result.append(knn[idx][_argmax_idx])

            result = th.stack(result, dim=0).to(self.device)

            assert len(result.shape) == 2
            assert result.shape[0] == batch_size
            assert result.shape[1] == knn.shape[2]

        return result

class CnnPolicy(TD3Policy):
    """
    Policy class (with both actor and critic) for TD3.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )


class MultiInputPolicy(TD3Policy):
    """
    Policy class (with both actor and critic) for TD3 to be used with Dict observation spaces.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )
