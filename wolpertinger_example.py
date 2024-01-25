import gymnasium as gym
import numpy as np

from stable_baselines3 import TD3, DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import faiss

env = gym.make("Pendulum-v1", render_mode="rgb_array")

n_actions = env.action_space.shape[-1]
faiss_index = faiss.IndexFlatL2(n_actions)
k = 0.5
#discretized_action_space = np.arange(-2., 2., 0.1).astype(np.float32) # Discretized environment
discretized_action_space = np.arange(-2., 2., 0.5).astype(np.float32) # Discretized environment
#discretized_action_space = np.arange(-2., 2., 1.).astype(np.float32) # Discretized environment
discretized_action_space = discretized_action_space.reshape((discretized_action_space.shape[0], 1))

faiss_index.add(discretized_action_space)

def retrieve_embeddings(embedding, _k):
    assert len(embedding.shape) == 2
    assert embedding.shape[1] == n_actions

    if isinstance(_k, float):
        _k = max(1, int(discretized_action_space.shape[0] * _k + 0.5))

    result = []
    D, I = faiss_index.search(embedding, _k)

    for i in I:
        result.append([])
        for i2 in i:
            result[-1].append(discretized_action_space[i2])

    result = np.array(result)

    return result

# The noise objects for TD3
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

#model = TD3( # Works, but WolpertingerPolicy was designed with DDPG in mind
model = DDPG(
    "WolpertingerPolicy",
    env,
    action_noise=action_noise,
    verbose=1,
    policy_kwargs={
        "callback_retrieve_knn": lambda e, k: retrieve_embeddings(e, k * 2),
        "callback_retrieve_knn_training": retrieve_embeddings,
        "k": k,
    },
)

model.learn(total_timesteps=10000, log_interval=10)
#model.save("td3_pendulum")
vec_env = model.get_env()

#del model # remove to demonstrate saving and loading

#model = TD3.load("td3_pendulum")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
