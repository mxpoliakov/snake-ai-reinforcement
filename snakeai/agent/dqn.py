import collections
import numpy as np

import torch
import torch.nn as nn

from snakeai.agent import AgentBase
from snakeai.utils.memory import ExperienceReplay


class DeepQNetworkAgent(AgentBase):
    """ Represents a Snake agent powered by DQN with experience replay. """

    def __init__(
        self, model, env_shape, num_actions, num_last_frames=4, memory_size=1000
    ):
        """
        Create a new DQN-based agent.

        Args:
            model: a DQN model.
            env_shape (int, int): shape of the environment.
            num_actions (int): number of actions.
            num_last_frames (int): the number of last frames the agent will consider.
            memory_size (int): memory size limit for experience replay (-1 for unlimited).
        """
        self.model = model
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self.model.parameters())

        self.num_last_frames = num_last_frames
        self.memory = ExperienceReplay(
            (num_last_frames,) + env_shape, num_actions, memory_size
        )
        self.frames = None

    def begin_episode(self):
        """ Reset the agent for a new episode. """
        self.frames = None

    def get_last_frames(self, observation):
        """
        Get the pixels of the last `num_last_frames` observations, the current frame being the last.

        Args:
            observation: observation at the current timestep.

        Returns:
            Observations for the last `num_last_frames` frames.
        """
        frame = observation
        if self.frames is None:
            self.frames = collections.deque([frame] * self.num_last_frames)
        else:
            self.frames.append(frame)
            self.frames.popleft()
        return np.expand_dims(self.frames, 0)

    def train(
        self,
        env,
        num_episodes=1000,
        batch_size=50,
        discount_factor=0.9,
        checkpoint_freq=None,
        exploration_range=(1.0, 0.1),
        exploration_phase_size=0.5,
    ):
        """
        Train the agent to perform well in the given Snake environment.

        Args:
            env:
                an instance of Snake environment.
            num_episodes (int):
                the number of episodes to run during the training.
            batch_size (int):
                the size of the learning sample for experience replay.
            discount_factor (float):
                discount factor (gamma) for computing the value function.
            checkpoint_freq (int):
                the number of episodes after which a new model checkpoint will be created.
            exploration_range (tuple):
                a (max, min) range specifying how the exploration rate should decay over time.
            exploration_phase_size (float):
                the percentage of the training process at which
                the exploration rate should reach its minimum.
        """

        # Calculate the constant exploration decay speed for each episode.
        max_exploration_rate, min_exploration_rate = exploration_range
        exploration_decay = (max_exploration_rate - min_exploration_rate) / (
            num_episodes * exploration_phase_size
        )
        exploration_rate = max_exploration_rate

        for episode in range(num_episodes):
            # Reset the environment for the new episode.
            timestep = env.new_episode()
            self.begin_episode()
            game_over = False
            loss = 0.0

            # Observe the initial state.
            state = self.get_last_frames(timestep.observation)
            while not game_over:
                if np.random.random() < exploration_rate:
                    # Explore: take a random action.
                    action = np.random.randint(env.num_actions)
                else:
                    # Exploit: take the best known action for this state.
                    with torch.no_grad():
                        q = self.model(torch.Tensor(state))
                    action = np.argmax(q[0]).item()

                # Act on the environment.
                env.choose_action(action)
                timestep = env.timestep()

                # Remember a new piece of experience.
                reward = timestep.reward
                state_next = self.get_last_frames(timestep.observation)
                game_over = timestep.is_episode_end
                experience_item = [state, action, reward, state_next, game_over]
                self.memory.remember(*experience_item)
                state = state_next

                # Sample a random batch from experience.
                batch = self.memory.get_batch(
                    model=self.model,
                    batch_size=batch_size,
                    discount_factor=discount_factor,
                )
                # Learn on the batch.
                if batch:
                    inputs, targets = batch
                    predictions = self.model(torch.Tensor(inputs))
                    batch_loss = self.loss_fn(predictions, torch.Tensor(targets))
                    # Backpropagation
                    self.optimizer.zero_grad()
                    batch_loss.backward()
                    self.optimizer.step()
                    loss += batch_loss

            if checkpoint_freq and (episode % checkpoint_freq) == 0:
                torch.save(self.model, f"dqn-{episode:08d}.model")

            if exploration_rate > min_exploration_rate:
                exploration_rate -= exploration_decay

            summary = (
                "Episode {:5d}/{:5d} | Loss {:8.4f} | Exploration {:.2f} | "
                + "Fruits {:2d} | Timesteps {:4d} | Total Reward {:4d}"
            )
            print(
                summary.format(
                    episode + 1,
                    num_episodes,
                    loss,
                    exploration_rate,
                    env.stats.fruits_eaten,
                    env.stats.timesteps_survived,
                    env.stats.sum_episode_rewards,
                )
            )

        torch.save(self.model, "dqn-final.model")

    def act(self, observation, reward):
        """
        Choose the next action to take.

        Args:
            observation: observable state for the current timestep.
            reward: reward received at the beginning of the current timestep.

        Returns:
            The index of the action to take next.
        """
        state = self.get_last_frames(observation)
        with torch.no_grad():
            q = self.model(torch.Tensor(state))
        action = np.argmax(q[0]).item()
        return action
