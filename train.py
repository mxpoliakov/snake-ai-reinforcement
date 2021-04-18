#!/usr/bin/env python3

""" Front-end script for training a Snake agent. """

import json
import sys

import torch
import torch.nn as nn

from snakeai.agent import DeepQNetworkAgent
from snakeai.gameplay.environment import Environment
from snakeai.utils.cli import HelpOnFailArgumentParser


def parse_command_line_args(args):
    """ Parse command-line arguments and organize them into a single structured object. """

    parser = HelpOnFailArgumentParser(
        description="Snake AI training client.",
        epilog="Example: train.py --level 10x10.json --num-episodes 30000",
    )

    parser.add_argument(
        "--level",
        required=True,
        type=str,
        help="JSON file containing a level definition.",
    )
    parser.add_argument(
        "--num-episodes",
        required=True,
        type=int,
        default=30000,
        help="The number of episodes to run consecutively.",
    )

    return parser.parse_args(args)


def create_snake_environment(level_filename):
    """ Create a new Snake environment from the config file. """

    with open(level_filename) as cfg:
        env_config = json.load(cfg)

    return Environment(config=env_config, verbose=1)


class DQN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=16, kernel_size=3, stride=2
        )
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=num_actions)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    parsed_args = parse_command_line_args(sys.argv[1:])
    num_last_frames = 4
    env = create_snake_environment(parsed_args.level)

    agent = DeepQNetworkAgent(
        model=DQN(num_last_frames, env.num_actions),
        env_shape=env.observation_shape,
        num_actions=env.num_actions,
        memory_size=-1,
        num_last_frames=num_last_frames,
    )
    agent.train(
        env,
        batch_size=64,
        num_episodes=parsed_args.num_episodes,
        checkpoint_freq=parsed_args.num_episodes // 10,
        discount_factor=0.95,
    )


if __name__ == "__main__":
    main()
