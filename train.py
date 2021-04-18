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

def main():
    parsed_args = parse_command_line_args(sys.argv[1:])
    num_last_frames = 4
    env = create_snake_environment(parsed_args.level)

    model = nn.Sequential(
        nn.Conv2d(num_last_frames, 16, 3),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(1152, 256),
        nn.ReLU(),
        nn.Linear(256, env.num_actions),
    )

    agent = DeepQNetworkAgent(
        model=model,
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
