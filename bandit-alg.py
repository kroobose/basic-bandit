import argparse
import random
import numpy as np
from numpy import exp, log, max, argmax
from numpy.random import binomial, randint, rand
from scipy.stats import beta
from arm import Arm
from prettytable import PrettyTable
from tqdm import tqdm

def arg_parse():
    parser = argparse.ArgumentParser(description='Bandit Algorithm')
    parser.add_argument('--arms', type=int, default=100, help='Number of arms')
    parser.add_argument('--T', type=int, default=1000, help='Number of iterations')
    parser.add_argument('--epsilon', type=float, default=0.2, help='epsilon in epsilon-greedy policy')
    parser.add_argument('--tau', type=float, default=0.1, help='tau in softmax policy')
    parser.add_argument('--policy', type=str, choices=['epsilon-greedy', 'softmax', 'ucb', 'thompson'], default='epsilon-greedy', help='Policy to use (epsilon-greedy, softmax, ucb, thompson)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--num_plays', type=int, default=100, help='Number of plays')
    args = parser.parse_args()

    return args

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def arms_reset(arms):
    for arm in arms:
        arm.reset()

class Bandit_alg:
    def __init__(self, arms):
        self.arms = arms
        self.reward = 0
        self.K = len(arms)

    def calc_success_ratio(self, arm) -> int:
        if arm.success + arm.fail == 0:
            return 0
        return arm.success / (arm.success + arm.fail)

    def selcet_arm_num(self, arms):
        avgs = [self.calc_success_ratio(arm) for arm in arms]
        return avgs.index(max(avgs))

    def epsilon_greedy_policy(self, epsilon, T=100):
        reward = self.reward
        K = self.K
        explore_num = int(round(epsilon*T/K))
        res_num = T - explore_num * K
        for arm_index in range(K):
            for _ in range(explore_num):
                reward += arms[arm_index].play()
        avgs = [self.calc_success_ratio(arm) for arm in arms]
        best_arm = avgs.index(max(avgs))
        for _ in range(res_num):
            reward += arms[best_arm].play()

        return reward

    def softmax_policy(self, T=100, tau=0.1):
        reward = self.reward
        K = self.K
        for _ in range(T):
            avgs = [self.calc_success_ratio(arm) for arm in arms]
            sum_avgs = sum([exp(avg/tau) for avg in avgs])
            probs = [exp(avg/tau)/sum_avgs for avg in avgs]
            cumulative_prob = np.cumsum(probs)
            rand = random.random()
            for idx, cum_prob in enumerate(cumulative_prob):
                if rand < cum_prob:
                    arm_index = idx
                    break
            reward += arms[arm_index].play()

        return reward

    def ucb_policy(self, T=100):
        reward = self.reward
        K = self.K
        for t in range(T):
            ucbs = [0 for _ in range(K)]
            avgs = [self.calc_success_ratio(arm) for arm in arms]
            for arm_index in range(K):
                if t == 0:
                    reward += arms[arm_index].play()
                    continue
                ucbs[arm_index] = avgs[arm_index] + (2 * log(T) / (arms[arm_index].success + arms[arm_index].fail))**0.5
            best_arm = ucbs.index(max(ucbs))
            reward += arms[best_arm].play()

        return reward

    def thompson_sampling_policy(self, T=100):
        reward = self.reward

        for _ in range(T):
            sampled_values = [
                beta.rvs(arm.success + 1, arm.fail + 1) for arm in self.arms
            ]
            chosen_arm = argmax(sampled_values)
            reward += self.arms[chosen_arm].play()

        return reward


if __name__ == "__main__":

    args = arg_parse()

    if args.seed:
        fix_seed(args.seed)
    arms = [Arm(p = rand()) for _ in range(0, args.arms)]
    bandit = Bandit_alg(arms)
    rewards = 0
    num_plays = args.num_plays
    for _ in tqdm(range(num_plays)):
        arms_reset(arms)
        match args.policy:
            case 'epsilon-greedy':
                reward = bandit.epsilon_greedy_policy(args.epsilon, args.T)

            case 'softmax':
                reward = bandit.softmax_policy(args.T, args.tau)

            case 'ucb':
                reward = bandit.ucb_policy(args.T)

            case 'thompson':
                reward = bandit.thompson_sampling_policy(args.T)

        rewards += reward


    print(f'result of {args.policy} policy')
    print(f'num of play: {num_plays}')
    print(f'average reward: {round(rewards/num_plays)}')

