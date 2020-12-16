import numpy as np
from RLlib.ppo_github_2.utils.utils import *
from RLlib.ppo_github_2.hparams import HyperParams as hp


def get_gae(rewards, masks, values):
    rewards = torch.tensor(rewards)
    masks = torch.tensor(masks)
    returns = torch.zeros_like(rewards)
    advants = torch.zeros_like(rewards)

    running_returns = 0
    previous_value = 0
    running_advants = 0

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + hp.gamma * running_returns * masks[t]
        running_tderror = rewards[t] + hp.gamma * previous_value * masks[t] - values.data[t]
        running_advants = running_tderror + hp.gamma * hp.lamda * running_advants * masks[t]

        returns[t] = running_returns
        previous_value = values.data[t]
        advants[t] = running_advants

    advants = (advants - advants.mean()) / advants.std()
    return returns, advants


def surrogate_loss(actor, advants, states, old_policy, actions, index):
    mu, std, logstd = actor(states)
    new_policy = log_density(actions, mu, std, logstd)
    old_policy = old_policy[index]

    ratio = torch.exp(new_policy - old_policy)
    surrogate = ratio * advants
    return surrogate, ratio


def train_model(actor, critic, memory, actor_optim, critic_optim):
    memory = np.array(memory)
    n_agents = memory.shape[0]
    states = None
    actions = None
    rewards = None
    masks = None
    returns = torch.tensor([])
    advants = torch.tensor([])
    for idx in range(n_agents):
        single_states = np.vstack(memory[idx][:, 0])
        single_actions = memory[idx][:, 1]
        single_rewards = np.array(memory[idx][:, 2], dtype=np.float32)
        single_masks = np.array(memory[idx][:, 3], dtype=np.float32)
        values = critic(torch.tensor(single_states, dtype=torch.float32))

    # ----------------------------
    # step 1: get returns and GAEs and log probability of old policy
        single_returns, single_advants = get_gae(single_rewards, single_masks, values)

        if states is None:
            states = single_states
            actions = single_actions
            rewards = single_rewards
            masks = single_masks
        else:
            states = np.vstack((states, single_states))
            actions = np.hstack((actions, single_actions))
            rewards = np.hstack((rewards, single_rewards))
            masks = np.hstack((masks, single_masks))
        returns = torch.cat((returns, single_returns))
        advants = torch.cat((advants, single_advants))

    mu, std, logstd = actor(torch.tensor(states, dtype=torch.float))
    old_policy = log_density(torch.tensor(list(actions)), mu, std, logstd)
    old_values = critic(torch.tensor(states, dtype=torch.float))

    criterion = torch.nn.MSELoss()
    n = len(states)
    arr = np.arange(n)

    # ----------------------------
    # step 2: get value loss and actor loss and update actor & critic
    for epoch in range(5):
        np.random.shuffle(arr)
        for i in range(32):
            batch_index = arr[hp.batch_size * i: hp.batch_size * (i + 1)]
            batch_index = torch.tensor(batch_index, dtype=torch.long)
            inputs = torch.tensor(states, dtype=torch.float)[batch_index]
            returns_samples = returns.unsqueeze(1)[batch_index]
            advants_samples = advants.unsqueeze(1)[batch_index]
            actions_samples = torch.tensor(list(actions))[batch_index]
            oldvalue_samples = old_values[batch_index].detach()

            loss, ratio = surrogate_loss(actor, advants_samples, inputs,
                                         old_policy.detach(), actions_samples,
                                         batch_index)

            values = critic(inputs)
            clipped_values = oldvalue_samples + torch.clamp(values - oldvalue_samples,
                                                            -hp.clip_param,
                                                            hp.clip_param)
            critic_loss1 = criterion(clipped_values, returns_samples)
            critic_loss2 = criterion(values, returns_samples)
            critic_loss = torch.max(critic_loss1, critic_loss2).mean()

            clipped_ratio = torch.clamp(ratio,
                                        1.0 - hp.clip_param,
                                        1.0 + hp.clip_param)
            clipped_loss = clipped_ratio * advants_samples
            actor_loss = -torch.min(loss, clipped_loss).mean()

            loss = actor_loss + 0.5 * critic_loss

            critic_optim.zero_grad()
            loss.backward(retain_graph=True)
            critic_optim.step()

            actor_optim.zero_grad()
            loss.backward()
            actor_optim.step()
