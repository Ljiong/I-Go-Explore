from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import numpy as np


# This multi-agent controller shares parameters between agents
class GoExploreMAC:
    def __init__(self, scheme, groups, args, is_ICM = None):
        self.n_agents = args.n_agents
        self.args = args
        # input_shape = self._get_input_shape(scheme)
        action_shape = None
        # self._build_agents(input_shape,action_shape)
           
        # self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.go_action_selector](args)


    def select_actions(self, ep_batch, t_ep, t_env):
        # Only select actions for the selected batch elements in bs
        # print(ep_batch["avail_actions"].shape,t_ep)
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        # print("IN BSC CHECK avail_actions:", avail_actions.device)
        chosen_actions = self.action_selector.select_action(avail_actions)
        # print("IN BSC CHECK avail_actions:", chosen_actions.shape)
        # self.repeat_action = np.random.geometric(1 / self.mean_repeat)

        return chosen_actions

    # def forward(self, ep_batch, t, test_mode=False):
    #     agent_inputs = self._build_inputs(ep_batch, t)
    #     avail_actions = ep_batch["avail_actions"][:, t]
    #     agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

    #     # Softmax the agent outputs if they're policy logits
    #     if self.agent_output_type == "pi_logits":

    #         if getattr(self.args, "mask_before_softmax", True):
    #             # Make the logits for unavailable actions very negative to minimise their affect on the softmax
    #             reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
    #             agent_outs[reshaped_avail_actions == 0] = -1e10
    #         agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

    #     return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    # def init_hidden(self, batch_size):
    #     self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    # def parameters(self):
    #     return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    # def save_models(self, path):
    #     th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    # def load_models(self, path):
    #     self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    # def _build_agents(self, input_shape, action_shape = None):
    #     if not action_shape:
    #         self.agent = agent_REGISTRY[self.args.bs_agent](input_shape, self.args)
    #     else:
    #         self.agent = agent_REGISTRY[self.args.bs_agent](input_shape, action_shape, self.args)

    # def _build_inputs(self, batch, t):
    #     # Assumes homogenous agents with flat observations.
    #     # Other MACs might want to e.g. delegate building inputs to each agent
    #     bs = batch.batch_size
    #     inputs = []
    #     inputs.append(batch["obs"][:, t])  # b1av
    #     # print("CHECK t:",t)
    #     if self.args.obs_last_action:
    #         if t == 0:
    #             inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
    #         else:
    #             inputs.append(batch["actions_onehot"][:, t-1])
    #     if self.args.obs_agent_id:
    #         inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

    #     inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
    #     # print("IN BSMAC INPUT SHAPE:", inputs.shape)
    #     return inputs

    # def _get_input_shape(self, scheme):
    #     input_shape = scheme["obs"]["vshape"]
    #     if self.args.obs_last_action:
    #         input_shape += scheme["actions_onehot"]["vshape"][0]
    #     if self.args.obs_agent_id:
    #         input_shape += self.n_agents

    #     return input_shape
    
    # def _get_action_shape(self, scheme):
    #     action_shape = scheme["action"]["vshape"]
    #     # NOT SURE
    #     if self.args.obs_agent_id:
    #         action_shape += self.n_agents
    #     return action_shape
