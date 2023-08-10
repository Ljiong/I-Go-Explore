from modules.agents import REGISTRY as agent_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
class ICMMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self.args.state_shape
        # action_shape = self._get_action_shape(scheme)

        # Feature function
        self._build_agents(input_shape,args.n_actions)

        self.agent_output_type = args.agent_output_type


        self.hidden_states = None


    #def forward(self, ep_batch, t,state, next_state, action_long, test_mode=False):
    def forward(self,ep_batch, state, next_state, action_long, test_mode=False):
        # agent_inputs = self._build_inputs(ep_batch, t)
        # actions = ep_batch["actions"][:, :]
        real_next_state_feature, pred_next_state_feature, pred_action_logit = self.agent(state, next_state, action_long)
        
        return real_next_state_feature, pred_next_state_feature, pred_action_logit

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def to(self, *args, **kwargs):
        self.agent.to(*args, **kwargs)

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape, action_shape):
        # self.agent = agent_REGISTRY["rnd_history"](input_shape, self.args)
        self.agent = agent_REGISTRY[self.args.reward_agent](input_shape, action_shape, self.args)
       

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["state"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
    
    def _get_action_shape(self, scheme):
        action_shape = scheme["actions"]["vshape"][0]
        # NOT SURE
        if self.args.obs_agent_id:
            action_shape += self.n_agents
        return action_shape