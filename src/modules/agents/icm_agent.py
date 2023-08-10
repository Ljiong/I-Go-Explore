import torch.nn as nn
import torch.nn.functional as F
import torch as th
from modules.encoder import FCEncoder, ConvEncoder
import copy

    
class ICMAgent(nn.Module):
    def __init__(self, input_shape, action_shape, args):
        super(ICMAgent, self).__init__()
        self.args = args
        self.device = "cuda" if args.use_cuda else "cpu"
        self.n_agents = args.n_agents

        # print("CHECK input_shape:", input_shape)
        hidden_size_list = [64, 64, 128]
        if isinstance(input_shape, int) or len(input_shape) == 1:
            # print("In ICM-AGNET input shape:",input_shape)
            self.feature = FCEncoder(input_shape, hidden_size_list).to(self.device)
        elif len(input_shape) == 3:
            self.feature = ConvEncoder(input_shape, hidden_size_list).to(self.device)
        else:
            raise KeyError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own ICM model".
                format(input_shape)
            )
        feature_output = hidden_size_list[-1]

        self.inverse_net = nn.Sequential(nn.Linear(feature_output * 2, 512), nn.ReLU(), nn.Linear(512, action_shape * self.n_agents)).to(self.device)
        self.residual = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(action_shape * self.n_agents + 512, 512),
                    nn.LeakyReLU(),
                    nn.Linear(512, 512),
                ) for _ in range(8)
            ]
        ).to(self.device)
        self.forward_net_1 = nn.Sequential(nn.Linear(action_shape * self.n_agents + feature_output, 512), nn.LeakyReLU()).to(self.device)
        # print(next(self.forward_net_1.parameters()).device)
        self.forward_net_2 = nn.Linear(action_shape * self.n_agents + 512, feature_output).to(self.device)
        # print(next(self.forward_net_2.parameters()).device)



    def init_hidden(self):
        # make hidden states on same device as model
        return self.forward_net_1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, state, next_state, action_long):
        # x = F.relu(self.fc1(inputs))
        # h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        # if self.args.use_rnn:
        #     h = self.rnn(x, h_in)
        # else:
        #     h = F.relu(self.rnn(x))
        # q = self.fc2(h)
        # return q, h
        action = nn.functional.one_hot(action_long, num_classes=self.args.n_actions).to(self.device)
        # print("In ICM-AGNET state:",state.shape)
        encode_state = self.feature(state).to(self.device)
        # print("encode_state: ",encode_state.shape)  # 输出：
        encode_next_state = self.feature(next_state).to(self.device)
        # print("encode_next_state: ",encode_next_state.shape)  # 输出：
        # get pred action logit
        concat_state = th.cat((encode_state, encode_next_state), 2).to(self.device)
        # print(concat_state.shape)  # 输出：
        # print("CHECK_________________________")
        pred_action_logit = self.inverse_net(concat_state).to(self.device)
        # print(pred_action_logit.shape)  # 输出：
        # ---------------------

        # get pred next state

        action = th.squeeze(action).to(self.device)
        # print("In ICM_agent action before:", action.shape)
        tmp_action = th.Tensor([]).to(self.device)
        for i in range(1,self.args.n_agents,2):
            tmp = th.Tensor([])
            tmp = th.cat((action[:,:,i-1],action[:,:,i]),2).to(self.device)
            tmp_action = th.cat((tmp_action,tmp),2)

        action = copy.deepcopy(tmp_action)
        # print("In ICM_agent action after:", action.shape)
        pred_next_state_feature_orig = th.cat((encode_state, action), 2)
        # print(pred_next_state_feature_orig.shape)  # 输出：
        pred_next_state_feature_orig = self.forward_net_1(pred_next_state_feature_orig)
        # print(pred_action_logit.shape)  # 输出：

        # residual
        for i in range(4):
            # print("LAST STEP IN ICM1:",pred_next_state_feature_orig.shape)
            # print("LAST STEP IN ICM2:",action.shape)
            pred_next_state_feature = self.residual[i * 2](th.cat((pred_next_state_feature_orig, action), 2))
            pred_next_state_feature_orig = self.residual[i * 2 + 1](
                th.cat((pred_next_state_feature, action), 2)
            ) + pred_next_state_feature_orig
        pred_next_state_feature = self.forward_net_2(th.cat((pred_next_state_feature_orig, action), 2))
        real_next_state_feature = encode_next_state
        return real_next_state_feature, pred_next_state_feature, pred_action_logit

