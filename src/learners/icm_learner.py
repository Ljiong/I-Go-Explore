import copy
from components.episode_buffer import EpisodeBatch
import torch.nn as nn
import torch as th
from torch.optim import Adam
import torch.nn.functional as func
from controllers import REGISTRY as mac_REGISTRY
import numpy as np
import torch.optim as optim
import torch


class ICMLearner:
    def __init__(self, mac, scheme, logger, args, groups=None):
        self.args = args
        self.mac = copy.deepcopy(mac)
        self.logger = logger
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions

        self.params = list(self.mac.parameters())

        self.state_shape = scheme["state"]["vshape"]

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.ce = nn.CrossEntropyLoss(reduction="mean")
        self.forward_mse = nn.MSELoss(reduction='none')
        self.reverse_scale = 1
        self.res = nn.Softmax(dim=-1)
        self.opt = Adam(params=self.params, lr=args.icm_lr)
        self.device = "cuda" if args.use_cuda else "cpu"


    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int,save_buffer=False, imac=None, timac=None):
        # Get the relevant quantities
        bs = batch.batch_size
        max_t = batch.max_seq_length

        actions = batch["actions"][:, :-1]
        # print("CHECK actions shape",actions.shape)
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()#.to(self.device)
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        # next_states = batch["state"][: 1:]#.to(self.device)
        # states = batch["obs"][:,:]

        # check the idx from coma_learner(several batch)
        # print("obs shape:",batch["obs"].shape)
        # states = th.cat((batch["obs"][:,:,0],batch["obs"][:,:,1]),2)#.to(self.device)
        # for i in range(1,self.args.n_agents):
        #     states = th.cat((batch["obs"][:,:,0],batch["obs"][:,:,i]),2)

        tmp_agent = th.Tensor([]).to(self.device)
        for i in range(1,self.args.n_agents,2):
            tmp = th.Tensor([])
            # tmp = th.cat((batch["obs"][:,:,i-1],batch["obs"][:,:,i]),2).to(self.device)
            tmp = th.cat((batch["obs"][:,:,i-1],batch["obs"][:,:,i]),2).to(self.device)
            # print("IN ICM_learner check merge state:",tmp.shape)
            tmp_agent = th.cat((tmp_agent,tmp),2)
        states = copy.deepcopy(tmp_agent)

        # print("CHECK states before:",states.shape)
        next_states = states[:,1:]
        # print("CHECK next states:",next_states.shape)
        states = states[:,:-1]
        # print("CHECK obs:",batch["obs"].shape)
        # print("CHECK states after:",states.shape)

        # print((batch["obs"][0,1,0]))
        # print((batch["obs"][0,1,1]))
        # print(next_states[0][0])


        mac_out = {
            "real_next_state_feature":th.Tensor([]),
            "pred_next_state_feature":th.Tensor([]),
            "pred_action_logit":th.Tensor([])
            }

        rl_nxt_state, pred_nxt_state, pred_action \
                = self.mac.forward(batch, states, next_states, actions)
        # print("rl_nxt_state",rl_nxt_state)
        # print("pred_nxt_state",pred_nxt_state)
        mac_out["real_next_state_feature"]=rl_nxt_state
        mac_out["pred_next_state_feature"]=pred_nxt_state
        mac_out["pred_action_logit"]=pred_action

        # print("CHECK pred_action",pred_action)
        avail_actions = batch["avail_actions"][:, :-1]

        # change the style same as icm_agent
        # for i in range(1,self.args.n_agents):
        #     avail_actions = th.cat((avail_actions[:,:,0],avail_actions[:,:,i]),2)
        # print("CHECK avail_actions shape:", avail_actions.shape)

        # TEST for 4 agent
        # tmp_action = th.Tensor([]).to(self.device)
        # for i in range(1,self.args.n_agents,2):
        #     tmp = th.Tensor([])
        #     tmp = th.cat((actions[:,:,i-1],actions[:,:,i]),2).to(self.device)
        #     tmp_action = th.cat((tmp_action,tmp),2)

        # avail_actions = copy.deepcopy(tmp_action)
        # # print("CHECK avail_actions shape:", avail_actions.shape)

        # avail_actions = avail_actions.type(th.FloatTensor).to(self.device)
        # print("CHECK avail_actions:", avail_actions.shape)

        # Inverse action
        actions = th.squeeze(actions)
        inverse_action = th.Tensor([]).to(self.device)
        for i in range(1,self.args.n_agents,2):
            tmp = th.Tensor([])
            action1 = func.one_hot(actions[:,:,i-1],num_classes=self.n_actions)
            action2 = func.one_hot(actions[:,:,i],num_classes=self.n_actions)
            tmp = th.cat((action1,action2),2).to(self.device)
            inverse_action = th.cat((inverse_action,tmp),2)
        # inverse_action = np.array(inverse_action)
        # print("inverse_action shape:", inverse_action.shape)
        # print("Check pred actions:", pred_action.shape)

        inverse_loss = \
            self.ce(pred_action, inverse_action) #-621
        # print("inverse_loss",inverse_loss)
        forward_loss = \
            self.forward_mse(mac_out["pred_next_state_feature"], mac_out["real_next_state_feature"].detach()).mean()
        
        intrisic_rewards = th.Tensor([])
        res_icm = ()
        for i in range(bs):
            tmp = []
            for j in range(batch.max_seq_length - 1):
                tmp.append(self.forward_mse(mac_out["pred_next_state_feature"][i,j,:].detach(),\
                                            mac_out["real_next_state_feature"][i,j,:].detach()).mean())
                # tmp += (self.forward_mse(mac_out["pred_next_state_feature"][i,j,:].detach(),\
                #                          mac_out["real_next_state_feature"][i,j,:].detach()).mean(),)
            tmp = th.stack(tmp,dim=0)
            res_icm+=(tmp,)

        intrisic_rewards = th.stack(res_icm)
        intrisic_rewards = th.unsqueeze(intrisic_rewards,dim=2)
        # print("CHECK intrisic_rewards in icm_learner:",intrisic_rewards.shape)
        
        # self.logger.add_scalar('icm_reward/forward_loss', forward_loss, episode_num)
        # self.logger.add_scalar('icm_reward/inverse_loss', inverse_loss, episode_num)
        # action = th.argmax(self.res(mac_out["pred_action_logit"]), -1)
        # accuracy = th.sum(action == actions.squeeze(-1)).item() / actions.shape[0]
        # self.logger.add_scalar('icm_reward/action_accuracy', accuracy, episode_num)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            # print("CHECK forward_loss:", forward_loss.shape, forward_loss)
            self.logger.log_stat("icm_reward_forward_loss", forward_loss, t_env)
            self.logger.log_stat("icm_reward_inverse_loss", inverse_loss, t_env)
            self.log_stats_t = t_env
        

        loss = self.reverse_scale * inverse_loss + forward_loss
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return intrisic_rewards

    def cuda(self):
        self.mac.cuda()


    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.opt.state_dict(), "{}/icm_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.opt.load_state_dict(th.load("{}/icm_opt.th".format(path), map_location=lambda storage, loc: storage))
