import torch as th
import torch.nn.functional as F
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import multiprocessing
from envs import REGISTRY as env_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from components.episode_buffer import EpisodeBatch
from functools import partial
from multiprocessing import Pipe, Process
import gym
from copy import copy,deepcopy
from gym.wrappers import RenderCollection

env_name = "rware:rware-small-4ag-v1"
downscale_features = (8,11,8)

def numberOfSetBits(i):
    i = i - ((i >> 1) & 0x55555555)
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333)
    return (((i + (i >> 4) & 0xF0F0F0F) * 0x1010101) & 0xffffffff) >> 24


def convert_score(e):
    # TODO: this doesn't work when actual score is used!! Fix?
    if isinstance(e, tuple):
        return len(e)
    return numberOfSetBits(e)

class Cell():
    # item & archive
    
    def __init__(self, idx, restore, frame, key, score=-np.inf, traj_len=np.inf):
    
        self.visits = 0
        
        self.restore = restore
        self.idx = idx
        self.key = key
        self.score = score
        self.traj_len = traj_len
        self.frame = frame
        self.cells = {}


class Archive():
    def __init__(self):
        # key | cell
        self.cells = {}
        
    def __iter__(self):
        return iter(self.cells)
    
    def init_archive(self, start_info):
        self.cells = {}
        # start cell
        self.cells[start_info[2]] = Cell(start_info[3],start_info[0],start_info[1],
                                         start_info[2], score=0, traj_len=0)
        # DONE cell
        self.cells[None] = Cell(start_info[3]+1, None, None, None)

class CellSeletor():
    # select starting cells
    
    def __init__(self, archive):
        self.archive = archive

    def get_pos_weight(self, cell, possible_scores):
        return 1 + 0.5 * 1 / np.sqrt(len(possible_scores) - possible_scores.index(cell.score))
        
    def select_cells(self, amount):
        keys = []
        visited_weights = []
        pos_weights = []
        for key in self.archive.cells:
            if key == None: # done cell
                visited_weights.append(0.0)
            else:
                visited_weights.append(1/(np.sqrt(self.archive.cells[key].visits)+1))
            keys.append(key)

        possible_scores = sorted(set(self.archive.cells[k].score for k in keys))
        total_weights = []
        for key in keys:
            if key == None:
                pos_weights.append(0.0)
            else:
                pos_weights.append(self.get_pos_weight(self.archive.cells[key],possible_scores))
        total_weights = np.array(visited_weights) + np.array(pos_weights)
        # print(total_weights,total_weights/np.sum(total_weights))

        # test_key = keys[0]

        # print(keys)
        # print(possible_scores)
        # print(self.archive.cells[test_key].score)
        # print(possible_scores.index(self.archive.cells[test_key].score))
        # print(self.get_pos_weight(self.archive.cells[test_key],possible_scores))
            
        # indexes = np.random.choice(range(len(visited_weights)),size=amount,p=visited_weights/np.sum(visited_weights))
        indexes = np.random.choice(range(len(total_weights)),size=amount,p=total_weights/np.sum(total_weights))
        # print(indexes)
        
        selected_cells = []

        for i in indexes:
            if self.archive.cells[keys[i]].key != None:
                selected_cells.append(self.archive.cells[keys[i]])
        return selected_cells
            
    
class GoExploreRunner():
    # agent env loop
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit

        self.t = 0

        self.t_env = 0
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac, buffer):
        # self.batch = self.new_batch()

        # # Reset the envs
        # for parent_conn in self.parent_conns:
        #     parent_conn.send(("reset", None))

        self.pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": []
        }

        self.post_transition_data = {
                "reward": [],
                "terminated": []
        }
        # # Get the obs, state and avail_actions back
        # for parent_conn in self.parent_conns:
        #     data = parent_conn.recv()
        #     pre_transition_data["state"].append(data["state"])
        #     pre_transition_data["avail_actions"].append(data["avail_actions"])
        #     pre_transition_data["obs"].append(data["obs"])

        # self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        # self.env_steps_this_run = 0
        self.new_batch = partial(EpisodeBatch, scheme, groups, 1, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.buffer = buffer


    def get_env_info(self):
        # return self.env_info
        return self.env.get_env_info()
    
    def save_replay(self):
        # self.parent_conns[0].send(("save_replay", None))
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        # self.env.reset()
        self.t = 0


    def make_representation(self,frame):
        h, w, p = downscale_features
        greyscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_img = cv2.resize(greyscale_img, (h,w))
        resized_img_pix_threshold = ((resized_img/255.0) * p).reshape(-1).astype(int)
        # print(len(tuple(resized_img_pix_threshold)))
        return tuple(resized_img_pix_threshold)

    def go_explore(self, start_cell, test_mode, max_steps=50):
        self.reset()

        start_env = start_cell.restore
        # restore = copy.deepcopy(start_env)

        # print(start_env)

        # start_env.close()
        temp_env = copy(start_env)

        terminated = [False for _ in range(self.args.n_agents)]

        traj_elemtents = []
        counter = 0
        all_terminated = all(terminated)
        while not all_terminated and counter < max_steps:
            # print("check t: ",self.t,counter)
            traj_element = ()
            restore = copy(temp_env)
            # pic=temp_env.render(mode='rgb_array')
            # temp_env.close()
            # key = self.make_representation(pic)

            temp_state_org = tuple([temp_env._make_obs(agent) for agent in temp_env.agents])
            temp_state_tuple = tuple(map(tuple, temp_state_org))
            key = abs(hash(temp_state_tuple))

            temp_state = np.array(temp_state_org).reshape(1,284)
            # print("CHECK AGENTS NUM:", temp_state.shape)
            temp_state = th.Tensor(temp_state)
            # print("CHECK AGENTS NUM:", temp_state.shape)

            temp_avl_actions = temp_env.action_space.sample()
            # print(temp_avl_actions)
            temp_avl_actions = th.tensor(temp_avl_actions)
            # print(temp_avl_actions)
            temp_avl_actions = F.one_hot(temp_avl_actions, num_classes = 5)
            # print(temp_avl_actions.shape, temp_avl_actions)

            pre_transition_data = {
                "state": temp_state,
                "avail_actions": temp_avl_actions,
            }
            self.batch.update(pre_transition_data, ts=self.t)

            # collect data
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env)
            # print(actions.shape)
            actions = th.squeeze(actions)
            # print(actions)
            actions = actions.cpu().detach().tolist()
            # print(actions)

            state, reward, terminated, env_info = temp_env.step(actions)

            state = np.array(state)
            pre_transition_data = {
                "obs": th.Tensor(state)
            }
            self.batch.update(pre_transition_data, ts=self.t)

            reward = np.array([[sum(reward)]])
            reward = th.Tensor(reward)
            terminated = np.array([[all(terminated)]])
            terminated = th.Tensor(terminated)
            post_transition_data = {
                "reward": reward,
                "terminated": terminated
            }
            self.batch.update(post_transition_data, ts=self.t)
            

            traj_element = (key,
                            state,
                            actions[0],
                            reward,
                            terminated,
                            restore)
            
            # save data
            traj_elemtents.append(traj_element)
            
            self.t += 1
            counter += 1
            all_terminated = all(terminated)
            if all_terminated:
                print("TEM")
            if self.t > 500 and not test_mode:
                self.buffer.insert_episode_batch(self.batch)
                self.reset()
            # temp_env.close()
        if not test_mode:
            self.buffer.insert_episode_batch(self.batch)
        return traj_elemtents


    def get_trajactory(self, start_cell, test_mode): 
        traj = self.go_explore(start_cell,test_mode,50)
        return traj

    # handle overlap conflict in archive
    def better_cell_than_current(self, current_cell, new_score, new_traj_len):
        return ((current_cell.score < new_score) or 
            (current_cell.score == new_score and current_cell.traj_len > new_traj_len))

    def run(self, env, max_steps, test_mode = False):    
        # init cell archive
        idx_counter = 0

        #render mode:'rgb_array'
        if self.args.just_go_explore:
            env_tmp = gym.make(env_name,max_steps=30000,max_inactivity_steps=30000).unwrapped
            env_tmp.reset()
        else:
            env_tmp = env.original_env.env
        # print(env_tmp)

        state_s_org = tuple([env_tmp._make_obs(agent) for agent in env_tmp.agents])
        # print(state_s_org)
        state_s_tuple = tuple(map(tuple, state_s_org))
        # print(state_s_tuple)
        start_s = np.array(state_s_org).reshape(1,284)
        start_s = th.Tensor(start_s)

        start_restore = deepcopy(env_tmp)

        # start_s_pic = env_tmp.render(mode='rgb_array')

        start_cell_info = [start_restore, start_s, abs(hash(state_s_tuple)), idx_counter]
        # start_cell_info = [start_restore, start_s, self.make_representation(start_s_pic), idx_counter]
        # env_tmp.close()

        archive = Archive()
        archive.init_archive(start_cell_info)
        idx_counter += 1
        
        # init selector
        selector = CellSeletor(archive)

        best_score = -np.inf
        
        steps = 0
        bs = 2
        while steps < max_steps:
            result = []
            # get data
            start_cells = selector.select_cells(bs)

            for i in range(bs):
                start_cell = start_cells[i]
                # print(start_cell)
                traj_res = self.get_trajactory(start_cell,test_mode)
                # print("length of traj_res:",len(traj_res))
                result.append(traj_res)

            # print(len(result))

            # Iterate all generated trajs
            for traj, start_cell in zip(result,start_cells):
                steps += len(traj)
                # print("Through traj:",steps)

                # if not test_mode:
                #     self.t_env += steps
                
                # compute score and traj len for current pos
                cur_score = start_cell.score
                cur_traj_len = start_cell.traj_len
                # print("cur_score:", cur_score)

                seen_keys = []
                
                for j,traj_element in enumerate(traj):
                    # print(len(traj_element))
                    key, frame, action, reward, done, restore = traj_element

                    if done:
                        key = None
                        restore = None

                    cur_score += sum(reward)
                    cur_traj_len += 1

                    if key in archive.cells: # replace cell or not
                        new_is_better = self.better_cell_than_current(archive.cells[key], cur_score, cur_traj_len)

                        if new_is_better:
                            # transform existing cell
                            cell = archive.cells[key]
                            cell.visits = 0
                            cell.restore = restore
                            cell.score = cur_score
                            cell.traj_len = cur_traj_len
                            cell.frame = frame
                            cell.idx = idx_counter
                    
                            idx_counter += 1
                            # print("Found new cell:", idx_counter)

                    else: # add new cell
                        new_cell = Cell(idx_counter, restore, frame, key, score=cur_score, traj_len=cur_traj_len)
                        archive.cells[key] = new_cell
                        idx_counter += 1
                    
                    if key not in seen_keys:
                        archive.cells[key].visits += 1
                        seen_keys.append(key)


                    if cur_score > best_score:
                        best_score = cur_score
                    # if cur_traj_len < best_traj:
                    #     best_traj = cur_traj_len
                # print(best_score)
            self.t_env += steps
            # print(self.t_env, steps, best_score.item())
            # self.logger.log_stat("go_explore_return", np.array(best_score).item(), self.t_env)
            # self.logger.log_stat("cell_counter", np.array(len(archive.cells)).item(), self.t_env)
            # self.logger.log_stat("traj_length", np.array(cur_traj_len).item(), self.t_env)



        # print("Found cell:", idx_counter)
        # if not test_mode:
        #     self.t_env += steps
        # print("t_env:",self.t_env)
        return best_score,len(archive.cells),cur_traj_len
