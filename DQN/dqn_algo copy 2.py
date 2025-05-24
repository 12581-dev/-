"""
修改后的 DQN 代码，增加了无人机轨迹记录功能
"""
import numpy as np
import tensorflow as tf
from UAV_env import UAVEnv
import time
from state_normalization import StateNormalization
import matplotlib.pyplot as plt
import os
import json

# 强制使用 CPU 以避免 cublas 错误
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

MAX_EPISODES = 500  # 增加到1000个回合
MEMORY_CAPACITY = 20000
BATCH_SIZE = 32  # 减小批次大小以降低内存使用

# 配置 TensorFlow 减少内存使用
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.99,
            replace_target_iter=200,
            memory_size=MEMORY_CAPACITY,
            batch_size=BATCH_SIZE,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon = 0.1

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((memory_size, n_features * 2 + 2), dtype=np.float32)

        # consist of [target_net, evaluate_net]
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session(config=config)

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 200, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            e3 = tf.layers.dense(e1, 100, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e3')
            self.q_eval = tf.layers.dense(e3, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 200, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            t3 = tf.layers.dense(t1, 100, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t3')
            self.q_next = tf.layers.dense(t3, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t4')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a], [r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.a: batch_memory[:, self.n_features].astype(int),
                self.r: batch_memory[:, self.n_features + 1],
                self.s_: batch_memory[:, -self.n_features:],
            })

        self.cost_his.append(cost)
        self.learn_step_counter += 1
        
        # 逐步降低探索率以实现更好的探索/利用平衡
        if self.epsilon > 0.05:
            self.epsilon = self.epsilon * 0.9999

    def save_model(self, path="./saved_model/"):
        """保存模型到磁盘"""
        if not os.path.exists(path):
            os.makedirs(path)
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, path + "model.ckpt")
        print("Model saved in path: %s" % save_path)
        
    def load_model(self, path="./saved_model/model.ckpt"):
        """从磁盘加载模型"""
        saver = tf.train.Saver()
        saver.restore(self.sess, path)
        print("Model restored from: %s" % path)


def extract_uav_position(state):
    """
    从状态中提取无人机位置
    注意：需要根据您的环境定义来调整这个函数
    假设状态包含无人机的位置信息在前三个元素
    """
    # 假设状态的前三个元素分别是x, y, z坐标
    # 如果状态包含的位置信息在其他位置，请相应调整
    return state[:3] if len(state) >= 3 else state  # 返回假定的位置信息


if __name__ == '__main__':
    try:
        env = UAVEnv()
        Normal = StateNormalization()  # 输入状态归一化
        
        # 设置随机种子以确保可重复性
        np.random.seed(1)
        tf.set_random_seed(1)
        
        s_dim = env.state_dim
        n_actions = env.n_actions
        
        DQN = DeepQNetwork(n_actions, s_dim, output_graph=False)
        
        t1 = time.time()
        ep_reward_list = []
        MAX_EP_STEPS = env.slot_num
        
        # 创建带时间戳的日志文件
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        logfile = f'training_log_{timestamp}.txt'
        
        # 创建轨迹保存目录
        trajectory_dir = f'uav_trajectories_{timestamp}'
        if not os.path.exists(trajectory_dir):
            os.makedirs(trajectory_dir)
        
        # 跟踪最佳性能以保存模型
        best_reward = -float('inf')
        
        for i in range(MAX_EPISODES):
            # 初始观察
            s = env.reset()
            ep_reward = 0
            j = 0
            
            # 创建此回合的轨迹记录
            trajectory = []
            
            # 记录初始位置
            position = extract_uav_position(s)
            trajectory.append({
                'step': j,
                'position': position.tolist() if isinstance(position, np.ndarray) else position,
                'state': s.tolist() if isinstance(s, np.ndarray) else s
            })
            
            while j < MAX_EP_STEPS:
                # RL基于观察选择动作
                a = DQN.choose_action(Normal.state_normal(s))
                
                # RL执行动作并获取下一个观察和奖励
                s_, r, is_terminal, step_redo, reset_offload_ratio = env.step(a)
                
                if step_redo:
                    continue
                    
                if reset_offload_ratio:
                    # 卸载比率重新设置为0
                    t1_offset = a % 11
                    a = a - t1_offset
                    
                # 记录当前位置到轨迹
                position = extract_uav_position(s_)
                trajectory.append({
                    'step': j + 1,
                    'position': position.tolist() if isinstance(position, np.ndarray) else position,
                    'action': int(a),
                    'reward': float(r),
                    'state': s_.tolist() if isinstance(s_, np.ndarray) else s_
                })
                
                # 存储归一化状态
                DQN.store_transition(Normal.state_normal(s), a, r, Normal.state_normal(s_))

                # 训练
                if DQN.memory_counter > BATCH_SIZE * 5:  # 收集一些经验后开始训练
                    DQN.learn()

                # 更新观察
                s = s_
                ep_reward += r

                if j == MAX_EP_STEPS - 1 or is_terminal:
                    # 记录回合信息
                    log_msg = f'Episode: {i}, Steps: {j}, Reward: {ep_reward:.2f}, Explore: {DQN.epsilon:.3f}'
                    print(log_msg)
                    
                    # 保存到日志文件
                    with open(logfile, 'a') as f:
                        f.write(log_msg + '\n')
                    
                    ep_reward_list.append(ep_reward)
                    
                    # 如果是迄今为止的最佳表现，则保存模型
                    if ep_reward > best_reward:
                        best_reward = ep_reward
                        DQN.save_model("./best_model/")
                        
                        # 保存最佳回合的轨迹
                        with open(f'{trajectory_dir}/best_trajectory.json', 'w') as f:
                            json.dump(trajectory, f, indent=2)
                    
                    # 每个回合都保存轨迹
                    with open(f'{trajectory_dir}/trajectory_episode_{i}.json', 'w') as f:
                        json.dump(trajectory, f, indent=2)
                    
                    break

                j = j + 1
                
            # 每50个回合保存检查点
            if i % 50 == 0 and i > 0:
                DQN.save_model(f"./checkpoint_model_{i}/")
                
                # 创建中间图表
                plt.figure()
                plt.plot(ep_reward_list)
                plt.xlabel("Episode")
                plt.ylabel("Reward")
                plt.title(f"DQN Learning Progress (Episode {i})")
                plt.savefig(f"dqn_progress_{i}.png")
                plt.close()
                
                # 可视化最近一个回合的轨迹
                try:
                    # 提取位置坐标
                    positions = np.array([point['position'] for point in trajectory])
                    
                    if positions.shape[1] >= 3:  # 如果是3D轨迹
                        fig = plt.figure(figsize=(10, 8))
                        ax = fig.add_subplot(111, projection='3d')
                        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-')
                        ax.plot(positions[0, 0], positions[0, 1], positions[0, 2], 'go', markersize=10, label='Start')
                        ax.plot(positions[-1, 0], positions[-1, 1], positions[-1, 2], 'ro', markersize=10, label='End')
                        ax.set_xlabel('X')
                        ax.set_ylabel('Y')
                        ax.set_zlabel('Z')
                        ax.set_title(f'UAV 3D Trajectory (Episode {i})')
                        ax.legend()
                        plt.savefig(f"{trajectory_dir}/trajectory_3d_episode_{i}.png")
                        plt.close()
                    
                    elif positions.shape[1] >= 2:  # 如果是2D轨迹
                        plt.figure(figsize=(10, 8))
                        plt.plot(positions[:, 0], positions[:, 1], 'b-')
                        plt.plot(positions[0, 0], positions[0, 1], 'go', markersize=10, label='Start')
                        plt.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=10, label='End')
                        plt.xlabel('X')
                        plt.ylabel('Y')
                        plt.title(f'UAV 2D Trajectory (Episode {i})')
                        plt.grid(True)
                        plt.legend()
                        plt.savefig(f"{trajectory_dir}/trajectory_2d_episode_{i}.png")
                        plt.close()
                except Exception as e:
                    print(f"无法绘制轨迹图: {e}")

        print('运行时间: ', time.time() - t1)
        
        # 创建最终奖励图表
        plt.figure(figsize=(10, 6))
        plt.plot(ep_reward_list)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"DQN Learning Curve ({MAX_EPISODES} Episodes)")
        plt.savefig("dqn_final.png")
        plt.show()
        
        # 保存最终模型
        DQN.save_model("./final_model/")
        
        # 分析轨迹数据
        print("分析最佳轨迹数据...")
        try:
            with open(f'{trajectory_dir}/best_trajectory.json', 'r') as f:
                best_trajectory = json.load(f)
                
            # 提取位置坐标
            positions = np.array([point['position'] for point in best_trajectory])
            
            # 计算总路径长度
            path_length = 0
            for i in range(1, len(positions)):
                path_length += np.linalg.norm(positions[i] - positions[i-1])
                
            print(f"最佳轨迹总路径长度: {path_length:.2f}")
            
            # 创建最佳轨迹的可视化
            if positions.shape[1] >= 3:  # 如果是3D轨迹
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')
                ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-')
                ax.plot(positions[0, 0], positions[0, 1], positions[0, 2], 'go', markersize=10, label='Start')
                ax.plot(positions[-1, 0], positions[-1, 1], positions[-1, 2], 'ro', markersize=10, label='End')
                
                # 添加每个点的索引标签
                for i, (x, y, z) in enumerate(zip(positions[:, 0], positions[:, 1], positions[:, 2])):
                    if i % 5 == 0:  # 每5个点标注一次，避免标签过密
                        ax.text(x, y, z, f'{i}', fontsize=8)
                
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title('Best UAV 3D Trajectory')
                ax.legend()
                plt.savefig(f"{trajectory_dir}/best_trajectory_3d.png")
                plt.show()
                
            elif positions.shape[1] >= 2:  # 如果是2D轨迹
                plt.figure(figsize=(12, 10))
                plt.plot(positions[:, 0], positions[:, 1], 'b-')
                plt.plot(positions[0, 0], positions[0, 1], 'go', markersize=10, label='Start')
                plt.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=10, label='End')
                
                # 添加每个点的索引标签
                for i, (x, y) in enumerate(zip(positions[:, 0], positions[:, 1])):
                    if i % 5 == 0:  # 每5个点标注一次，避免标签过密
                        plt.text(x, y, f'{i}', fontsize=8)
                
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.title('Best UAV 2D Trajectory')
                plt.grid(True)
                plt.legend()
                plt.savefig(f"{trajectory_dir}/best_trajectory_2d.png")
                plt.show()
                
        except Exception as e:
            print(f"无法分析最佳轨迹: {e}")
            
    except Exception as e:
        print(f"发生错误: {e}")