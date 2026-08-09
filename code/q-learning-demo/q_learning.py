"""
Q-Learning Algorithm Implementation
การเขียนโปรแกรม Q-learning algorithm
"""

import numpy as np
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.1, 
                 discount_factor=0.95, epsilon=0.1):
        """
        สร้าง Q-Learning Agent
        
        Args:
            n_states (int): จำนวน states
            n_actions (int): จำนวน actions
            learning_rate (float): อัตราการเรียนรู้ (alpha)
            discount_factor (float): ปัจจัยลด (gamma)
            epsilon (float): อัตราการสำรวจ (exploration rate)
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # สร้าง Q-table
        self.q_table = np.zeros((n_states, n_actions))
        
        # สถิติการเรียนรู้
        self.episode_rewards = []
        self.episode_lengths = []
    
    def get_action(self, state, training=True):
        """
        เลือก action ตาม epsilon-greedy policy
        
        Args:
            state (int): state ปัจจุบัน
            training (bool): เป็นการฝึกหรือไม่
            
        Returns:
            int: action ที่เลือก
        """
        if training and random.random() < self.epsilon:
            # Exploration: เลือก action แบบสุ่ม
            return random.randint(0, self.n_actions - 1)
        else:
            # Exploitation: เลือก action ที่ดีที่สุด
            return np.argmax(self.q_table[state])
    
    def update_q_value(self, state, action, reward, next_state):
        """
        อัปเดต Q-value ตาม Q-learning formula
        
        Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        
        Args:
            state (int): state ปัจจุบัน
            action (int): action ที่ทำ
            reward (float): reward ที่ได้รับ
            next_state (int): state ถัดไป
        """
        # หา Q-value สูงสุดของ next_state
        max_next_q = np.max(self.q_table[next_state])
        
        # คำนวณ target Q-value
        target_q = reward + self.discount_factor * max_next_q
        
        # คำนวณ temporal difference error
        td_error = target_q - self.q_table[state, action]
        
        # อัปเดต Q-value
        self.q_table[state, action] += self.learning_rate * td_error
    
    def train_episode(self, env, max_steps=100):
        """
        ฝึก agent หนึ่ง episode
        
        Args:
            env: environment object
            max_steps (int): จำนวน steps สูงสุดต่อ episode
            
        Returns:
            tuple: (total_reward, steps)
        """
        state = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            # เลือก action
            action = self.get_action(state, training=True)
            
            # ทำ action
            next_state, reward, done = env.step(action)
            
            # อัปเดต Q-value
            self.update_q_value(state, action, reward, next_state)
            
            # อัปเดตสถิติ
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        return total_reward, steps
    
    def train(self, env, n_episodes=1000, verbose=True):
        """
        ฝึก agent หลาย episodes
        
        Args:
            env: environment object
            n_episodes (int): จำนวน episodes ที่จะฝึก
            verbose (bool): แสดงผลการฝึกหรือไม่
        """
        for episode in range(n_episodes):
            reward, steps = self.train_episode(env)
            
            self.episode_rewards.append(reward)
            self.episode_lengths.append(steps)
            
            # ลด epsilon เมื่อเวลาผ่านไป (epsilon decay)
            if episode > 0 and episode % 100 == 0:
                self.epsilon = max(0.01, self.epsilon * 0.95)
            
            if verbose and episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_steps = np.mean(self.episode_lengths[-100:])
                print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, "
                      f"Avg Steps = {avg_steps:.1f}, Epsilon = {self.epsilon:.3f}")
    
    def test_episode(self, env, max_steps=100, render=False):
        """
        ทดสอบ agent หนึ่ง episode (ไม่มีการเรียนรู้)
        
        Args:
            env: environment object
            max_steps (int): จำนวน steps สูงสุด
            render (bool): แสดงภาพหรือไม่
            
        Returns:
            tuple: (total_reward, steps, path)
        """
        state = env.reset()
        total_reward = 0
        steps = 0
        path = [env.state_to_pos(state)]
        
        for step in range(max_steps):
            # เลือก action แบบ greedy (ไม่มี exploration)
            action = self.get_action(state, training=False)
            
            # ทำ action
            next_state, reward, done = env.step(action)
            
            # อัปเดตสถิติ
            total_reward += reward
            steps += 1
            state = next_state
            path.append(env.state_to_pos(state))
            
            if done:
                break
        
        return total_reward, steps, path
    
    def get_policy(self):
        """
        สร้าง policy จาก Q-table
        
        Returns:
            numpy.ndarray: policy matrix
        """
        policy = np.zeros((self.n_states, self.n_actions))
        for state in range(self.n_states):
            best_action = np.argmax(self.q_table[state])
            policy[state, best_action] = 1.0
        return policy
    
    def get_value_function(self):
        """
        คำนวณ value function จาก Q-table
        
        Returns:
            numpy.ndarray: state values
        """
        return np.max(self.q_table, axis=1)
