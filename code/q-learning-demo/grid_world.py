"""
Grid World Environment for Q-Learning Demo
ตัวอย่าง environment สำหรับสาธิต Q-learning
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches

class GridWorld:
    def __init__(self, size=5):
        """
        สร้าง Grid World environment
        
        Args:
            size (int): ขนาดของ grid (size x size)
        """
        self.size = size
        self.n_states = size * size
        self.n_actions = 4  # up, down, left, right
        
        # กำหนดตำแหน่งเริ่มต้น, เป้าหมาย, และอุปสรรค
        self.start_pos = (0, 0)
        self.goal_pos = (size-1, size-1)
        self.obstacles = [(1, 1), (2, 1), (1, 2)] if size >= 4 else []
        
        # กำหนด reward
        self.goal_reward = 10
        self.obstacle_penalty = -5
        self.step_penalty = -0.1
        
        # Action mappings
        self.actions = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1)    # right
        }
        self.action_names = ['↑', '↓', '←', '→']
        
        self.reset()
    
    def reset(self):
        """รีเซ็ตสภาพแวดล้อมกลับไปที่จุดเริ่มต้น"""
        self.current_pos = self.start_pos
        return self.get_state()
    
    def get_state(self):
        """แปลงตำแหน่งปัจจุบันเป็น state number"""
        return self.current_pos[0] * self.size + self.current_pos[1]
    
    def pos_to_state(self, pos):
        """แปลงตำแหน่ง (row, col) เป็น state number"""
        return pos[0] * self.size + pos[1]
    
    def state_to_pos(self, state):
        """แปลง state number เป็นตำแหน่ง (row, col)"""
        return (state // self.size, state % self.size)
    
    def is_valid_pos(self, pos):
        """ตรวจสอบว่าตำแหน่งอยู่ในขอบเขตของ grid หรือไม่"""
        row, col = pos
        return 0 <= row < self.size and 0 <= col < self.size
    
    def is_obstacle(self, pos):
        """ตรวจสอบว่าตำแหน่งเป็นอุปสรรคหรือไม่"""
        return pos in self.obstacles
    
    def step(self, action):
        """
        ทำ action และคืนค่า (next_state, reward, done)
        
        Args:
            action (int): action ที่จะทำ (0-3)
            
        Returns:
            tuple: (next_state, reward, done)
        """
        # คำนวณตำแหน่งใหม่
        delta = self.actions[action]
        new_pos = (self.current_pos[0] + delta[0], self.current_pos[1] + delta[1])
        
        # ตรวจสอบว่าตำแหน่งใหม่ถูกต้องหรือไม่
        if not self.is_valid_pos(new_pos) or self.is_obstacle(new_pos):
            # ถ้าไม่ถูกต้อง ยังคงอยู่ที่เดิม
            new_pos = self.current_pos
        
        self.current_pos = new_pos
        new_state = self.get_state()
        
        # คำนวณ reward
        if new_pos == self.goal_pos:
            reward = self.goal_reward
            done = True
        elif new_pos in self.obstacles:
            reward = self.obstacle_penalty
            done = False
        else:
            reward = self.step_penalty
            done = False
        
        return new_state, reward, done
    
    def visualize(self, q_table=None, policy=None):
        """
        แสดงภาพ Grid World พร้อม Q-values หรือ policy
        
        Args:
            q_table (numpy.ndarray): Q-table ที่จะแสดง
            policy (numpy.ndarray): Policy ที่จะแสดง
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # วาดตาราง
        for i in range(self.size + 1):
            ax.axhline(i, color='black', linewidth=1)
            ax.axvline(i, color='black', linewidth=1)
        
        # วาดเซลล์ต่างๆ
        for row in range(self.size):
            for col in range(self.size):
                pos = (row, col)
                
                # เซลล์เริ่มต้น
                if pos == self.start_pos:
                    rect = Rectangle((col, self.size-1-row), 1, 1, 
                                   facecolor='green', alpha=0.3)
                    ax.add_patch(rect)
                    ax.text(col+0.5, self.size-1-row+0.5, 'S', 
                           ha='center', va='center', fontsize=16, fontweight='bold')
                
                # เซลล์เป้าหมาย
                elif pos == self.goal_pos:
                    rect = Rectangle((col, self.size-1-row), 1, 1, 
                                   facecolor='gold', alpha=0.7)
                    ax.add_patch(rect)
                    ax.text(col+0.5, self.size-1-row+0.5, 'G', 
                           ha='center', va='center', fontsize=16, fontweight='bold')
                
                # อุปสรรค
                elif pos in self.obstacles:
                    rect = Rectangle((col, self.size-1-row), 1, 1, 
                                   facecolor='red', alpha=0.7)
                    ax.add_patch(rect)
                    ax.text(col+0.5, self.size-1-row+0.5, 'X', 
                           ha='center', va='center', fontsize=16, fontweight='bold')
                
                # เซลล์ปกติ
                else:
                    state = self.pos_to_state(pos)
                    
                    if policy is not None:
                        # แสดง optimal action
                        action = np.argmax(policy[state])
                        arrow = self.action_names[action]
                        ax.text(col+0.5, self.size-1-row+0.5, arrow, 
                               ha='center', va='center', fontsize=14)
                    
                    if q_table is not None:
                        # แสดง Q-values ของแต่ละ action
                        q_vals = q_table[state]
                        
                        # แสดง Q-value สำหรับแต่ละทิศทาง
                        positions = [(0.5, 0.8), (0.5, 0.2), (0.2, 0.5), (0.8, 0.5)]
                        for a, (dx, dy) in enumerate(positions):
                            ax.text(col+dx, self.size-1-row+dy, f'{q_vals[a]:.1f}', 
                                   ha='center', va='center', fontsize=8)
        
        # ตำแหน่งปัจจุบันของ agent
        if hasattr(self, 'current_pos'):
            row, col = self.current_pos
            circle = plt.Circle((col+0.5, self.size-1-row+0.5), 0.2, 
                              color='blue', alpha=0.8)
            ax.add_patch(circle)
        
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.set_aspect('equal')
        ax.set_title('Grid World Environment\nS=Start, G=Goal, X=Obstacle', fontsize=14)
        
        # ซ่อน ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.tight_layout()
        return fig, ax
