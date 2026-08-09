"""
Deep Q-Network (DQN) Implementation
การใช้ neural network แทน Q-table สำหรับ function approximation

ต้องติดตั้ง: pip install torch numpy matplotlib
"""

import random
import numpy as np
from collections import deque
import json

# ใช้ pure Python สำหรับ neural network แบบง่าย
class SimpleNeuralNetwork:
    """Neural Network แบบง่ายใช้ pure Python"""
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # สุ่ม weights
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        """Forward propagation"""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.z2  # Linear output for Q-values
        return self.a2
    
    def backward(self, X, y, output):
        """Backward propagation"""
        m = X.shape[0]
        
        # Output layer
        dZ2 = output - y
        dW2 = (1/m) * np.dot(self.a1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # Hidden layer
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.a1)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # Update weights
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def train(self, X, y):
        """Train the network"""
        output = self.forward(X)
        self.backward(X, y, output)
        return output
    
    def predict(self, X):
        """Predict using the network"""
        return self.forward(X)

class DQNAgent:
    """Deep Q-Network Agent"""
    
    def __init__(self, state_size, action_size, learning_rate=0.001, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 memory_size=2000, batch_size=32, target_update=100):
        
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.target_update = target_update
        self.update_count = 0
        
        # Neural networks
        self.q_network = SimpleNeuralNetwork(state_size, 64, action_size, learning_rate)
        self.target_network = SimpleNeuralNetwork(state_size, 64, action_size, learning_rate)
        
        # Copy weights to target network
        self.update_target_network()
        
        # Statistics
        self.episode_rewards = []
        self.losses = []
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.W1 = self.q_network.W1.copy()
        self.target_network.b1 = self.q_network.b1.copy()
        self.target_network.W2 = self.q_network.W2.copy()
        self.target_network.b2 = self.q_network.b2.copy()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def get_action(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Convert state to feature vector
        state_vector = self.state_to_features(state).reshape(1, -1)
        q_values = self.q_network.predict(state_vector)
        return np.argmax(q_values[0])
    
    def state_to_features(self, state):
        """Convert state to feature vector"""
        # Simple feature representation
        features = np.zeros(self.state_size)
        features[state] = 1.0  # One-hot encoding
        return features
    
    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        
        states = np.array([self.state_to_features(e[0]) for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([self.state_to_features(e[3]) for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Current Q-values
        current_q_values = self.q_network.predict(states)
        
        # Target Q-values
        next_q_values = self.target_network.predict(next_states)
        max_next_q_values = np.max(next_q_values, axis=1)
        
        # Calculate targets
        targets = current_q_values.copy()
        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + 0.95 * max_next_q_values[i]
        
        # Train network
        self.q_network.train(states, targets)
        
        # Update target network periodically
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train_episode(self, env, max_steps=100):
        """Train one episode"""
        state = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            action = self.get_action(state)
            next_state, reward, done = env.step(action)
            
            self.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Learn from experience
        self.replay()
        
        return total_reward, steps
    
    def train(self, env, episodes=1000, verbose=True):
        """Train for multiple episodes"""
        for episode in range(episodes):
            reward, steps = self.train_episode(env)
            self.episode_rewards.append(reward)
            
            if verbose and episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, "
                      f"Epsilon = {self.epsilon:.3f}, Memory = {len(self.memory)}")
    
    def test(self, env, show_path=False):
        """Test the trained agent"""
        state = env.reset()
        total_reward = 0
        steps = 0
        path = []
        
        if show_path:
            print("DQN Agent เส้นทางการเดิน:")
            env.print_grid()
        
        for step in range(50):
            action = self.get_action(state, training=False)
            next_state, reward, done = env.step(action)
            
            total_reward += reward
            steps += 1
            path.append(env.state_to_pos(state))
            
            if show_path:
                action_names = ['↑', '↓', '←', '→']
                print(f"Step {step+1}: {env.state_to_pos(state)} -> {action_names[action]}")
                env.print_grid()
            
            state = next_state
            
            if done:
                if show_path:
                    print("ถึงเป้าหมายแล้ว!")
                break
        
        return total_reward, steps, path

# ใช้ SimpleGridWorld จากไฟล์ที่มีอยู่แล้ว
try:
    from simple_q_learning import SimpleGridWorld, SimpleQLearning
    HAS_SIMPLE_QL = True
except:
    HAS_SIMPLE_QL = False
    print("ไม่สามารถ import SimpleGridWorld ได้")

def compare_tabular_vs_dqn():
    """เปรียบเทียบ Tabular Q-Learning กับ DQN"""
    if not HAS_SIMPLE_QL:
        print("ต้องการไฟล์ simple_q_learning.py")
        return
    
    print("🧠 การเปรียบเทียบ Tabular Q-Learning vs Deep Q-Network")
    print("=" * 60)
    
    # สร้าง environment
    env = SimpleGridWorld(size=4, difficulty='normal')
    env.print_environment_info()
    env.print_grid()
    
    episodes = 500
    
    print("\n1. ทดสอบ Tabular Q-Learning:")
    print("-" * 30)
    
    # Tabular Q-Learning
    tabular_agent = SimpleQLearning(
        n_states=16,
        n_actions=4,
        learning_rate=0.1,
        discount=0.9,
        epsilon=0.3
    )
    
    print("กำลังฝึก Tabular Q-Learning...")
    tabular_agent.train(env, episodes=episodes)
    
    env.reset()
    tab_reward, tab_steps, _ = tabular_agent.test(env, show_path=False)
    print(f"Tabular - Reward: {tab_reward:.2f}, Steps: {tab_steps}")
    
    print("\n2. ทดสอบ Deep Q-Network:")
    print("-" * 30)
    
    # DQN
    dqn_agent = DQNAgent(
        state_size=16,  # One-hot encoding
        action_size=4,
        learning_rate=0.01,
        epsilon=1.0,
        epsilon_decay=0.995
    )
    
    print("กำลังฝึก DQN...")
    dqn_agent.train(env, episodes=episodes)
    
    env.reset()
    dqn_reward, dqn_steps, _ = dqn_agent.test(env, show_path=False)
    print(f"DQN - Reward: {dqn_reward:.2f}, Steps: {dqn_steps}")
    
    print("\n3. การเปรียบเทียบ:")
    print("-" * 30)
    print(f"Tabular Q-Learning: {tab_reward:.2f} reward, {tab_steps} steps")
    print(f"Deep Q-Network:     {dqn_reward:.2f} reward, {dqn_steps} steps")
    
    if tab_reward > dqn_reward:
        print("→ Tabular Q-Learning ทำได้ดีกว่าในปัญหานี้")
    else:
        print("→ DQN ทำได้ดีกว่าในปัญหานี้")
    
    print("\n📝 ข้อสังเกต:")
    print("- Tabular Q-Learning เหมาะกับ state space เล็ก")
    print("- DQN เหมาะกับ state space ใหญ่")
    print("- ในปัญหา Grid World เล็ก Tabular มักจะดีกว่า")
    print("- DQN มีประโยชน์เมื่อ state space ใหญ่มาก")

def demonstrate_feature_engineering():
    """แสดงการทำ feature engineering สำหรับ DQN"""
    print("\n🔧 Feature Engineering สำหรับ DQN")
    print("=" * 40)
    
    if not HAS_SIMPLE_QL:
        return
    
    class EnhancedDQNAgent(DQNAgent):
        """DQN Agent พร้อม enhanced features"""
        
        def state_to_features(self, state):
            """Convert state to enhanced feature vector"""
            env = self.env_ref  # จะต้องเซ็ตใน train
            pos = env.state_to_pos(state)
            
            features = []
            
            # 1. Position features (normalized)
            features.extend([pos[0] / env.size, pos[1] / env.size])
            
            # 2. Distance to goal
            goal_dist = abs(pos[0] - env.goal[0]) + abs(pos[1] - env.goal[1])
            features.append(goal_dist / (2 * env.size))
            
            # 3. Surrounding obstacles (4 directions)
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                new_pos = (pos[0] + dr, pos[1] + dc)
                if new_pos in env.obstacles or not env.is_valid(new_pos):
                    features.append(1.0)
                else:
                    features.append(0.0)
            
            # 4. One-hot position (optional)
            one_hot = [0.0] * (env.size * env.size)
            one_hot[state] = 1.0
            features.extend(one_hot)
            
            return np.array(features)
    
    env = SimpleGridWorld(size=4, difficulty='hard')
    
    # แสดงตัวอย่าง features
    enhanced_agent = EnhancedDQNAgent(
        state_size=2 + 1 + 4 + 16,  # pos + dist + obstacles + one-hot
        action_size=4
    )
    enhanced_agent.env_ref = env
    
    print("ตัวอย่าง Enhanced Features:")
    state = 5  # ตำแหน่ง (1,1)
    features = enhanced_agent.state_to_features(state)
    print(f"State {state} -> Features length: {len(features)}")
    print(f"Position features: {features[:2]}")
    print(f"Distance to goal: {features[2]:.2f}")
    print(f"Obstacle features: {features[3:7]}")
    print("One-hot features: [แสดงบางส่วน]", features[7:11], "...")

def advanced_dqn_demo():
    """Demo ขั้นสูงของ DQN"""
    print("\n🚀 Advanced DQN Features")
    print("=" * 30)
    
    if not HAS_SIMPLE_QL:
        return
    
    class AdvancedDQN(DQNAgent):
        """DQN พร้อมฟีเจอร์ขั้นสูง"""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.double_dqn = True
            self.prioritized_replay = False  # เพิ่มได้ในอนาคต
        
        def replay(self):
            """Enhanced replay with Double DQN"""
            if len(self.memory) < self.batch_size:
                return
            
            batch = random.sample(self.memory, self.batch_size)
            
            states = np.array([self.state_to_features(e[0]) for e in batch])
            actions = np.array([e[1] for e in batch])
            rewards = np.array([e[2] for e in batch])
            next_states = np.array([self.state_to_features(e[3]) for e in batch])
            dones = np.array([e[4] for e in batch])
            
            current_q_values = self.q_network.predict(states)
            
            if self.double_dqn:
                # Double DQN: ใช้ main network เลือก action, target network ประเมิน value
                next_q_main = self.q_network.predict(next_states)
                next_actions = np.argmax(next_q_main, axis=1)
                next_q_target = self.target_network.predict(next_states)
                max_next_q_values = next_q_target[np.arange(self.batch_size), next_actions]
            else:
                # Standard DQN
                next_q_values = self.target_network.predict(next_states)
                max_next_q_values = np.max(next_q_values, axis=1)
            
            targets = current_q_values.copy()
            for i in range(self.batch_size):
                if dones[i]:
                    targets[i][actions[i]] = rewards[i]
                else:
                    targets[i][actions[i]] = rewards[i] + 0.95 * max_next_q_values[i]
            
            self.q_network.train(states, targets)
            
            self.update_count += 1
            if self.update_count % self.target_update == 0:
                self.update_target_network()
            
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
    
    print("Advanced DQN พร้อม Double DQN algorithm")
    print("- ลดปัญหา overestimation bias")
    print("- เสถียรกว่า standard DQN")

def main():
    """ฟังก์ชันหลัก"""
    print("🧠 Deep Q-Network (DQN) Demo")
    print("Neural Network Function Approximation สำหรับ Q-Learning")
    print("=" * 60)
    
    if not HAS_SIMPLE_QL:
        print("❌ ต้องการไฟล์ simple_q_learning.py เพื่อรันการเปรียบเทียบ")
        return
    
    try:
        import numpy as np
        print("✅ NumPy พร้อมใช้งาน")
    except ImportError:
        print("❌ ต้องติดตั้ง NumPy: pip install numpy")
        return
    
    while True:
        print("\nเลือกโหมด:")
        print("1. เปรียบเทียบ Tabular vs DQN")
        print("2. Feature Engineering Demo")
        print("3. Advanced DQN Features")
        print("4. ออกจากโปรแกรม")
        
        choice = input("\nเลือก (1-4): ").strip()
        
        if choice == '1':
            compare_tabular_vs_dqn()
        elif choice == '2':
            demonstrate_feature_engineering()
        elif choice == '3':
            advanced_dqn_demo()
        elif choice == '4':
            print("ขอบคุณที่ใช้งาน!")
            break
        else:
            print("กรุณาเลือก 1-4")
        
        print("\n" + "="*60)

if __name__ == "__main__":
    main()
