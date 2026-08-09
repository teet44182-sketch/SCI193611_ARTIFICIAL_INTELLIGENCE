"""
Pure Python Deep Q-Network Implementation
การใช้ neural network แทน Q-table สำหรับ function approximation
ใช้ Python ธรรมดา ไม่ต้องติดตั้ง package เพิ่มเติม
"""

import random
import math
from collections import deque

class SimpleMLP:
    """Multi-Layer Perceptron แบบง่ายใช้ pure Python"""
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # สุ่ม weights แบบง่าย
        self.W1 = [[random.gauss(0, 0.1) for _ in range(hidden_size)] 
                   for _ in range(input_size)]
        self.b1 = [0.0] * hidden_size
        self.W2 = [[random.gauss(0, 0.1) for _ in range(output_size)] 
                   for _ in range(hidden_size)]
        self.b2 = [0.0] * output_size
        
        # เก็บค่าสำหรับ backpropagation
        self.hidden_output = []
        self.final_output = []
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        try:
            return 1 / (1 + math.exp(-max(-500, min(500, x))))
        except:
            return 0.5
    
    def forward(self, inputs):
        """Forward propagation"""
        # Hidden layer
        self.hidden_output = []
        for j in range(self.hidden_size):
            sum_val = self.b1[j]
            for i in range(self.input_size):
                sum_val += inputs[i] * self.W1[i][j]
            self.hidden_output.append(self.sigmoid(sum_val))
        
        # Output layer (linear)
        self.final_output = []
        for k in range(self.output_size):
            sum_val = self.b2[k]
            for j in range(self.hidden_size):
                sum_val += self.hidden_output[j] * self.W2[j][k]
            self.final_output.append(sum_val)
        
        return self.final_output[:]
    
    def train(self, inputs, targets):
        """Train with one sample"""
        # Forward pass
        outputs = self.forward(inputs)
        
        # Calculate output layer errors
        output_errors = []
        for k in range(self.output_size):
            error = targets[k] - outputs[k]
            output_errors.append(error)
        
        # Calculate hidden layer errors
        hidden_errors = []
        for j in range(self.hidden_size):
            error = 0.0
            for k in range(self.output_size):
                error += output_errors[k] * self.W2[j][k]
            # Sigmoid derivative
            sigmoid_derivative = self.hidden_output[j] * (1 - self.hidden_output[j])
            hidden_errors.append(error * sigmoid_derivative)
        
        # Update output layer weights
        for j in range(self.hidden_size):
            for k in range(self.output_size):
                self.W2[j][k] += self.learning_rate * output_errors[k] * self.hidden_output[j]
        for k in range(self.output_size):
            self.b2[k] += self.learning_rate * output_errors[k]
        
        # Update hidden layer weights
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                self.W1[i][j] += self.learning_rate * hidden_errors[j] * inputs[i]
        for j in range(self.hidden_size):
            self.b1[j] += self.learning_rate * hidden_errors[j]
        
        return outputs
    
    def predict(self, inputs):
        """Predict without updating weights"""
        return self.forward(inputs)
    
    def copy_weights_from(self, other_network):
        """Copy weights from another network"""
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                self.W1[i][j] = other_network.W1[i][j]
        
        for j in range(self.hidden_size):
            for k in range(self.output_size):
                self.W2[j][k] = other_network.W2[j][k]
            self.b1[j] = other_network.b1[j]
        
        for k in range(self.output_size):
            self.b2[k] = other_network.b2[k]

class PurePythonDQN:
    """Pure Python DQN Agent"""
    
    def __init__(self, state_size, action_size, learning_rate=0.01, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 memory_size=1000, batch_size=16, target_update=50):
        
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
        self.q_network = SimpleMLP(state_size, 32, action_size, learning_rate)
        self.target_network = SimpleMLP(state_size, 32, action_size, learning_rate)
        
        # Copy weights to target network
        self.update_target_network()
        
        # Statistics
        self.episode_rewards = []
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.copy_weights_from(self.q_network)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def get_action(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Convert state to feature vector
        features = self.state_to_features(state)
        q_values = self.q_network.predict(features)
        
        # Find action with max Q-value
        max_q = max(q_values)
        max_actions = [i for i, q in enumerate(q_values) if q == max_q]
        return random.choice(max_actions)
    
    def state_to_features(self, state):
        """Convert state to feature vector (one-hot encoding)"""
        features = [0.0] * self.state_size
        if 0 <= state < self.state_size:
            features[state] = 1.0
        return features
    
    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in batch:
            # Current Q-values
            current_features = self.state_to_features(state)
            current_q_values = self.q_network.predict(current_features)
            
            # Target Q-value
            if done:
                target_q = reward
            else:
                next_features = self.state_to_features(next_state)
                next_q_values = self.target_network.predict(next_features)
                target_q = reward + 0.95 * max(next_q_values)
            
            # Update target
            target_q_values = current_q_values[:]
            target_q_values[action] = target_q
            
            # Train
            self.q_network.train(current_features, target_q_values)
        
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
    
    def train(self, env, episodes=500, verbose=True):
        """Train for multiple episodes"""
        for episode in range(episodes):
            reward, steps = self.train_episode(env)
            self.episode_rewards.append(reward)
            
            if verbose and episode % 100 == 0:
                avg_reward = sum(self.episode_rewards[-50:]) / min(50, len(self.episode_rewards))
                print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, "
                      f"Epsilon = {self.epsilon:.3f}, Memory = {len(self.memory)}")
    
    def test(self, env, show_path=False):
        """Test the trained agent"""
        state = env.reset()
        total_reward = 0
        steps = 0
        path = []
        action_names = ['↑', '↓', '←', '→']
        
        if show_path:
            print("Pure Python DQN เส้นทางการเดิน:")
            env.print_grid()
        
        for step in range(50):
            action = self.get_action(state, training=False)
            next_state, reward, done = env.step(action)
            
            total_reward += reward
            steps += 1
            path.append(env.state_to_pos(state))
            
            if show_path:
                print(f"Step {step+1}: {env.state_to_pos(state)} -> {action_names[action]}")
                env.print_grid()
            
            state = next_state
            
            if done:
                if show_path:
                    print("ถึงเป้าหมายแล้ว!")
                break
        
        return total_reward, steps, path

# Import SimpleGridWorld จากไฟล์ที่มีอยู่แล้ว
try:
    from simple_q_learning import SimpleGridWorld, SimpleQLearning
    HAS_SIMPLE_QL = True
except:
    HAS_SIMPLE_QL = False

def compare_tabular_vs_pure_dqn():
    """เปรียบเทียบ Tabular Q-Learning กับ Pure Python DQN"""
    if not HAS_SIMPLE_QL:
        print("ต้องการไฟล์ simple_q_learning.py")
        return
    
    print("🧠 เปรียบเทียบ Tabular Q-Learning vs Pure Python DQN")
    print("=" * 60)
    
    # สร้าง environment
    env = SimpleGridWorld(size=4, difficulty='normal')
    env.print_environment_info()
    env.print_grid()
    
    episodes = 400
    
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
    
    print("\n2. ทดสอบ Pure Python DQN:")
    print("-" * 30)
    
    # Pure Python DQN
    dqn_agent = PurePythonDQN(
        state_size=16,  # One-hot encoding
        action_size=4,
        learning_rate=0.02,
        epsilon=1.0,
        epsilon_decay=0.99
    )
    
    print("กำลังฝึก Pure Python DQN...")
    dqn_agent.train(env, episodes=episodes)
    
    env.reset()
    dqn_reward, dqn_steps, _ = dqn_agent.test(env, show_path=False)
    print(f"Pure DQN - Reward: {dqn_reward:.2f}, Steps: {dqn_steps}")
    
    print("\n3. การเปรียบเทียบ:")
    print("-" * 30)
    print(f"Tabular Q-Learning: {tab_reward:.2f} reward, {tab_steps} steps")
    print(f"Pure Python DQN:    {dqn_reward:.2f} reward, {dqn_steps} steps")
    
    difference = abs(tab_reward - dqn_reward)
    if difference < 1.0:
        print("→ ทั้งสองวิธีให้ผลใกล้เคียงกัน")
    elif tab_reward > dqn_reward:
        print("→ Tabular Q-Learning ทำได้ดีกว่าในปัญหานี้")
    else:
        print("→ Pure Python DQN ทำได้ดีกว่าในปัญหานี้")
    
    print("\n📝 ข้อสังเกตสำคัญ:")
    print("- Tabular: เก็บ Q-value แยกแต่ละ state")
    print("- DQN: ใช้ neural network เรียนรู้ pattern")
    print("- ในปัญหาเล็ก: Tabular มักจะเรียนรู้เร็วกว่า")
    print("- ในปัญหาใหญ่: DQN จะมีประโยชน์มากกว่า")
    
    return tabular_agent, dqn_agent

def demonstrate_network_learning():
    """แสดงการเรียนรู้ของ neural network"""
    print("\n🔍 การวิเคราะห์การเรียนรู้ของ Neural Network")
    print("=" * 50)
    
    if not HAS_SIMPLE_QL:
        return
    
    env = SimpleGridWorld(size=3, difficulty='easy')
    print("ใช้ Grid World 3x3 แบบง่ายเพื่อวิเคราะห์:")
    env.print_grid()
    
    # สร้าง DQN agent
    agent = PurePythonDQN(
        state_size=9,
        action_size=4,
        learning_rate=0.05,
        epsilon=0.5
    )
    
    print("\nการเรียนรู้ทีละ episode:")
    
    for episode in [0, 50, 100, 200]:
        if episode > 0:
            # ฝึกเพิ่ม
            agent.train(env, episodes=50, verbose=False)
        
        print(f"\nEpisode {episode}:")
        
        # ทดสอบ Q-values ของแต่ละ state
        print("Q-values สำหรับแต่ละ state:")
        for state in range(9):
            features = agent.state_to_features(state)
            q_values = agent.q_network.predict(features)
            best_action = q_values.index(max(q_values))
            action_names = ['↑', '↓', '←', '→']
            pos = env.state_to_pos(state)
            print(f"  State {state} {pos}: Best={action_names[best_action]} "
                  f"(Q={max(q_values):.2f})")
        
        # ทดสอบ performance
        env.reset()
        reward, steps, _ = agent.test(env, show_path=False)
        print(f"  Performance: {reward:.2f} reward, {steps} steps")

def interactive_dqn_demo():
    """Demo DQN แบบ interactive"""
    if not HAS_SIMPLE_QL:
        print("ต้องการไฟล์ simple_q_learning.py")
        return
    
    print("\n🎮 Interactive Pure Python DQN Demo")
    print("=" * 40)
    
    # รับ input จากผู้ใช้
    try:
        size = int(input("ขนาด grid (3-5, default=4): ") or "4")
        difficulty = input("ระดับความยาก (easy/normal/hard/maze, default=normal): ") or "normal"
        episodes = int(input("จำนวน episodes (100-1000, default=400): ") or "400")
        learning_rate = float(input("Learning rate (0.01-0.1, default=0.02): ") or "0.02")
    except ValueError:
        size, difficulty, episodes, learning_rate = 4, 'normal', 400, 0.02
        print("ใช้ค่า default")
    
    print(f"\nสร้าง Pure Python DQN:")
    print(f"Grid: {size}x{size}, Difficulty: {difficulty}")
    print(f"Episodes: {episodes}, Learning Rate: {learning_rate}")
    
    env = SimpleGridWorld(size=size, difficulty=difficulty)
    env.print_environment_info()
    env.print_grid()
    
    # สร้าง DQN agent
    agent = PurePythonDQN(
        state_size=size * size,
        action_size=4,
        learning_rate=learning_rate,
        epsilon=1.0,
        epsilon_decay=0.995
    )
    
    print("\nกำลังฝึก...")
    agent.train(env, episodes=episodes, verbose=True)
    
    print("\nผลลัพธ์การทดสอบ:")
    env.reset()
    reward, steps, _ = agent.test(env, show_path=False)
    optimal = env.get_optimal_path_length()
    efficiency = (optimal / steps * 100) if steps > 0 else 0
    
    print(f"Final Reward: {reward:.2f}")
    print(f"Steps: {steps}")
    print(f"Optimal Path: {optimal} steps")
    print(f"Efficiency: {efficiency:.1f}%")
    
    # ถามว่าต้องการดูเส้นทางหรือไม่
    show_path = input("\nต้องการดูเส้นทางการเดิน? (y/n): ").lower() == 'y'
    if show_path:
        print("\nเส้นทางที่เรียนรู้ได้:")
        agent.test(env, show_path=True)

def main():
    """ฟังก์ชันหลัก"""
    print("🧠 Pure Python Deep Q-Network Demo")
    print("Neural Network Function Approximation (ไม่ต้องติดตั้ง package)")
    print("=" * 65)
    
    if not HAS_SIMPLE_QL:
        print("❌ ต้องการไฟล์ simple_q_learning.py เพื่อรันการเปรียบเทียบ")
        print("   คุณสามารถสร้าง SimpleGridWorld ได้ในไฟล์นี้")
        return
    
    print("✅ Pure Python DQN พร้อมใช้งาน (ไม่ต้อง numpy หรือ pytorch)")
    
    while True:
        print("\nเลือกโหมด:")
        print("1. เปรียบเทียบ Tabular vs Pure Python DQN")
        print("2. วิเคราะห์การเรียนรู้ของ Neural Network")
        print("3. Interactive DQN Demo")
        print("4. ออกจากโปรแกรม")
        
        choice = input("\nเลือก (1-4): ").strip()
        
        if choice == '1':
            compare_tabular_vs_pure_dqn()
        elif choice == '2':
            demonstrate_network_learning()
        elif choice == '3':
            interactive_dqn_demo()
        elif choice == '4':
            print("ขอบคุณที่ใช้งาน!")
            break
        else:
            print("กรุณาเลือก 1-4")
        
        print("\n" + "="*65)

if __name__ == "__main__":
    main()
