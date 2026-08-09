"""
Simple Q-Learning Example (Pure Python)
ตัวอย่าง Q-learning แบบง่าย ใช้ Python ธรรมดาเท่านั้น
ไม่ต้องพึ่ง numpy หรือ matplotlib

การใช้งาน: python simple_q_learning.py
"""

import random
import json

class SimpleGridWorld:
    """Grid World Environment แบบง่าย"""
    
    def __init__(self, size=4, difficulty='normal'):
        self.size = size
        self.start = (0, 0)
        self.goal = (size-1, size-1)
        self.difficulty = difficulty
        
        # กำหนดอุปสรรคตามระดับความยาก
        self.obstacles = self._generate_obstacles(size, difficulty)
        
        # Actions: 0=up, 1=down, 2=left, 3=right
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.action_names = ['↑', '↓', '←', '→']
        
        self.reset()
    
    def _generate_obstacles(self, size, difficulty):
        """สร้างอุปสรรคตามระดับความยาก"""
        if size < 3:
            return []
        
        if difficulty == 'easy':
            if size == 3:
                return [(1, 1)]
            elif size == 4:
                return [(1, 1), (2, 2)]
            elif size == 5:
                return [(1, 1), (2, 2), (3, 1)]
            else:  # size >= 6
                return [(1, 1), (2, 2), (3, 1), (1, 3)]
        
        elif difficulty == 'normal':
            if size == 3:
                return [(1, 1)]
            elif size == 4:
                return [(1, 1), (2, 1)]
            elif size == 5:
                return [(1, 1), (2, 1), (1, 3), (3, 2)]
            else:  # size >= 6
                return [(1, 1), (2, 1), (1, 3), (3, 2), (4, 3), (2, 4)]
        
        elif difficulty == 'hard':
            if size == 3:
                return [(1, 1), (1, 0)]
            elif size == 4:
                return [(1, 1), (2, 1), (1, 2), (3, 1)]
            elif size == 5:
                return [(1, 1), (2, 1), (1, 2), (3, 1), (1, 3), (3, 3)]
            else:  # size >= 6
                return [(1, 1), (2, 1), (1, 2), (3, 1), (1, 3), (3, 3), 
                       (4, 2), (2, 4), (4, 4), (0, 3)]
        
        elif difficulty == 'maze':
            # สร้างเขาวงกต
            if size == 4:
                return [(1, 0), (1, 2), (2, 0), (2, 2)]
            elif size == 5:
                return [(1, 0), (1, 2), (1, 4), (2, 2), (3, 0), (3, 2), (3, 4)]
            elif size == 6:
                return [(1, 0), (1, 2), (1, 4), (2, 2), (2, 4), 
                       (3, 0), (3, 2), (4, 0), (4, 2), (4, 4)]
            else:  # size >= 7
                obstacles = []
                # สร้างกำแพงแนวตั้ง
                for i in range(1, size-1, 2):
                    for j in range(0, size-1, 2):
                        if (i, j) != self.start and (i, j) != self.goal:
                            obstacles.append((i, j))
                # สร้างกำแพงแนวนอน
                for i in range(0, size-1, 2):
                    for j in range(1, size-1, 2):
                        if (i, j) != self.start and (i, j) != self.goal:
                            obstacles.append((i, j))
                return obstacles
        
        else:  # custom หรือ default
            return [(1, 1), (2, 1)] if size >= 4 else []
    
    def get_difficulty_info(self):
        """แสดงข้อมูลเกี่ยวกับระดับความยาก"""
        info = {
            'easy': 'ง่าย - อุปสรรคน้อย เหมาะสำหรับเริ่มต้น',
            'normal': 'ปกติ - อุปสรรคปานกลาง เหมาะสำหรับการเรียนรู้',
            'hard': 'ยาก - อุปสรรคมาก ต้องใช้กลยุทธ์',
            'maze': 'เขาวงกต - รูปแบบเขาวงกต ท้าทายที่สุด'
        }
        return info.get(self.difficulty, 'กำหนดเอง')
    
    def count_obstacles(self):
        """นับจำนวนอุปสรรค"""
        return len(self.obstacles)
    
    def get_optimal_path_length(self):
        """คำนวณความยาวเส้นทางที่สั้นที่สุด (Manhattan distance)"""
        return abs(self.goal[0] - self.start[0]) + abs(self.goal[1] - self.start[1])
    
    def print_environment_info(self):
        """แสดงข้อมูลสภาพแวดล้อม"""
        print(f"Grid World {self.size}x{self.size}")
        print(f"ระดับความยาก: {self.difficulty} - {self.get_difficulty_info()}")
        print(f"จำนวนอุปสรรค: {self.count_obstacles()}")
        print(f"ระยะทางที่สั้นที่สุด: {self.get_optimal_path_length()} steps")
        print(f"จุดเริ่มต้น: {self.start}, เป้าหมาย: {self.goal}")
        if self.obstacles:
            print(f"ตำแหน่งอุปสรรค: {self.obstacles}")
        print()
    
    def reset(self):
        """รีเซ็ตกลับไปจุดเริ่มต้น"""
        self.pos = self.start
        return self.get_state()
    
    def get_state(self):
        """แปลง position เป็น state number"""
        return self.pos[0] * self.size + self.pos[1]
    
    def state_to_pos(self, state):
        """แปลง state number เป็น position"""
        return (state // self.size, state % self.size)
    
    def is_valid(self, pos):
        """ตรวจสอบว่า position ถูกต้องหรือไม่"""
        r, c = pos
        return 0 <= r < self.size and 0 <= c < self.size and pos not in self.obstacles
    
    def step(self, action):
        """ทำ action และคืนค่า (next_state, reward, done)"""
        dr, dc = self.actions[action]
        new_pos = (self.pos[0] + dr, self.pos[1] + dc)
        
        # ถ้าไม่สามารถเดินได้ ยังคงอยู่ที่เดิม
        if not self.is_valid(new_pos):
            new_pos = self.pos
        
        self.pos = new_pos
        
        # คำนวณ reward
        if self.pos == self.goal:
            reward = 10
            done = True
        elif self.pos in self.obstacles:
            reward = -5
            done = False
        else:
            reward = -0.1
            done = False
        
        return self.get_state(), reward, done
    
    def print_grid(self, q_table=None, policy=None):
        """แสดง grid พร้อม agent position"""
        print("Grid World:")
        for r in range(self.size):
            row = []
            for c in range(self.size):
                pos = (r, c)
                if pos == self.pos:
                    row.append('A')  # Agent
                elif pos == self.start:
                    row.append('S')  # Start
                elif pos == self.goal:
                    row.append('G')  # Goal
                elif pos in self.obstacles:
                    row.append('X')  # Obstacle
                else:
                    if policy:
                        state = r * self.size + c
                        best_action = max(range(4), key=lambda a: policy[state][a])
                        row.append(self.action_names[best_action])
                    else:
                        row.append('.')
            print(' '.join(row))
        print()

class SimpleQLearning:
    """Q-Learning Agent แบบง่าย"""
    
    def __init__(self, n_states, n_actions, learning_rate=0.1, 
                 discount=0.9, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        
        # สร้าง Q-table เป็น list of lists
        self.q_table = [[0.0 for _ in range(n_actions)] for _ in range(n_states)]
        
        # สถิติ
        self.episode_rewards = []
    
    def get_action(self, state, training=True):
        """เลือก action ด้วย epsilon-greedy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return max(range(self.n_actions), key=lambda a: self.q_table[state][a])
    
    def update(self, state, action, reward, next_state):
        """อัปเดต Q-value"""
        # หา max Q-value ของ next state
        max_next_q = max(self.q_table[next_state])
        
        # Q-learning update
        target = reward + self.gamma * max_next_q
        error = target - self.q_table[state][action]
        self.q_table[state][action] += self.lr * error
    
    def train_episode(self, env, max_steps=100):
        """ฝึก 1 episode"""
        state = env.reset()
        total_reward = 0
        steps = 0
        
        for _ in range(max_steps):
            action = self.get_action(state)
            next_state, reward, done = env.step(action)
            
            self.update(state, action, reward, next_state)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        return total_reward, steps
    
    def train(self, env, episodes=1000):
        """ฝึกหลาย episodes"""
        print(f"เริ่มฝึก {episodes} episodes...")
        
        for episode in range(episodes):
            reward, steps = self.train_episode(env)
            self.episode_rewards.append(reward)
            
            # ลด epsilon
            if episode > 0 and episode % 100 == 0:
                self.epsilon = max(0.01, self.epsilon * 0.95)
            
            # แสดงผล
            if episode % 200 == 0:
                avg_reward = sum(self.episode_rewards[-50:]) / min(50, len(self.episode_rewards))
                print(f"Episode {episode}: Average reward = {avg_reward:.2f}, Epsilon = {self.epsilon:.3f}")
    
    def test(self, env, show_path=True):
        """ทดสอบ agent"""
        state = env.reset()
        total_reward = 0
        steps = 0
        path = []
        action_names = ['↑', '↓', '←', '→']
        
        if show_path:
            print("เส้นทางการเดิน:")
            env.print_grid()
        
        for step in range(50):
            action = self.get_action(state, training=False)
            next_state, reward, done = env.step(action)
            
            total_reward += reward
            steps += 1
            path.append((env.state_to_pos(state), action_names[action]))
            
            if show_path:
                print(f"Step {step+1}: {env.state_to_pos(state)} -> {action_names[action]}")
                env.print_grid()
            
            state = next_state
            
            if done:
                print("ถึงเป้าหมายแล้ว!")
                break
        
        return total_reward, steps, path
    
    def print_q_table(self):
        """แสดง Q-table"""
        print("Q-Table:")
        print("State |   ↑   |   ↓   |   ←   |   →   ")
        print("-" * 40)
        for state in range(self.n_states):
            q_vals = self.q_table[state]
            print(f"{state:5d} | {q_vals[0]:5.2f} | {q_vals[1]:5.2f} | {q_vals[2]:5.2f} | {q_vals[3]:5.2f}")
        print()
    
    def get_policy(self):
        """สร้าง policy จาก Q-table"""
        policy = []
        for state in range(self.n_states):
            policy_state = [0.0] * self.n_actions
            best_action = max(range(self.n_actions), key=lambda a: self.q_table[state][a])
            policy_state[best_action] = 1.0
            policy.append(policy_state)
        return policy

def demo_simple():
    """การสาธิตแบบง่าย"""
    print("=" * 60)
    print("Q-Learning Demo - Enhanced Version")
    print("ตัวอย่าง Q-learning ใน Grid World พร้อมระดับความยาก")
    print("=" * 60)
    print()
    
    # เลือกระดับความยาก
    print("เลือกระดับความยาก:")
    print("1. Easy (ง่าย)")
    print("2. Normal (ปกติ)")
    print("3. Hard (ยาก)")
    print("4. Maze (เขาวงกต)")
    
    difficulty_map = {'1': 'easy', '2': 'normal', '3': 'hard', '4': 'maze'}
    choice = input("เลือก (1-4, default=2): ").strip() or '2'
    difficulty = difficulty_map.get(choice, 'normal')
    
    # สร้าง environment
    env = SimpleGridWorld(size=5, difficulty=difficulty)
    env.print_environment_info()
    env.print_grid()
    # สร้าง agent
    agent = SimpleQLearning(
        n_states=env.size * env.size,
        n_actions=4,
        learning_rate=0.1,
        discount=0.9,
        epsilon=0.3  # เพิ่ม exploration สำหรับ environment ที่ซับซ้อนขึ้น
    )
    
    print("ก่อนการฝึก - ทดสอบ random policy:")
    reward, steps, _ = agent.test(env, show_path=False)
    print(f"Reward: {reward:.2f}, Steps: {steps}")
    print()
    
    # ฝึก agent (เพิ่มจำนวน episodes สำหรับ environment ที่ซับซ้อน)
    episodes = 1500 if difficulty in ['hard', 'maze'] else 1000
    print(f"เริ่มฝึก {episodes} episodes...")
    agent.train(env, episodes=episodes)
    print()
    
    print("หลังการฝึก - ทดสอบ learned policy:")
    reward, steps, path = agent.test(env, show_path=False)  # ไม่แสดง path ใน overview
    print(f"Final reward: {reward:.2f}, Steps: {steps}")
    optimal_steps = env.get_optimal_path_length()
    efficiency = (optimal_steps / steps * 100) if steps > 0 else 0
    print(f"ประสิทธิภาพ: {efficiency:.1f}% (เทียบกับเส้นทางที่สั้นที่สุด {optimal_steps} steps)")
    print()
    
    # ถามว่าต้องการดู details หรือไม่
    show_details = input("ต้องการดู Q-table และเส้นทางการเดิน? (y/n): ").lower() == 'y'
    if show_details:
        print("\nQ-Table (10 states แรก):")
        print("State |   ↑   |   ↓   |   ←   |   →   ")
        print("-" * 40)
        for state in range(min(10, env.size * env.size)):
            q_vals = agent.q_table[state]
            print(f"{state:5d} | {q_vals[0]:5.2f} | {q_vals[1]:5.2f} | {q_vals[2]:5.2f} | {q_vals[3]:5.2f}")
        
        print("\nเส้นทางการเดิน:")
        agent.test(env, show_path=True)
    
    # แสดง learned policy
    print("Learned Policy (แสดงทิศทางที่ดีที่สุดในแต่ละ state):")
    policy = agent.get_policy()
    env.reset()
    env.print_grid(policy=policy)
    
    # สถิติการเรียนรู้
    print("สถิติการเรียนรู้:")
    print(f"Episode แรก 10 ตอน - Average reward: {sum(agent.episode_rewards[:10])/10:.2f}")
    print(f"Episode สุดท้าย 10 ตอน - Average reward: {sum(agent.episode_rewards[-10:])/10:.2f}")
    
    # วิเคราะห์ความยาก
    print(f"\nการวิเคราะห์ความยาก:")
    print(f"- จำนวนอุปสรรค: {env.count_obstacles()}")
    print(f"- อัตราส่วนอุปสรรคต่อพื้นที่: {env.count_obstacles()/(env.size*env.size)*100:.1f}%")
    success_rate = sum(1 for r in agent.episode_rewards[-100:] if r > 5) / min(100, len(agent.episode_rewards)) * 100
    print(f"- อัตราความสำเร็จในช่วงท้าย: {success_rate:.1f}%")

def interactive_demo():
    """การสาธิตแบบ interactive"""
    print("=" * 60)
    print("Q-Learning Interactive Demo - Enhanced")
    print("=" * 60)
    print()
    
    # รับ input
    try:
        size = int(input("ขนาด grid (3-8, default=5): ") or "5")
        
        print("\nเลือกระดับความยาก:")
        print("1. Easy (ง่าย)")
        print("2. Normal (ปกติ)")  
        print("3. Hard (ยาก)")
        print("4. Maze (เขาวงกต)")
        difficulty_map = {'1': 'easy', '2': 'normal', '3': 'hard', '4': 'maze'}
        diff_choice = input("ระดับความยาก (1-4, default=2): ").strip() or '2'
        difficulty = difficulty_map.get(diff_choice, 'normal')
        
        lr = float(input("Learning rate (0.01-1.0, default=0.1): ") or "0.1")
        epsilon = float(input("Epsilon (0.01-1.0, default=0.3): ") or "0.3")
        episodes = int(input("จำนวน episodes (500-3000, default=1500): ") or "1500")
    except ValueError:
        size, difficulty, lr, epsilon, episodes = 5, 'normal', 0.1, 0.3, 1500
        print("ใช้ค่า default")
    
    print(f"\nสร้าง Grid World {size}x{size} - {difficulty}")
    print(f"พารามิเตอร์: lr={lr}, epsilon={epsilon}, episodes={episodes}")
    print()
    
    # รัน experiment
    env = SimpleGridWorld(size=size, difficulty=difficulty)
    env.print_environment_info()
    
    agent = SimpleQLearning(
        n_states=size * size,
        n_actions=4,
        learning_rate=lr,
        discount=0.9,
        epsilon=epsilon
    )
    
    env.print_grid()
    agent.train(env, episodes=episodes)
    
    print("\nผลลัพธ์:")
    reward, steps, _ = agent.test(env, show_path=False)
    optimal_steps = env.get_optimal_path_length()
    efficiency = (optimal_steps / steps * 100) if steps > 0 else 0
    print(f"Final performance: Reward={reward:.2f}, Steps={steps}")
    print(f"ประสิทธิภาพ: {efficiency:.1f}% (เทียบกับเส้นทางที่สั้นที่สุด {optimal_steps} steps)")
    
    # วิเคราะห์ความยาก
    print(f"\nการวิเคราะห์:")
    print(f"- จำนวนอุปสรรค: {env.count_obstacles()}")
    success_rate = sum(1 for r in agent.episode_rewards[-100:] if r > 5) / min(100, len(agent.episode_rewards)) * 100
    print(f"- อัตราความสำเร็จ: {success_rate:.1f}%")
    
    # ถามว่าต้องการดู details หรือไม่
    show_details = input("\\nต้องการดู Q-table และเส้นทาง? (y/n): ").lower() == 'y'
    if show_details:
        agent.print_q_table()
        print("\\nทดสอบเส้นทาง:")
        agent.test(env, show_path=True)

def compare_difficulties():
    """เปรียบเทียบระดับความยากต่างๆ"""
    print("=" * 60)
    print("Difficulty Comparison Demo")
    print("เปรียบเทียบการเรียนรู้ในระดับความยากต่างๆ")
    print("=" * 60)
    print()
    
    difficulties = ['easy', 'normal', 'hard', 'maze']
    size = 5
    episodes = 1000
    results = {}
    
    print(f"ทดสอบในกริด {size}x{size} ด้วย {episodes} episodes แต่ละระดับ")
    print("กรุณารอสักครู่...\n")
    
    for diff in difficulties:
        print(f"กำลังทดสอบระดับ: {diff}")
        
        env = SimpleGridWorld(size=size, difficulty=diff)
        agent = SimpleQLearning(
            n_states=size * size,
            n_actions=4,
            learning_rate=0.1,
            discount=0.9,
            epsilon=0.3
        )
        
        agent.train(env, episodes=episodes)
        reward, steps, _ = agent.test(env, show_path=False)
        
        optimal_steps = env.get_optimal_path_length()
        success_rate = sum(1 for r in agent.episode_rewards[-100:] if r > 5) / min(100, len(agent.episode_rewards)) * 100
        
        results[diff] = {
            'obstacles': env.count_obstacles(),
            'final_reward': reward,
            'final_steps': steps,
            'optimal_steps': optimal_steps,
            'efficiency': (optimal_steps / steps * 100) if steps > 0 else 0,
            'success_rate': success_rate,
            'avg_early': sum(agent.episode_rewards[:100]) / 100,
            'avg_late': sum(agent.episode_rewards[-100:]) / 100
        }
        
        print(f"  ✓ เสร็จแล้ว - Final reward: {reward:.2f}, Steps: {steps}")
    
    # แสดงผลการเปรียบเทียบ
    print("\n" + "=" * 60)
    print("สรุปผลการเปรียบเทียบ:")
    print("=" * 60)
    
    headers = ["ระดับ", "อุปสรรค", "Reward", "Steps", "ประสิทธิภาพ", "ความสำเร็จ"]
    print(f"{headers[0]:<8} {headers[1]:<8} {headers[2]:<8} {headers[3]:<6} {headers[4]:<12} {headers[5]:<10}")
    print("-" * 60)
    
    for diff in difficulties:
        r = results[diff]
        print(f"{diff:<8} {r['obstacles']:<8} {r['final_reward']:<8.2f} {r['final_steps']:<6} "
              f"{r['efficiency']:<12.1f}% {r['success_rate']:<10.1f}%")
    
    print("\n📊 การวิเคราะห์:")
    print("- ระดับที่ยากขึ้น → อุปสรรคมากขึ้น → ใช้เวลาเรียนรู้นานขึ้น")
    print("- Maze level มักจะท้าทายที่สุดเพราะต้องหาเส้นทางที่ซับซ้อน")
    print("- ประสิทธิภาพจะลดลงเมื่อความซับซ้อนเพิ่มขึ้น")

def main():
    """ฟังก์ชันหลัก"""
    print("🤖 Enhanced Q-Learning Demo")
    print("ใช้ Python ธรรมดา ไม่ต้องติดตั้ง package เพิ่ม")
    print("🆕 รองรับระดับความยากหลายแบบ!")
    print()
    
    while True:
        print("เลือกโหมด:")
        print("1. Demo แบบง่าย (เลือกระดับความยากได้)")
        print("2. Interactive Demo (ปรับแต่งพารามิเตอร์)")
        print("3. เปรียบเทียบระดับความยาก (ใหม่!)")
        print("4. ออกจากโปรแกรม")
        print()
        
        choice = input("เลือก (1-4): ").strip()
        
        if choice == '1':
            demo_simple()
        elif choice == '2':
            interactive_demo()
        elif choice == '3':
            compare_difficulties()
        elif choice == '4':
            print("ขอบคุณที่ใช้งาน!")
            break
        else:
            print("กรุณาเลือก 1-4")
        
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
    main()
