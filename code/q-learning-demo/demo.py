"""
Q-Learning Demo - ตัวอย่างการใช้งาน Q-learning
สำหรับใช้ประกอบการสอนและสาธิต

การใช้งาน:
1. รันไฟล์นี้เพื่อฝึก Q-learning agent
2. ดูผลการเรียนรู้และ visualizations ต่างๆ

หมายเหตุ: ต้องติดตั้ง numpy และ matplotlib ก่อน
pip install numpy matplotlib
"""

import sys
import os

# เพิ่ม path สำหรับ import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import numpy as np
    import matplotlib.pyplot as plt
    from grid_world import GridWorld
    from q_learning import QLearningAgent
    HAS_PACKAGES = True
except ImportError as e:
    print(f"Warning: ไม่สามารถ import packages ได้: {e}")
    print("กรุณาติดตั้ง: pip install numpy matplotlib")
    HAS_PACKAGES = False

def run_basic_demo():
    """รันตัวอย่างพื้นฐานของ Q-learning (ไม่ต้องใช้ numpy/matplotlib)"""
    print("=== Q-Learning Demo (Basic Version) ===")
    print("นี่คือการจำลอง Q-learning แบบง่าย ๆ")
    print()
    
    # สร้าง simple grid world 3x3
    grid_size = 3
    print(f"Grid World ขนาด {grid_size}x{grid_size}:")
    print("S = Start (0,0)")
    print("G = Goal (2,2)")
    print("X = Obstacle (1,1)")
    print()
    
    # แสดง grid
    for i in range(grid_size):
        row_str = ""
        for j in range(grid_size):
            if (i, j) == (0, 0):
                row_str += "S "
            elif (i, j) == (2, 2):
                row_str += "G "
            elif (i, j) == (1, 1):
                row_str += "X "
            else:
                row_str += ". "
        print(row_str)
    print()
    
    # จำลอง Q-learning process
    print("การเรียนรู้ Q-learning:")
    print("- เริ่มต้นด้วย Q-values = 0 ทั้งหมด")
    print("- ใช้ epsilon-greedy policy (ε = 0.1)")
    print("- Learning rate α = 0.1")
    print("- Discount factor γ = 0.9")
    print()
    
    # แสดงตัวอย่าง Q-learning update
    print("ตัวอย่าง Q-learning update:")
    print("หาก agent อยู่ที่ (0,0) และเลือก action 'right'")
    print("ไปยัง state (0,1) และได้ reward = -0.1")
    print()
    print("Q-learning formula:")
    print("Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]")
    print()
    print("ตัวอย่างการคำนวณ:")
    print("Q(0,right) ← 0 + 0.1 × [-0.1 + 0.9 × 0 - 0]")
    print("Q(0,right) ← 0 + 0.1 × (-0.1)")
    print("Q(0,right) ← -0.01")
    print()
    
    print("หลังจากฝึก 1000 episodes:")
    print("Agent จะเรียนรู้ที่จะหาเส้นทางที่ดีที่สุดไปยังเป้าหมาย")
    print("และหลีกเลี่ยงอุปสรรค")

def run_full_demo():
    """รันตัวอย่างเต็มรูปแบบพร้อม visualizations"""
    print("=== Q-Learning Demo (Full Version) ===")
    
    # สร้าง environment
    env = GridWorld(size=5)
    print(f"สร้าง Grid World ขนาด {env.size}x{env.size}")
    print(f"Start: {env.start_pos}, Goal: {env.goal_pos}")
    print(f"Obstacles: {env.obstacles}")
    print()
    
    # สร้าง agent
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.1
    )
    
    print("เริ่มการฝึก Q-learning...")
    
    # ฝึก agent
    agent.train(env, n_episodes=1000, verbose=True)
    
    print("\nการฝึกเสร็จสิ้น!")
    
    # ทดสอบ agent
    print("\nทดสอบ trained agent:")
    total_reward, steps, path = agent.test_episode(env)
    print(f"Total reward: {total_reward:.2f}")
    print(f"Steps taken: {steps}")
    print(f"Path: {path}")
    
    # แสดง Q-table
    print("\nQ-table (บางส่วน):")
    print("State | Up    | Down  | Left  | Right")
    print("-" * 40)
    for state in range(min(10, env.n_states)):
        q_vals = agent.q_table[state]
        print(f"{state:5d} | {q_vals[0]:5.2f} | {q_vals[1]:5.2f} | {q_vals[2]:5.2f} | {q_vals[3]:5.2f}")
    
    # สร้าง visualizations
    create_visualizations(env, agent)

def create_visualizations(env, agent):
    """สร้างภาพแสดงผลต่างๆ"""
    print("\nสร้าง visualizations...")
    
    # 1. Learning curve
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    window_size = 50
    if len(agent.episode_rewards) >= window_size:
        smoothed_rewards = np.convolve(agent.episode_rewards, 
                                     np.ones(window_size)/window_size, mode='valid')
        plt.plot(smoothed_rewards)
    else:
        plt.plot(agent.episode_rewards)
    plt.title('Learning Curve (Smoothed Rewards)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(True)
    
    # 2. Grid world with Q-values
    plt.subplot(1, 3, 2)
    plt.title('Grid World Environment')
    env.visualize()
    
    # 3. Optimal Policy
    plt.subplot(1, 3, 3)
    plt.title('Learned Policy')
    policy = agent.get_policy()
    env.visualize(policy=policy)
    
    plt.tight_layout()
    plt.savefig('q_learning_results.png', dpi=150, bbox_inches='tight')
    print("บันทึกภาพผลลัพธ์เป็น 'q_learning_results.png'")
    
    # แสดงผล
    plt.show()

def interactive_demo():
    """Demo แบบ interactive"""
    if not HAS_PACKAGES:
        print("Interactive demo ต้องการ numpy และ matplotlib")
        return
    
    print("=== Interactive Q-Learning Demo ===")
    print("คุณสามารถปรับแต่งพารามิเตอร์ต่างๆ ได้")
    print()
    
    # รับ input จากผู้ใช้
    try:
        grid_size = int(input("ขนาด grid (3-8, default=5): ") or "5")
        learning_rate = float(input("Learning rate (0.01-1.0, default=0.1): ") or "0.1")
        epsilon = float(input("Epsilon (0.01-1.0, default=0.1): ") or "0.1")
        episodes = int(input("จำนวน episodes (100-5000, default=1000): ") or "1000")
    except ValueError:
        print("ใช้ค่า default")
        grid_size, learning_rate, epsilon, episodes = 5, 0.1, 0.1, 1000
    
    # สร้างและรัน experiment
    env = GridWorld(size=grid_size)
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=learning_rate,
        discount_factor=0.95,
        epsilon=epsilon
    )
    
    print(f"\nเริ่มฝึกด้วยพารามิเตอร์:")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epsilon: {epsilon}")
    print(f"Episodes: {episodes}")
    print()
    
    agent.train(env, n_episodes=episodes, verbose=True)
    
    # แสดงผลลัพธ์
    total_reward, steps, path = agent.test_episode(env)
    print(f"\nผลลัพธ์การทดสอบ:")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Steps taken: {steps}")
    
    # สร้างภาพ
    create_visualizations(env, agent)

def main():
    """ฟังก์ชันหลัก"""
    print("🤖 Q-Learning Demo for AI Course")
    print("ตัวอย่างการใช้งาน Q-learning สำหรับบทเรียน AI")
    print("=" * 50)
    print()
    
    if HAS_PACKAGES:
        print("เลือกโหมดการสาธิต:")
        print("1. Basic Demo (การจำลองพื้นฐาน)")
        print("2. Full Demo (พร้อม visualizations)")
        print("3. Interactive Demo (ปรับแต่งพารามิเตอร์ได้)")
        print("4. ออกจากโปรแกรม")
        print()
        
        while True:
            try:
                choice = input("เลือก (1-4): ").strip()
                if choice == '1':
                    run_basic_demo()
                    break
                elif choice == '2':
                    run_full_demo()
                    break
                elif choice == '3':
                    interactive_demo()
                    break
                elif choice == '4':
                    print("ขอบคุณที่ใช้งาน!")
                    break
                else:
                    print("กรุณาเลือก 1-4")
            except KeyboardInterrupt:
                print("\nขอบคุณที่ใช้งาน!")
                break
    else:
        print("รันโหมดพื้นฐาน (ไม่ต้องใช้ external packages)")
        run_basic_demo()

if __name__ == "__main__":
    main()
