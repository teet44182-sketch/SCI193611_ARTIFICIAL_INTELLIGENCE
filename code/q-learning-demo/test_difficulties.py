#!/usr/bin/env python3
"""
ทดสอบระดับความยากใหม่ของ Q-learning
"""

from simple_q_learning import SimpleGridWorld, SimpleQLearning

def test_all_difficulties():
    """ทดสอบระดับความยากทั้งหมด"""
    print("🔥 ทดสอบระดับความยากใหม่!")
    print("=" * 50)
    
    difficulties = ['easy', 'normal', 'hard', 'maze']
    size = 5
    
    for diff in difficulties:
        print(f"\n📊 ทดสอบระดับ: {diff.upper()}")
        print("-" * 30)
        
        # สร้าง environment
        env = SimpleGridWorld(size=size, difficulty=diff)
        env.print_environment_info()
        env.print_grid()
        
        # สร้าง agent
        agent = SimpleQLearning(
            n_states=size * size,
            n_actions=4,
            learning_rate=0.1,
            discount=0.9,
            epsilon=0.3
        )
        
        # ฝึกสั้นๆ เพื่อดูผล
        print("กำลังฝึกสั้นๆ...")
        agent.train(env, episodes=300)
        
        # ทดสอบ
        reward, steps, _ = agent.test(env, show_path=False)
        optimal = env.get_optimal_path_length()
        
        print(f"ผลลัพธ์: Reward={reward:.2f}, Steps={steps}")
        print(f"เส้นทางที่สั้นที่สุด: {optimal} steps")
        print(f"ประสิทธิภาพ: {(optimal/steps*100) if steps > 0 else 0:.1f}%")

def test_maze_detailed():
    """ทดสอบโหมด maze แบบละเอียด"""
    print("\n🌀 ทดสอบโหมด MAZE แบบละเอียด")
    print("=" * 40)
    
    env = SimpleGridWorld(size=6, difficulty='maze')
    env.print_environment_info()
    env.print_grid()
    
    agent = SimpleQLearning(
        n_states=36,
        n_actions=4,
        learning_rate=0.1,
        discount=0.95,
        epsilon=0.4
    )
    
    print("ฝึก 800 episodes...")
    agent.train(env, episodes=800)
    
    print("\nทดสอบ learned policy:")
    reward, steps, path = agent.test(env, show_path=True)
    
    print(f"\nสรุป:")
    print(f"- Final Reward: {reward:.2f}")
    print(f"- Steps: {steps}")
    print(f"- Optimal Path: {env.get_optimal_path_length()} steps")
    print(f"- Efficiency: {(env.get_optimal_path_length()/steps*100) if steps > 0 else 0:.1f}%")

if __name__ == "__main__":
    test_all_difficulties()
    test_maze_detailed()
