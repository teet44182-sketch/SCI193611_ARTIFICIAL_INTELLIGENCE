#!/usr/bin/env python3
"""
ตัวอย่างการใช้งาน Q-Learning ระดับความยากต่างๆ
สำหรับอาจารย์ที่ต้องการสาธิตในชั้นเรียน
"""

from simple_q_learning import SimpleGridWorld, SimpleQLearning

def demo_for_class():
    """Demo สั้นๆ สำหรับในชั้นเรียน"""
    
    print("🎓 Q-Learning Class Demo")
    print("=" * 30)
    
    # แสดงความแตกต่างของ environment
    difficulties = ['easy', 'normal', 'hard', 'maze']
    
    print("1. แสดง Environment แต่ละระดับ:")
    for i, diff in enumerate(difficulties, 1):
        print(f"\n{i}. ระดับ {diff.upper()}:")
        env = SimpleGridWorld(size=4, difficulty=diff)
        print(f"   อุปสรรค: {env.count_obstacles()} ตำแหน่ง")
        print(f"   ระยะทางสั้นสุด: {env.get_optimal_path_length()} steps")
        env.print_grid()
    
    # ทดลองฝึก Hard level
    print("\n2. ทดลองฝึก Hard Level:")
    print("-" * 30)
    
    env = SimpleGridWorld(size=4, difficulty='hard')
    agent = SimpleQLearning(
        n_states=16,
        n_actions=4,
        learning_rate=0.1,
        discount=0.9,
        epsilon=0.3
    )
    
    print("ก่อนฝึก:")
    reward_before, steps_before, _ = agent.test(env, show_path=False)
    print(f"  Reward: {reward_before:.2f}, Steps: {steps_before}")
    
    print("\nฝึก 500 episodes...")
    agent.train(env, episodes=500)
    
    print("หลังฝึก:")
    reward_after, steps_after, _ = agent.test(env, show_path=False)
    print(f"  Reward: {reward_after:.2f}, Steps: {steps_after}")
    
    improvement = ((reward_after - reward_before) / abs(reward_before) * 100) if reward_before != 0 else 100
    print(f"  การพัฒนา: {improvement:.1f}%")
    
    print("\n3. Policy ที่เรียนรู้ได้:")
    policy = agent.get_policy()
    env.reset()
    env.print_grid(policy=policy)
    
 
    print("- เริ่มจาก Easy → Normal → Hard → Maze")
    print("- แสดงให้เห็นว่า complexity เพิ่มขึ้น → การเรียนรู้ยากขึ้น")
    print("- เปรียบเทียบ performance ในแต่ละระดับ")
    print("- ให้นักเรียนทดลองปรับพารามิเตอร์")

if __name__ == "__main__":
    demo_for_class()
