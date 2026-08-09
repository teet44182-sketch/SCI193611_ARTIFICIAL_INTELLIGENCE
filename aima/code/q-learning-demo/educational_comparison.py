#!/usr/bin/env python3
"""
สาธิตการเปรียบเทียบ Tabular Q-Learning กับ Neural Network Function Approximation
สำหรับใช้ในการสอน
"""

from simple_q_learning import SimpleGridWorld, SimpleQLearning
from pure_python_dqn import PurePythonDQN

def educational_comparison():
    """เปรียบเทียบแบบ step-by-step สำหรับการสอน"""
    
    print("📚 Tabular Q-Learning vs Neural Network Function Approximation")
    print("=" * 70)
    print()
    
    # สร้าง environment
    env = SimpleGridWorld(size=4, difficulty='normal')
    print("🌍 Environment Setup:")
    env.print_environment_info()
    env.print_grid()
    
    print("\n" + "="*70)
    print("📊 การเปรียบเทียบ 2 แนวทาง")
    print("="*70)
    
    # 1. Tabular Q-Learning
    print("\n🗂️  1. TABULAR Q-LEARNING")
    print("-" * 30)
    print("💡 แนวคิด: เก็บ Q-value แยกแต่ละ (state, action) pair")
    print("📝 ข้อมูล: Q-table ขนาด 16 states × 4 actions = 64 values")
    
    tabular_agent = SimpleQLearning(
        n_states=16,
        n_actions=4,
        learning_rate=0.1,
        discount=0.9,
        epsilon=0.3
    )
    
    print("🏃 ฝึก 300 episodes...")
    tabular_agent.train(env, episodes=300)
    
    env.reset()
    tab_reward, tab_steps, _ = tabular_agent.test(env, show_path=False)
    
    print(f"✅ ผลลัพธ์: Reward = {tab_reward:.2f}, Steps = {tab_steps}")
    
    # แสดง Q-table บางส่วน
    print("\n📋 Q-table ตัวอย่าง (5 states แรก):")
    for state in range(5):
        q_vals = tabular_agent.q_table[state]
        pos = env.state_to_pos(state)
        print(f"   State {state}{pos}: {[f'{x:.2f}' for x in q_vals]}")
    
    # 2. Neural Network DQN
    print("\n🧠 2. NEURAL NETWORK FUNCTION APPROXIMATION (DQN)")
    print("-" * 50)
    print("💡 แนวคิด: ใช้ neural network เรียนรู้ function Q(s,a)")
    print("📝 โครงสร้าง: Input(16) → Hidden(32) → Output(4)")
    print("🔗 พารามิเตอร์: ~580 weights ใน neural network")
    
    dqn_agent = PurePythonDQN(
        state_size=16,
        action_size=4,
        learning_rate=0.02,
        epsilon=1.0,
        epsilon_decay=0.99
    )
    
    print("🏃 ฝึก 300 episodes...")
    dqn_agent.train(env, episodes=300, verbose=False)
    
    env.reset()
    dqn_reward, dqn_steps, _ = dqn_agent.test(env, show_path=False)
    
    print(f"✅ ผลลัพธ์: Reward = {dqn_reward:.2f}, Steps = {dqn_steps}")
    
    # แสดง neural network predictions
    print("\n📋 Neural Network Q-values ตัวอย่าง (5 states แรก):")
    for state in range(5):
        features = dqn_agent.state_to_features(state)
        q_vals = dqn_agent.q_network.predict(features)
        pos = env.state_to_pos(state)
        print(f"   State {state}{pos}: {[f'{x:.2f}' for x in q_vals]}")
    
    # 3. การเปรียบเทียบ
    print("\n" + "="*70)
    print("🔍 การวิเคราะห์เปรียบเทียบ")
    print("="*70)
    
    print(f"\n📊 ผลลัพธ์:")
    print(f"   Tabular Q-Learning: {tab_reward:.2f} reward, {tab_steps} steps")
    print(f"   Neural Network DQN: {dqn_reward:.2f} reward, {dqn_steps} steps")
    
    if abs(tab_reward - dqn_reward) < 1.0:
        winner = "ผลลัพธ์ใกล้เคียงกัน"
    elif tab_reward > dqn_reward:
        winner = "Tabular Q-Learning ดีกว่า"
    else:
        winner = "Neural Network DQN ดีกว่า"
    
    print(f"   🏆 {winner}")
    
    print(f"\n📈 ข้อดี/ข้อเสีย:")
    print("   Tabular Q-Learning:")
    print("   ✅ เรียนรู้เร็ว, เสถียร, แน่นอน")
    print("   ❌ จำกัดด้วยขนาด state space")
    print("   🎯 เหมาะกับ: ปัญหาที่ state space เล็ก")
    
    print("\n   Neural Network DQN:")
    print("   ✅ รองรับ state space ใหญ่, generalization")
    print("   ❌ เรียนรู้ช้า, ไม่เสถียร, ซับซ้อน")
    print("   🎯 เหมาะกับ: ปัญหาที่ state space ใหญ่มาก")
    
    print(f"\n🎓 บทเรียนสำคัญ:")
    print("   • ในปัญหา Grid World เล็ก → Tabular มักจะดีกว่า")
    print("   • ในปัญหาใหญ่ (เช่น Atari games) → Neural Network จำเป็น")
    print("   • Function approximation ช่วยให้ generalize ได้")
    print("   • Trade-off ระหว่าง simplicity กับ scalability")

def demonstrate_learning_curves():
    """แสดงการเรียนรู้แบบ step-by-step"""
    print("\n" + "="*70)
    print("📈 การสาธิตเส้นโค้งการเรียนรู้")
    print("="*70)
    
    env = SimpleGridWorld(size=3, difficulty='easy')  # ใช้ environment เล็กเพื่อความชัดเจน
    print("ใช้ Grid World 3x3 เพื่อความง่าย:")
    env.print_grid()
    
    # สร้าง agents
    tabular = SimpleQLearning(9, 4, learning_rate=0.1, epsilon=0.3)
    dqn = PurePythonDQN(9, 4, learning_rate=0.05, epsilon=0.5)
    
    print("\nการเรียนรู้ทีละ 25 episodes:")
    
    for episode_batch in [25, 50, 75, 100]:
        print(f"\nEpisode {episode_batch}:")
        
        # ฝึก
        tabular.train(env, episodes=25)
        dqn.train(env, episodes=25, verbose=False)
        
        # ทดสอบ
        env.reset()
        tab_r, tab_s, _ = tabular.test(env, show_path=False)
        env.reset()
        dqn_r, dqn_s, _ = dqn.test(env, show_path=False)
        
        print(f"  Tabular: {tab_r:.1f} reward, {tab_s} steps")
        print(f"  DQN:     {dqn_r:.1f} reward, {dqn_s} steps")
    
    print("\n💡 สังเกต: Tabular มักจะ converge เร็วกว่าในปัญหาเล็ก")

def main():
    """ฟังก์ชันหลัก"""
    print("🎯 Educational Demo: Tabular vs Function Approximation")
    print("สำหรับใช้ประกอบการสอน Reinforcement Learning")
    print()
    
    try:
        educational_comparison()
        demonstrate_learning_curves()
        
        print(f"\n🎉 การสาธิตเสร็จสมบูรณ์!")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
