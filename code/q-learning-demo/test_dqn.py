#!/usr/bin/env python3
"""
ทดสอบ Pure Python DQN แบบง่าย
"""

from pure_python_dqn import PurePythonDQN, compare_tabular_vs_pure_dqn

try:
    from simple_q_learning import SimpleGridWorld, SimpleQLearning
    
    print("🧠 ทดสอบ Pure Python DQN")
    print("=" * 30)
    
    # ทดสอบ Neural Network
    print("1. ทดสอบ Neural Network แบบง่าย:")
    from pure_python_dqn import SimpleMLP
    
    network = SimpleMLP(input_size=4, hidden_size=8, output_size=4, learning_rate=0.1)
    
    # ทดสอบ forward pass
    test_input = [1.0, 0.0, 0.0, 0.0]
    output = network.predict(test_input)
    print(f"  Input: {test_input}")
    print(f"  Output: {[f'{x:.3f}' for x in output]}")
    
    # ทดสอบ training
    target = [1.0, 0.0, 0.0, 0.0]
    network.train(test_input, target)
    output2 = network.predict(test_input)
    print(f"  After training: {[f'{x:.3f}' for x in output2]}")
    
    print("\n2. ทดสอบ DQN Agent:")
    
    # สร้าง environment เล็กๆ
    env = SimpleGridWorld(size=3, difficulty='easy')
    print("  Grid World 3x3:")
    env.print_grid()
    
    # สร้าง DQN agent
    agent = PurePythonDQN(
        state_size=9,
        action_size=4,
        learning_rate=0.05,
        epsilon=0.5
    )
    
    print("  ฝึก 100 episodes...")
    agent.train(env, episodes=100, verbose=False)
    
    # ทดสอบ
    env.reset()
    reward, steps, _ = agent.test(env, show_path=False)
    print(f"  ผลลัพธ์: Reward={reward:.2f}, Steps={steps}")
    
    print("\n3. เปรียบเทียบกับ Tabular Q-Learning:")
    
    # Tabular agent
    tabular = SimpleQLearning(9, 4, learning_rate=0.1, epsilon=0.3)
    tabular.train(env, episodes=100)
    env.reset()
    tab_reward, tab_steps, _ = tabular.test(env, show_path=False)
    
    print(f"  Tabular:  Reward={tab_reward:.2f}, Steps={tab_steps}")
    print(f"  DQN:      Reward={reward:.2f}, Steps={steps}")
    
    if abs(reward - tab_reward) < 1.0:
        print("  → ผลลัพธ์ใกล้เคียงกัน!")
    elif reward > tab_reward:
        print("  → DQN ดีกว่า!")
    else:
        print("  → Tabular ดีกว่า!")
    
    print("\n✅ ทดสอบสำเร็จ!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
