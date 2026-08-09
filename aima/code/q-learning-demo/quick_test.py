#!/usr/bin/env python3

# ทดสอบง่ายๆ ว่าระดับความยากใหม่ทำงานได้หรือไม่
import sys
sys.path.append('.')

try:
    from simple_q_learning import SimpleGridWorld
    
    print("🧪 ทดสอบระดับความยากใหม่")
    print("=" * 30)
    
    difficulties = ['easy', 'normal', 'hard', 'maze']
    
    for diff in difficulties:
        print(f"\n{diff.upper()}:")
        env = SimpleGridWorld(size=4, difficulty=diff)
        print(f"  อุปสรรค: {env.count_obstacles()} ตำแหน่ง")
        print(f"  ตำแหน่ง: {env.obstacles}")
        env.print_grid()
        
    print("\n✅ ทดสอบเสร็จสิ้น!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
