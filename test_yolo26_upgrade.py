#!/usr/bin/env python3
"""
YOLO26 升级验证测试脚本

测试升级后的系统是否正常工作。
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试关键模块导入"""
    print("=" * 60)
    print("测试 1: 关键模块导入")
    print("=" * 60)
    
    tests = [
        ("YOLO26 模型配置", "ultralytics.cfg.models.26", "yolo26"),
        ("Muon 优化器", "ultralytics.optim", "Muon"),
        ("Pose26 类", "ultralytics.nn.modules.head", "Pose26"),
        ("OBB26 类", "ultralytics.nn.modules.head", "OBB26"),
        ("Segment26 类", "ultralytics.nn.modules.head", "Segment26"),
        ("v8PoseLoss", "ultralytics.utils.loss", "v8PoseLoss"),
        ("PoseLoss26", "ultralytics.utils.loss", "PoseLoss26"),
        ("RLELoss", "ultralytics.utils.loss", "RLELoss"),
        ("自定义 Pose 类", "ultralytics.nn.modules.head", "Pose"),
        ("ArmorPoseModel", "armor_detect.models", "ArmorPoseModel"),
        ("ArmorPoseLoss", "armor_detect.losses", "ArmorPoseLoss"),
    ]
    
    passed = 0
    failed = 0
    
    for name, module_path, class_name in tests:
        try:
            module = __import__(module_path, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✓ {name}")
            passed += 1
        except Exception as e:
            print(f"✗ {name}: {e}")
            failed += 1
    
    print(f"\n结果: {passed} 通过, {failed} 失败")
    return failed == 0


def test_config():
    """测试配置文件"""
    print("\n" + "=" * 60)
    print("测试 2: 配置文件验证")
    print("=" * 60)
    
    # 检查 YOLO26 配置
    yolo26_dir = "ultralytics/cfg/models/26"
    if os.path.exists(yolo26_dir):
        configs = [f for f in os.listdir(yolo26_dir) if f.endswith('.yaml')]
        print(f"✓ YOLO26 配置文件: {len(configs)} 个")
        for cfg in configs:
            print(f"  - {cfg}")
    else:
        print("✗ YOLO26 配置目录不存在")
        return False
    
    # 检查默认配置
    with open("ultralytics/cfg/default.yaml") as f:
        content = f.read()
    
    checks = [
        ("multi_scale: 0.0", "multi_scale 改为浮点数"),
        ("rle: 1.0", "RLE 损失权重"),
        ("angle: 1.0", "角度损失权重"),
        ("color: 1.0", "装甲板颜色分类权重"),
        ("size: 1.0", "装甲板尺寸分类权重"),
        ("wing_omega:", "WingLoss omega 参数"),
        ("focal_gamma:", "Focal Loss gamma"),
    ]
    
    passed = 0
    for pattern, desc in checks:
        if pattern in content:
            print(f"✓ {desc}")
            passed += 1
        else:
            print(f"✗ {desc}")
    
    return passed == len(checks)


def test_head_classes():
    """测试 Head 类的兼容性"""
    print("\n" + "=" * 60)
    print("测试 3: Head 类兼容性")
    print("=" * 60)
    
    try:
        from ultralytics.nn.modules.head import Pose, Pose26, Detect
        from ultralytics.nn.modules import __all__ as head_all
        
        print(f"✓ 可导入的 head 类: {[x for x in ['Pose', 'Pose26', 'Detect', 'OBB26', 'Segment26'] if x in dir()]}")
        
        # 检查 Pose 类的属性
        pose_attrs = dir(Pose)
        print(f"✓ Pose 类属性数: {len(pose_attrs)}")
        
        # 检查 Pose26 类的属性
        pose26_attrs = dir(Pose26)
        print(f"✓ Pose26 类属性数: {len(pose26_attrs)}")
        
        return True
    except Exception as e:
        print(f"✗ Head 类测试失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║         YOLO26 升级验证测试                                  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    
    results = []
    results.append(("导入测试", test_imports()))
    results.append(("配置测试", test_config()))
    results.append(("Head 类测试", test_head_classes()))
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有测试通过！升级成功。")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
