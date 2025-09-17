#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 MapSage V5 - 分阶段知识蒸馏实验

基于深度分析的改进策略：
1. 三阶段训练策略
2. 选择性知识蒸馏
3. 动态架构匹配
4. 智能损失平衡

Author: MapSage Team
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from datetime import datetime
import json
import os
from collections import defaultdict

# 🎯 分阶段蒸馏配置
STAGED_DISTILL_CONFIG = {
    'stage1_task_focus': {
        'epochs': 8,
        'task_weight': 1.0,
        'distill_weight': 0.0,
        'feature_weight': 0.0,
        'learning_rate': 0.001,
        'description': '纯任务学习阶段 - 建立基础任务能力'
    },
    'stage2_gentle_distill': {
        'epochs': 12,
        'task_weight': 0.8,
        'distill_weight': 0.15,
        'feature_weight': 0.05,
        'learning_rate': 0.0008,
        'description': '温和蒸馏阶段 - 逐步引入教师知识'
    },
    'stage3_balanced_learning': {
        'epochs': 8,
        'task_weight': 0.65,
        'distill_weight': 0.25,
        'feature_weight': 0.1,
        'learning_rate': 0.0005,
        'description': '平衡学习阶段 - 优化最终性能'
    }
}

# 🧠 选择性知识蒸馏模块
class SelectiveDistillationModule(nn.Module):
    """基于注意力的选择性知识蒸馏"""
    
    def __init__(self, teacher_dim, student_dim):
        super().__init__()
        
        # 特征维度适配
        self.feature_adapter = nn.Sequential(
            nn.Linear(teacher_dim, student_dim),
            nn.LayerNorm(student_dim),
            nn.ReLU()
        )
        
        # 任务相关性评估网络
        self.relevance_gate = nn.Sequential(
            nn.Linear(teacher_dim, teacher_dim // 4),
            nn.ReLU(),
            nn.Linear(teacher_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # 知识质量评估
        self.quality_assessor = nn.Sequential(
            nn.Linear(student_dim, student_dim // 2),
            nn.ReLU(),
            nn.Linear(student_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, teacher_features, student_features, task_loss=None):
        """选择性知识蒸馏前向传播"""
        
        # 评估教师特征的任务相关性
        relevance_scores = self.relevance_gate(teacher_features)
        
        # 选择性特征传递
        selected_teacher = teacher_features * relevance_scores
        adapted_teacher = self.feature_adapter(selected_teacher)
        
        # 评估学生特征质量
        student_quality = self.quality_assessor(student_features)
        
        # 计算选择性蒸馏损失
        distill_loss = F.mse_loss(student_features, adapted_teacher)
        
        # 质量加权
        if task_loss is not None:
            # 任务损失高时，降低蒸馏权重
            quality_weight = torch.exp(-task_loss)
            distill_loss = distill_loss * quality_weight
        
        return {
            'distill_loss': distill_loss,
            'relevance_scores': relevance_scores.mean(),
            'student_quality': student_quality.mean(),
            'adapted_features': adapted_teacher
        }

# 🏗️ 渐进式学生模型
class ProgressiveStudentModel(nn.Module):
    """动态复杂度调整的学生模型"""
    
    def __init__(self, input_dim=128, hidden_dims=[64, 32], output_dim=10):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.current_depth = 1  # 当前激活的层数
        
        # 构建渐进式层结构
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layer = nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            self.layers.append(layer)
            prev_dim = hidden_dim
        
        # 输出层 - 需要根据当前激活层调整输入维度
        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_dims[i], output_dim) for i in range(len(hidden_dims))
        ])
        
        # 特征提取器（用于蒸馏）- 动态维度
        self.feature_extractors = nn.ModuleList([
            nn.Linear(hidden_dims[i], 64) for i in range(len(hidden_dims))
        ])
    
    def expand_model(self, stage):
        """根据训练阶段扩展模型复杂度"""
        if stage == 'stage1_task_focus':
            self.current_depth = 1
        elif stage == 'stage2_gentle_distill':
            self.current_depth = min(2, len(self.layers))
        elif stage == 'stage3_balanced_learning':
            self.current_depth = len(self.layers)
        
        print(f"📈 模型复杂度调整: 激活 {self.current_depth}/{len(self.layers)} 层")
    
    def forward(self, x, return_features=False):
        """前向传播"""
        
        # 渐进式前向传播
        layer_outputs = []
        for i in range(self.current_depth):
            x = self.layers[i](x)
            layer_outputs.append(x)
        
        # 提取特征（用于蒸馏）- 使用当前激活层的最后一层
        if return_features and layer_outputs:
            current_layer_idx = min(self.current_depth - 1, len(self.feature_extractors) - 1)
            features = self.feature_extractors[current_layer_idx](layer_outputs[-1])
        else:
            features = None
        
        # 输出预测 - 使用对应的输出层
        if layer_outputs:
            current_layer_idx = min(self.current_depth - 1, len(self.output_layers) - 1)
            output = self.output_layers[current_layer_idx](layer_outputs[-1])
        else:
            # 如果没有激活层，使用第一个输出层
            output = self.output_layers[0](x)
        
        if return_features:
            return output, features
        return output

# 🎓 智能教师模型
class IntelligentTeacherModel(nn.Module):
    """增强版教师模型"""
    
    def __init__(self, input_dim=128, hidden_dims=[256, 128, 64], output_dim=10):
        super().__init__()
        
        # 构建教师网络
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        # 多层特征提取
        self.feature_extractors = nn.ModuleDict({
            'shallow': nn.Linear(hidden_dims[0], 64),
            'middle': nn.Linear(hidden_dims[1], 64),
            'deep': nn.Linear(hidden_dims[2], 64)
        })
    
    def forward(self, x, return_features=False):
        """前向传播"""
        
        features = {}
        
        # 逐层前向传播并提取特征
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            
            # 在关键层提取特征
            if i == 3:  # 第一个隐藏层后
                features['shallow'] = self.feature_extractors['shallow'](x)
            elif i == 7:  # 第二个隐藏层后
                features['middle'] = self.feature_extractors['middle'](x)
            elif i == 11:  # 第三个隐藏层后
                features['deep'] = self.feature_extractors['deep'](x)
        
        output = self.output_layer(x)
        
        if return_features:
            return output, features
        return output

# 🚀 分阶段蒸馏训练器
class StagedDistillationTrainer:
    """分阶段知识蒸馏训练器"""
    
    def __init__(self, config=None):
        self.config = config or STAGED_DISTILL_CONFIG
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型
        self.teacher_model = IntelligentTeacherModel().to(self.device)
        self.student_model = ProgressiveStudentModel().to(self.device)
        
        # 选择性蒸馏模块
        self.selective_distill = SelectiveDistillationModule(
            teacher_dim=64, student_dim=64
        ).to(self.device)
        
        # 训练历史
        self.training_history = defaultdict(list)
        self.stage_results = {}
        
        # 预训练教师模型
        self._pretrain_teacher()
    
    def _pretrain_teacher(self):
        """预训练教师模型"""
        print("🎓 预训练教师模型...")
        
        # 简单的预训练过程
        optimizer = torch.optim.Adam(self.teacher_model.parameters(), lr=0.001)
        
        for epoch in range(10):
            # 模拟训练数据
            x = torch.randn(64, 128).to(self.device)
            y = torch.randint(0, 10, (64,)).to(self.device)
            
            optimizer.zero_grad()
            output = self.teacher_model(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            optimizer.step()
            
            if epoch % 3 == 0:
                print(f"    教师预训练 Epoch {epoch+1}/10, Loss: {loss.item():.4f}")
        
        self.teacher_model.eval()
        print("✅ 教师模型预训练完成")
    
    def _create_synthetic_data(self, batch_size=32):
        """创建合成训练数据"""
        x = torch.randn(batch_size, 128).to(self.device)
        y = torch.randint(0, 10, (batch_size,)).to(self.device)
        return x, y
    
    def train_stage(self, stage_name, stage_config):
        """训练单个阶段"""
        print(f"\n🚀 开始 {stage_name}: {stage_config['description']}")
        print(f"    训练轮数: {stage_config['epochs']}")
        print(f"    损失权重: 任务={stage_config['task_weight']:.2f}, "
              f"蒸馏={stage_config['distill_weight']:.2f}, "
              f"特征={stage_config['feature_weight']:.2f}")
        
        # 调整学生模型复杂度
        self.student_model.expand_model(stage_name)
        
        # 设置优化器
        optimizer = torch.optim.Adam(
            list(self.student_model.parameters()) + 
            list(self.selective_distill.parameters()),
            lr=stage_config['learning_rate']
        )
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=stage_config['epochs']
        )
        
        stage_losses = []
        
        for epoch in range(stage_config['epochs']):
            epoch_losses = {'total': 0.0, 'task': 0.0, 'distill': 0.0, 'feature': 0.0}
            num_batches = 20  # 每个epoch的批次数
            
            for batch in range(num_batches):
                # 生成训练数据
                x, y = self._create_synthetic_data()
                
                optimizer.zero_grad()
                
                # 学生模型前向传播
                student_output, student_features = self.student_model(x, return_features=True)
                
                # 教师模型前向传播（无梯度）
                with torch.no_grad():
                    teacher_output, teacher_features = self.teacher_model(x, return_features=True)
                
                # 计算任务损失
                task_loss = F.cross_entropy(student_output, y)
                
                # 计算蒸馏损失
                distill_loss = torch.tensor(0.0).to(self.device)
                feature_loss = torch.tensor(0.0).to(self.device)
                
                if stage_config['distill_weight'] > 0:
                    # 输出蒸馏
                    temperature = 4.0
                    soft_teacher = F.softmax(teacher_output / temperature, dim=1)
                    soft_student = F.log_softmax(student_output / temperature, dim=1)
                    distill_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
                
                if stage_config['feature_weight'] > 0:
                    # 选择性特征蒸馏
                    distill_result = self.selective_distill(
                        teacher_features['deep'], student_features, task_loss
                    )
                    feature_loss = distill_result['distill_loss']
                
                # 总损失
                total_loss = (
                    stage_config['task_weight'] * task_loss +
                    stage_config['distill_weight'] * distill_loss +
                    stage_config['feature_weight'] * feature_loss
                )
                
                # 反向传播
                total_loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    list(self.student_model.parameters()) + 
                    list(self.selective_distill.parameters()),
                    max_norm=1.0
                )
                optimizer.step()
                
                # 记录损失
                epoch_losses['total'] += float(total_loss.item())
                epoch_losses['task'] += float(task_loss.item())
                epoch_losses['distill'] += float(distill_loss.item())
                epoch_losses['feature'] += float(feature_loss.item())
            
            # 平均损失
            for key in epoch_losses:
                epoch_losses[key] = float(epoch_losses[key]) / num_batches
            
            stage_losses.append(epoch_losses)
            scheduler.step()
            
            # 打印进度
            if (epoch + 1) % 3 == 0 or epoch == stage_config['epochs'] - 1:
                print(f"    Epoch {epoch+1:2d}/{stage_config['epochs']:2d}: "
                      f"总损失={epoch_losses['total']:.4f}, "
                      f"任务={epoch_losses['task']:.4f}, "
                      f"蒸馏={epoch_losses['distill']:.4f}, "
                      f"特征={epoch_losses['feature']:.4f}, "
                      f"LR={scheduler.get_last_lr()[0]:.6f}")
        
        # 保存阶段结果
        initial_task_loss = stage_losses[0]['task'] if stage_losses else 0.0
        final_task_loss = stage_losses[-1]['task'] if stage_losses else 0.0
        improvement = initial_task_loss - final_task_loss
        
        self.stage_results[stage_name] = {
            'config': stage_config,
            'losses': stage_losses,
            'final_task_loss': final_task_loss,
            'improvement': improvement
        }
        
        print(f"✅ {stage_name} 完成")
        print(f"    任务损失改善: {self.stage_results[stage_name]['improvement']:.4f}")
        
        return stage_losses
    
    def run_full_experiment(self):
        """运行完整的分阶段实验"""
        print("🎯 开始分阶段知识蒸馏实验")
        print(f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_time = time.time()
        
        # 依次执行各个阶段
        for stage_name, stage_config in self.config.items():
            self.train_stage(stage_name, stage_config)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n🎉 分阶段实验完成! 总用时: {total_time:.2f}秒")
        
        # 分析结果
        self.analyze_results()
        
        return self.stage_results
    
    def analyze_results(self):
        """分析实验结果"""
        print("\n📊 分阶段实验结果分析:")
        print("=" * 60)
        
        total_improvement = 0
        initial_loss = None
        final_loss = None
        
        for stage_name, results in self.stage_results.items():
            config = results['config']
            improvement = results['improvement']
            final_task_loss = results['final_task_loss']
            
            if initial_loss is None:
                initial_loss = results['losses'][0]['task']
            final_loss = final_task_loss
            
            print(f"\n📈 {stage_name}:")
            print(f"    描述: {config['description']}")
            print(f"    训练轮数: {config['epochs']}")
            print(f"    任务损失改善: {improvement:.4f} ({improvement/results['losses'][0]['task']*100:.1f}%)")
            print(f"    最终任务损失: {final_task_loss:.4f}")
            
            total_improvement += improvement
        
        # 整体分析
        overall_improvement = (initial_loss - final_loss) / initial_loss * 100
        
        print(f"\n🎯 整体实验效果:")
        print(f"    初始任务损失: {initial_loss:.4f}")
        print(f"    最终任务损失: {final_loss:.4f}")
        print(f"    总体改善: {total_improvement:.4f} ({overall_improvement:.1f}%)")
        
        # 成功评估
        if overall_improvement > 15:
            print("\n✅ 实验成功! 分阶段策略显著改善了性能")
            print("    建议: 在完整数据集上验证并进一步优化")
        elif overall_improvement > 5:
            print("\n⚠️  实验部分成功，有改善但仍有优化空间")
            print("    建议: 调整各阶段的损失权重和训练轮数")
        else:
            print("\n❌ 实验效果有限，需要重新设计策略")
            print("    建议: 检查模型架构匹配度和数据质量")
        
        # 保存结果
        self.save_results()
    
    def save_results(self):
        """保存实验结果"""
        results_dir = "experiments/staged_distillation"
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"{results_dir}/staged_distill_results_{timestamp}.json"
        
        # 准备保存数据
        save_data = {
            'timestamp': timestamp,
            'config': self.config,
            'results': {}
        }
        
        for stage_name, results in self.stage_results.items():
            save_data['results'][stage_name] = {
                'config': results['config'],
                'final_task_loss': results['final_task_loss'],
                'improvement': results['improvement'],
                'loss_history': [loss['task'] for loss in results['losses']]
            }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 实验结果已保存至: {results_file}")

# 🎯 主函数
def main():
    """主函数 - 运行分阶段知识蒸馏实验"""
    print("🎯 MapSage V5 - 分阶段知识蒸馏实验")
    print("=" * 50)
    
    # 创建训练器
    trainer = StagedDistillationTrainer()
    
    # 运行实验
    results = trainer.run_full_experiment()
    
    print("\n🎯 分阶段知识蒸馏实验完成!")
    return results

if __name__ == "__main__":
    results = main()