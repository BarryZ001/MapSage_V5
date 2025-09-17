#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 MapSage V5 - 任务导向知识蒸馏实验

基于综合分析的最终改进方案：
1. 任务导向的知识选择性传递
2. 对抗式知识质量验证
3. 动态蒸馏策略调整
4. 多维度性能评估

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

# 🎯 任务导向蒸馏配置
TASK_ORIENTED_CONFIG = {
    'model': {
        'teacher_dim': 256,
        'student_dim': 128,
        'hidden_dim': 64,
        'num_classes': 10
    },
    'training': {
        'total_epochs': 30,
        'warmup_epochs': 8,
        'learning_rate': 0.001,
        'batch_size': 32,
        'num_batches': 25
    },
    'distillation': {
        'initial_task_weight': 0.8,
        'final_task_weight': 0.6,
        'initial_distill_weight': 0.1,
        'final_distill_weight': 0.3,
        'adversarial_weight': 0.1,
        'relevance_threshold': 0.6,
        'quality_threshold': 0.7
    }
}

# 🧠 任务导向知识过滤器
class TaskOrientedKnowledgeFilter(nn.Module):
    """任务导向的知识选择性传递模块"""
    
    def __init__(self, teacher_dim, student_dim, hidden_dim=64):
        super().__init__()
        
        self.teacher_dim = teacher_dim
        self.student_dim = student_dim
        
        # 任务相关性评估网络
        self.relevance_assessor = nn.Sequential(
            nn.Linear(teacher_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 知识质量评估网络
        self.quality_assessor = nn.Sequential(
            nn.Linear(teacher_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 特征适配器
        self.feature_adapter = nn.Sequential(
            nn.Linear(teacher_dim, student_dim),
            nn.LayerNorm(student_dim),
            nn.ReLU()
        )
        
        # 知识融合网络
        self.knowledge_fusion = nn.Sequential(
            nn.Linear(student_dim * 2, student_dim),
            nn.ReLU(),
            nn.Linear(student_dim, student_dim)
        )
    
    def forward(self, teacher_features, student_features, task_loss=None):
        """任务导向的知识过滤和传递"""
        
        batch_size = teacher_features.size(0)
        
        # 评估教师知识的任务相关性
        relevance_scores = self.relevance_assessor(teacher_features)
        quality_scores = self.quality_assessor(teacher_features)
        
        # 计算综合知识价值分数
        knowledge_value = relevance_scores * quality_scores
        
        # 如果有任务损失信息，调整知识价值
        if task_loss is not None:
            # 任务损失高时，提高知识选择的严格程度
            task_difficulty = torch.sigmoid(task_loss)
            knowledge_threshold = 0.5 + 0.3 * task_difficulty
            knowledge_mask = (knowledge_value > knowledge_threshold).float()
        else:
            knowledge_mask = (knowledge_value > 0.6).float()
        
        # 选择性知识传递
        filtered_teacher = teacher_features * knowledge_mask
        adapted_teacher = self.feature_adapter(filtered_teacher)
        
        # 知识融合
        fused_features = torch.cat([student_features, adapted_teacher], dim=1)
        enhanced_student = self.knowledge_fusion(fused_features)
        
        # 计算蒸馏损失
        distill_loss = F.mse_loss(enhanced_student, adapted_teacher)
        
        return {
            'distill_loss': distill_loss,
            'enhanced_features': enhanced_student,
            'relevance_scores': relevance_scores.mean(),
            'quality_scores': quality_scores.mean(),
            'knowledge_utilization': knowledge_mask.mean(),
            'knowledge_value': knowledge_value.mean()
        }

# 🛡️ 对抗式知识验证器
class AdversarialKnowledgeValidator(nn.Module):
    """对抗式知识质量验证模块"""
    
    def __init__(self, feature_dim, hidden_dim=64):
        super().__init__()
        
        # 知识质量判别器
        self.quality_discriminator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 任务性能预测器
        self.performance_predictor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, student_features, task_loss):
        """对抗式知识质量验证"""
        
        batch_size = student_features.size(0)
        
        # 评估学生特征质量
        feature_quality = self.quality_discriminator(student_features)
        
        # 预测任务性能（损失越低，性能越好）
        predicted_performance = self.performance_predictor(student_features)
        
        # 将标量任务损失扩展为批次维度
        if task_loss.dim() == 0:  # 标量
            actual_performance = torch.full((batch_size,), 1.0 / (1.0 + task_loss.item()), device=student_features.device)
        else:
            actual_performance = 1.0 / (1.0 + task_loss)
        
        # 性能预测损失
        performance_loss = F.mse_loss(predicted_performance.squeeze(), actual_performance)
        
        # 对抗损失：特征质量应该与实际任务性能一致
        target_quality = (actual_performance > 0.5).float()
        adversarial_loss = F.binary_cross_entropy(feature_quality.squeeze(), target_quality)
        
        return {
            'performance_loss': performance_loss,
            'adversarial_loss': adversarial_loss,
            'feature_quality': feature_quality.mean(),
            'predicted_performance': predicted_performance.mean()
        }

# 🎓 智能教师模型
class IntelligentTeacher(nn.Module):
    """增强版教师模型"""
    
    def __init__(self, input_dim=128, teacher_dim=256, num_classes=10):
        super().__init__()
        
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, teacher_dim),
            nn.BatchNorm1d(teacher_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(teacher_dim, teacher_dim),
            nn.BatchNorm1d(teacher_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(teacher_dim, teacher_dim // 2),
            nn.BatchNorm1d(teacher_dim // 2),
            nn.ReLU()
        )
        
        self.feature_extractor = nn.Linear(teacher_dim // 2, teacher_dim)
        self.classifier = nn.Linear(teacher_dim // 2, num_classes)
    
    def forward(self, x, return_features=False):
        features = self.backbone(x)
        extracted_features = self.feature_extractor(features)
        output = self.classifier(features)
        
        if return_features:
            return output, extracted_features
        return output

# 🎯 任务导向学生模型
class TaskOrientedStudent(nn.Module):
    """任务导向的学生模型"""
    
    def __init__(self, input_dim=128, student_dim=128, num_classes=10):
        super().__init__()
        
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, student_dim),
            nn.BatchNorm1d(student_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(student_dim, student_dim // 2),
            nn.BatchNorm1d(student_dim // 2),
            nn.ReLU()
        )
        
        self.feature_extractor = nn.Linear(student_dim // 2, student_dim)
        self.classifier = nn.Linear(student_dim // 2, num_classes)
        
        # 任务适应层
        self.task_adapter = nn.Sequential(
            nn.Linear(student_dim, student_dim),
            nn.ReLU(),
            nn.Linear(student_dim, student_dim)
        )
    
    def forward(self, x, return_features=False):
        features = self.backbone(x)
        extracted_features = self.feature_extractor(features)
        
        # 任务适应
        adapted_features = self.task_adapter(extracted_features)
        
        output = self.classifier(features)
        
        if return_features:
            return output, adapted_features
        return output

# 🚀 任务导向蒸馏训练器
class TaskOrientedDistillationTrainer:
    """任务导向知识蒸馏训练器"""
    
    def __init__(self, config=None):
        self.config = config or TASK_ORIENTED_CONFIG
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型
        self.teacher = IntelligentTeacher(
            teacher_dim=self.config['model']['teacher_dim'],
            num_classes=self.config['model']['num_classes']
        ).to(self.device)
        
        self.student = TaskOrientedStudent(
            student_dim=self.config['model']['student_dim'],
            num_classes=self.config['model']['num_classes']
        ).to(self.device)
        
        # 知识过滤器
        self.knowledge_filter = TaskOrientedKnowledgeFilter(
            teacher_dim=self.config['model']['teacher_dim'],
            student_dim=self.config['model']['student_dim'],
            hidden_dim=self.config['model']['hidden_dim']
        ).to(self.device)
        
        # 对抗式验证器
        self.adversarial_validator = AdversarialKnowledgeValidator(
            feature_dim=self.config['model']['student_dim'],
            hidden_dim=self.config['model']['hidden_dim']
        ).to(self.device)
        
        # 训练历史
        self.training_history = defaultdict(list)
        
        # 预训练教师模型
        self._pretrain_teacher()
    
    def _pretrain_teacher(self):
        """预训练教师模型"""
        print("🎓 预训练教师模型...")
        
        optimizer = torch.optim.Adam(self.teacher.parameters(), lr=0.001)
        
        for epoch in range(15):
            total_loss = 0
            for batch in range(20):
                x = torch.randn(32, 128).to(self.device)
                y = torch.randint(0, 10, (32,)).to(self.device)
                
                optimizer.zero_grad()
                output = self.teacher(x)
                loss = F.cross_entropy(output, y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 5 == 0:
                print(f"    教师预训练 Epoch {epoch+1}/15, Loss: {total_loss/20:.4f}")
        
        self.teacher.eval()
        print("✅ 教师模型预训练完成")
    
    def _create_synthetic_data(self, batch_size=32):
        """创建合成训练数据"""
        x = torch.randn(batch_size, 128).to(self.device)
        y = torch.randint(0, 10, (batch_size,)).to(self.device)
        return x, y
    
    def _get_dynamic_weights(self, epoch, total_epochs):
        """动态调整损失权重"""
        progress = epoch / total_epochs
        
        # 任务权重：从高到低
        task_weight = (
            self.config['distillation']['initial_task_weight'] * (1 - progress) +
            self.config['distillation']['final_task_weight'] * progress
        )
        
        # 蒸馏权重：从低到高
        distill_weight = (
            self.config['distillation']['initial_distill_weight'] * (1 - progress) +
            self.config['distillation']['final_distill_weight'] * progress
        )
        
        adversarial_weight = self.config['distillation']['adversarial_weight']
        
        return task_weight, distill_weight, adversarial_weight
    
    def train_epoch(self, epoch, total_epochs):
        """训练单个epoch"""
        
        # 获取动态权重
        task_weight, distill_weight, adversarial_weight = self._get_dynamic_weights(epoch, total_epochs)
        
        # 设置优化器
        all_params = (
            list(self.student.parameters()) +
            list(self.knowledge_filter.parameters()) +
            list(self.adversarial_validator.parameters())
        )
        optimizer = torch.optim.Adam(all_params, lr=self.config['training']['learning_rate'])
        
        epoch_metrics = {
            'total_loss': 0, 'task_loss': 0, 'distill_loss': 0, 'adversarial_loss': 0,
            'relevance_scores': 0, 'quality_scores': 0, 'knowledge_utilization': 0,
            'feature_quality': 0, 'accuracy': 0
        }
        
        num_batches = self.config['training']['num_batches']
        
        for batch in range(num_batches):
            # 生成训练数据
            x, y = self._create_synthetic_data(self.config['training']['batch_size'])
            
            optimizer.zero_grad()
            
            # 学生模型前向传播
            student_output, student_features = self.student(x, return_features=True)
            
            # 教师模型前向传播（无梯度）
            with torch.no_grad():
                teacher_output, teacher_features = self.teacher(x, return_features=True)
            
            # 计算任务损失
            task_loss = F.cross_entropy(student_output, y)
            
            # 任务导向知识过滤
            filter_result = self.knowledge_filter(
                teacher_features, student_features, task_loss
            )
            distill_loss = filter_result['distill_loss']
            
            # 对抗式知识验证
            validator_result = self.adversarial_validator(
                filter_result['enhanced_features'], task_loss
            )
            adversarial_loss = validator_result['adversarial_loss']
            
            # 总损失
            total_loss = (
                task_weight * task_loss +
                distill_weight * distill_loss +
                adversarial_weight * adversarial_loss
            )
            
            # 反向传播
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()
            
            # 计算准确率
            with torch.no_grad():
                pred = student_output.argmax(dim=1)
                accuracy = (pred == y).float().mean()
            
            # 累积指标
            epoch_metrics['total_loss'] += float(total_loss.item())
            epoch_metrics['task_loss'] += float(task_loss.item())
            epoch_metrics['distill_loss'] += float(distill_loss.item())
            epoch_metrics['adversarial_loss'] += float(adversarial_loss.item())
            epoch_metrics['relevance_scores'] += float(filter_result['relevance_scores'].item())
            epoch_metrics['quality_scores'] += float(filter_result['quality_scores'].item())
            epoch_metrics['knowledge_utilization'] += float(filter_result['knowledge_utilization'].item())
            epoch_metrics['feature_quality'] += float(validator_result['feature_quality'].item())
            epoch_metrics['accuracy'] += float(accuracy.item())
        
        # 平均化指标
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        # 添加权重信息
        epoch_metrics['task_weight'] = task_weight
        epoch_metrics['distill_weight'] = distill_weight
        epoch_metrics['adversarial_weight'] = adversarial_weight
        
        return epoch_metrics
    
    def run_experiment(self):
        """运行完整实验"""
        print("🎯 开始任务导向知识蒸馏实验")
        print(f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔧 配置: {json.dumps(self.config, indent=2, ensure_ascii=False)}")
        
        start_time = time.time()
        total_epochs = self.config['training']['total_epochs']
        warmup_epochs = self.config['training']['warmup_epochs']
        
        best_accuracy = 0
        best_task_loss = float('inf')
        
        for epoch in range(total_epochs):
            # 训练一个epoch
            metrics = self.train_epoch(epoch, total_epochs)
            
            # 记录历史
            for key, value in metrics.items():
                self.training_history[key].append(value)
            
            # 更新最佳指标
            if metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
            if metrics['task_loss'] < best_task_loss:
                best_task_loss = metrics['task_loss']
            
            # 打印进度
            if (epoch + 1) % 5 == 0 or epoch < warmup_epochs:
                stage = "预热阶段" if epoch < warmup_epochs else "蒸馏阶段"
                print(f"\n📈 Epoch {epoch+1:2d}/{total_epochs} [{stage}]:")
                print(f"    总损失: {metrics['total_loss']:.4f}")
                print(f"    任务损失: {metrics['task_loss']:.4f} | 准确率: {metrics['accuracy']:.3f}")
                print(f"    蒸馏损失: {metrics['distill_loss']:.4f} | 对抗损失: {metrics['adversarial_loss']:.4f}")
                print(f"    知识相关性: {metrics['relevance_scores']:.3f} | 知识质量: {metrics['quality_scores']:.3f}")
                print(f"    知识利用率: {metrics['knowledge_utilization']:.3f} | 特征质量: {metrics['feature_quality']:.3f}")
                print(f"    权重 - 任务: {metrics['task_weight']:.2f}, 蒸馏: {metrics['distill_weight']:.2f}, 对抗: {metrics['adversarial_weight']:.2f}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n🎉 任务导向蒸馏实验完成! 总用时: {total_time:.2f}秒")
        print(f"📊 最佳准确率: {best_accuracy:.3f}")
        print(f"📊 最佳任务损失: {best_task_loss:.4f}")
        
        # 分析结果
        self.analyze_results()
        
        return self.training_history
    
    def analyze_results(self):
        """分析实验结果"""
        print("\n📊 任务导向蒸馏实验结果分析:")
        print("=" * 60)
        
        # 获取关键指标
        initial_task_loss = self.training_history['task_loss'][0]
        final_task_loss = self.training_history['task_loss'][-1]
        initial_accuracy = self.training_history['accuracy'][0]
        final_accuracy = self.training_history['accuracy'][-1]
        
        task_improvement = (initial_task_loss - final_task_loss) / initial_task_loss * 100
        accuracy_improvement = (final_accuracy - initial_accuracy) / initial_accuracy * 100
        
        print(f"\n🎯 任务性能分析:")
        print(f"    初始任务损失: {initial_task_loss:.4f} → 最终: {final_task_loss:.4f}")
        print(f"    任务损失改善: {task_improvement:.1f}%")
        print(f"    初始准确率: {initial_accuracy:.3f} → 最终: {final_accuracy:.3f}")
        print(f"    准确率提升: {accuracy_improvement:.1f}%")
        
        # 知识传递效果分析
        avg_relevance = np.mean(self.training_history['relevance_scores'][-10:])
        avg_quality = np.mean(self.training_history['quality_scores'][-10:])
        avg_utilization = np.mean(self.training_history['knowledge_utilization'][-10:])
        
        print(f"\n🧠 知识传递分析:")
        print(f"    平均知识相关性: {avg_relevance:.3f}")
        print(f"    平均知识质量: {avg_quality:.3f}")
        print(f"    平均知识利用率: {avg_utilization:.3f}")
        
        # 成功评估
        if task_improvement > 15 and accuracy_improvement > 10:
            print("\n✅ 实验非常成功! 任务导向策略显著改善了性能")
            print("    建议: 在真实数据集上验证并部署")
        elif task_improvement > 8 and accuracy_improvement > 5:
            print("\n⚠️  实验部分成功，有明显改善")
            print("    建议: 进一步优化知识过滤策略")
        elif task_improvement > 3:
            print("\n🔄 实验有改善但仍有优化空间")
            print("    建议: 调整对抗训练权重和知识过滤阈值")
        else:
            print("\n❌ 实验效果有限，需要重新评估策略")
            print("    建议: 检查数据质量和模型架构匹配度")
        
        # 保存结果
        self.save_results()
    
    def save_results(self):
        """保存实验结果"""
        results_dir = "experiments/task_oriented_distillation"
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"{results_dir}/task_oriented_results_{timestamp}.json"
        
        # 准备保存数据
        save_data = {
            'timestamp': timestamp,
            'config': self.config,
            'training_history': dict(self.training_history),
            'final_metrics': {
                'task_loss': self.training_history['task_loss'][-1],
                'accuracy': self.training_history['accuracy'][-1],
                'distill_loss': self.training_history['distill_loss'][-1],
                'knowledge_utilization': self.training_history['knowledge_utilization'][-1]
            }
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 实验结果已保存至: {results_file}")

# 🎯 主函数
def main():
    """主函数 - 运行任务导向知识蒸馏实验"""
    print("🎯 MapSage V5 - 任务导向知识蒸馏实验")
    print("=" * 50)
    
    # 创建训练器
    trainer = TaskOrientedDistillationTrainer()
    
    # 运行实验
    results = trainer.run_experiment()
    
    print("\n🎯 任务导向知识蒸馏实验完成!")
    return results

if __name__ == "__main__":
    results = main()