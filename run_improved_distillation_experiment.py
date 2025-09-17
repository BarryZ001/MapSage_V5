#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 MapSage V5 - 改进版知识蒸馏实验执行脚本

基于前期实验分析，设计更精细、成功概率更高的知识蒸馏实验

核心改进策略:
1. 渐进式训练 (任务预训练 → 知识蒸馏)
2. 自适应损失权重平衡
3. 多层次混合蒸馏策略
4. 轻量化师生架构匹配
5. 动态温度调度

Author: MapSage Team
Date: 2024
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.append('/Users/barryzhang/myDev3/MapSage_V5')

# 导入改进的蒸馏配置
from configs.train_distill_dinov3_v2_improved import (
    ImprovedKnowledgeDistillationModel,
    improved_distillation_training,
    compare_with_previous_experiment
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# 🔧 实验配置类
class ExperimentConfig:
    """改进版实验配置管理"""
    
    def __init__(self):
        # 基础配置
        self.project_name = "MapSage_V5_Improved_Distillation"
        self.experiment_name = f"improved_distill_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 训练配置
        self.epochs = 25
        self.batch_size = 4
        self.num_classes = 7
        self.input_size = (512, 512)
        
        # 优化器配置
        self.learning_rate = 0.0001
        self.weight_decay = 0.01
        self.momentum = 0.9
        self.betas = (0.9, 0.999)
        
        # 蒸馏配置 (关键改进)
        self.distill_config = {
            'task_weight': 0.6,        # 提高任务权重
            'distill_weight': 0.3,     # 降低蒸馏权重
            'feature_weight': 0.1,     # 降低特征权重
            'attention_weight': 0.05,  # 注意力权重
            'initial_temperature': 6.0,
            'final_temperature': 3.0,
            'warmup_epochs': 5         # 渐进式训练
        }
        
        # 调度器配置
        self.scheduler_config = {
            'type': 'cosine',
            'T_max': self.epochs,
            'eta_min': self.learning_rate * 0.01
        }
        
        # 保存配置
        self.save_interval = 5
        self.log_interval = 10
        self.eval_interval = 5
        
        # 路径配置
        self.output_dir = Path(f'/Users/barryzhang/myDev3/MapSage_V5/experiments/{self.experiment_name}')
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'
        self.result_dir = self.output_dir / 'results'
        
        # 创建目录
        for dir_path in [self.output_dir, self.checkpoint_dir, self.log_dir, self.result_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def save_config(self):
        """保存实验配置"""
        config_dict = {
            'project_name': self.project_name,
            'experiment_name': self.experiment_name,
            'device': self.device,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'distill_config': self.distill_config,
            'scheduler_config': self.scheduler_config
        }
        
        config_path = self.output_dir / 'experiment_config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        print(f"📋 实验配置已保存: {config_path}")

# 🎯 虚拟数据集类 (用于实验演示)
class DummySegmentationDataset(Dataset):
    """虚拟分割数据集 - 用于实验演示"""
    
    def __init__(self, num_samples=1000, image_size=(512, 512), num_classes=7):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 生成虚拟图像 (模拟遥感图像特征)
        image = torch.randn(3, *self.image_size)
        
        # 生成虚拟分割标签 (模拟地物分类)
        mask = torch.randint(0, self.num_classes, self.image_size)
        
        return image, mask

# 📊 实验监控类
class ExperimentMonitor:
    """实验监控和日志记录"""
    
    def __init__(self, config):
        self.config = config
        self.writer = SummaryWriter(log_dir=str(config.log_dir))
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(config.log_dir / 'experiment.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 实验记录
        self.epoch_results = []
        self.best_metrics = {
            'best_loss': float('inf'),
            'best_epoch': 0,
            'best_task_loss': float('inf')
        }
    
    def log_epoch(self, epoch, metrics, lr, temperature):
        """记录每个epoch的结果"""
        # 控制台输出
        self.logger.info(f"\n📈 Epoch {epoch+1}/{self.config.epochs}:")
        self.logger.info(f"    总损失: {metrics['total_loss']:.4f}")
        self.logger.info(f"    任务损失: {metrics['task_loss']:.4f}")
        self.logger.info(f"    蒸馏损失: {metrics['distill_loss']:.4f}")
        self.logger.info(f"    特征损失: {metrics['feature_loss']:.4f}")
        self.logger.info(f"    注意力损失: {metrics['attention_loss']:.4f}")
        self.logger.info(f"    学习率: {lr:.6f}")
        self.logger.info(f"    温度: {temperature:.2f}")
        
        # TensorBoard记录
        self.writer.add_scalar('Loss/Total', metrics['total_loss'], epoch)
        self.writer.add_scalar('Loss/Task', metrics['task_loss'], epoch)
        self.writer.add_scalar('Loss/Distillation', metrics['distill_loss'], epoch)
        self.writer.add_scalar('Loss/Feature', metrics['feature_loss'], epoch)
        self.writer.add_scalar('Loss/Attention', metrics['attention_loss'], epoch)
        self.writer.add_scalar('Training/LearningRate', lr, epoch)
        self.writer.add_scalar('Training/Temperature', temperature, epoch)
        
        # 更新最佳指标
        if metrics['total_loss'] < self.best_metrics['best_loss']:
            self.best_metrics['best_loss'] = metrics['total_loss']
            self.best_metrics['best_epoch'] = epoch
            self.best_metrics['best_task_loss'] = metrics['task_loss']
            self.logger.info(f"    💾 保存最佳模型 (损失: {metrics['total_loss']:.4f})")
        
        # 保存epoch结果
        epoch_result = {
            'epoch': epoch,
            'metrics': metrics,
            'lr': lr,
            'temperature': temperature
        }
        self.epoch_results.append(epoch_result)
    
    def save_results(self):
        """保存实验结果"""
        results = {
            'best_metrics': self.best_metrics,
            'epoch_results': self.epoch_results,
            'config': self.config.__dict__
        }
        
        result_path = self.config.result_dir / 'experiment_results.json'
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"📊 实验结果已保存: {result_path}")
    
    def close(self):
        """关闭监控"""
        self.writer.close()

# 🚀 改进版训练器
class ImprovedDistillationTrainer:
    """改进版知识蒸馏训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        # 创建模型
        self.model = ImprovedKnowledgeDistillationModel(
            distill_cfg=config.distill_config
        ).to(self.device)
        
        # 创建数据集
        self.train_dataset = DummySegmentationDataset(
            num_samples=1000,
            image_size=config.input_size,
            num_classes=config.num_classes
        )
        
        self.val_dataset = DummySegmentationDataset(
            num_samples=200,
            image_size=config.input_size,
            num_classes=config.num_classes
        )
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # 创建优化器 (只优化学生模型)
        student_params = (
            list(self.model.student_model.parameters()) +
            list(self.model.multi_scale_adapters.parameters()) +
            list(self.model.attention_transfer.parameters())
        )
        
        self.optimizer = torch.optim.AdamW(
            student_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas
        )
        
        # 创建学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.scheduler_config['T_max'],
            eta_min=config.scheduler_config['eta_min']
        )
        
        # 创建监控器
        self.monitor = ExperimentMonitor(config)
        
        print(f"✅ 改进版训练器初始化完成")
        print(f"📊 训练样本: {len(self.train_dataset)}")
        print(f"📊 验证样本: {len(self.val_dataset)}")
        print(f"🎯 设备: {self.device}")
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        self.model.update_temperature(epoch, self.config.epochs)
        
        epoch_metrics = {
            'total_loss': 0,
            'task_loss': 0,
            'distill_loss': 0,
            'feature_loss': 0,
            'attention_loss': 0
        }
        
        num_batches = len(self.train_loader)
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # 前向传播
            outputs = self.model(images, targets, epoch)
            
            # 反向传播
            self.optimizer.zero_grad()
            outputs['total_loss'].backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.student_model.parameters(), 
                max_norm=1.0
            )
            
            self.optimizer.step()
            
            # 累积损失
            for key in epoch_metrics:
                if key in outputs:
                    epoch_metrics[key] += outputs[key].item()
            
            # 定期打印进度
            if batch_idx % self.config.log_interval == 0:
                progress = 100. * batch_idx / num_batches
                print(f"\r训练进度: {progress:.1f}% [{batch_idx}/{num_batches}]", end='')
        
        # 计算平均损失
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def validate_epoch(self, epoch):
        """验证一个epoch"""
        self.model.eval()
        
        val_metrics = {
            'total_loss': 0,
            'task_loss': 0,
            'distill_loss': 0,
            'feature_loss': 0,
            'attention_loss': 0
        }
        
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images, targets, epoch)
                
                # 累积损失
                for key in val_metrics:
                    if key in outputs:
                        val_metrics[key] += outputs[key].item()
        
        # 计算平均损失
        for key in val_metrics:
            val_metrics[key] /= num_batches
        
        return val_metrics
    
    def save_checkpoint(self, epoch, metrics):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config.__dict__
        }
        
        checkpoint_path = self.config.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if metrics['total_loss'] == self.monitor.best_metrics['best_loss']:
            best_path = self.config.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
    
    def train(self):
        """完整训练流程"""
        print(f"\n🚀 开始改进版知识蒸馏训练...")
        print(f"📊 实验配置: {self.config.experiment_name}")
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            
            # 训练
            train_metrics = self.train_epoch(epoch)
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            current_temp = self.model.current_temperature
            
            # 验证
            if (epoch + 1) % self.config.eval_interval == 0:
                val_metrics = self.validate_epoch(epoch)
                print(f"\n🔍 验证结果 - 总损失: {val_metrics['total_loss']:.4f}")
            
            # 记录结果
            self.monitor.log_epoch(epoch, train_metrics, current_lr, current_temp)
            
            # 保存检查点
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch, train_metrics)
            
            epoch_time = time.time() - epoch_start
            print(f"\n⏱️  Epoch用时: {epoch_time:.2f}s")
        
        total_time = time.time() - start_time
        print(f"\n🎉 训练完成! 总用时: {total_time/3600:.2f}小时")
        
        # 保存最终结果
        self.monitor.save_results()
        self.monitor.close()
        
        return self.monitor.best_metrics

# 📈 实验分析器
class ExperimentAnalyzer:
    """实验结果分析器"""
    
    def __init__(self, config):
        self.config = config
    
    def analyze_training_progress(self, results):
        """分析训练进度"""
        print("\n📈 训练进度分析:")
        
        epoch_results = results['epoch_results']
        
        # 损失趋势分析
        total_losses = [r['metrics']['total_loss'] for r in epoch_results]
        task_losses = [r['metrics']['task_loss'] for r in epoch_results]
        distill_losses = [r['metrics']['distill_loss'] for r in epoch_results]
        
        print(f"    📊 总损失变化: {total_losses[0]:.4f} → {total_losses[-1]:.4f}")
        print(f"    📊 任务损失变化: {task_losses[0]:.4f} → {task_losses[-1]:.4f}")
        print(f"    📊 蒸馏损失变化: {distill_losses[0]:.4f} → {distill_losses[-1]:.4f}")
        
        # 收敛性分析
        if len(total_losses) >= 10:
            recent_losses = total_losses[-10:]
            loss_std = np.std(recent_losses)
            print(f"    📊 最近10轮损失标准差: {loss_std:.4f} (越小越稳定)")
        
        # 任务性能分析
        task_improvement = (task_losses[0] - task_losses[-1]) / task_losses[0] * 100
        print(f"    📊 任务损失改善: {task_improvement:.2f}%")
        
        if task_improvement > 5:
            print("    ✅ 任务性能显著提升!")
        elif task_improvement > 1:
            print("    ⚠️  任务性能轻微提升")
        else:
            print("    ❌ 任务性能未明显改善")
    
    def compare_with_baseline(self, results):
        """与基线实验对比"""
        print("\n🔍 与前期实验对比:")
        
        # 前期实验结果 (参考值)
        baseline_results = {
            'final_total_loss': 53.1705,
            'final_task_loss': 1.9589,
            'final_distill_loss': 75.0790,
            'task_improvement': 0.05  # 几乎无改善
        }
        
        # 当前实验结果
        current_results = results['epoch_results'][-1]['metrics']
        
        print(f"    📊 总损失对比:")
        print(f"        前期: {baseline_results['final_total_loss']:.4f}")
        print(f"        当前: {current_results['total_loss']:.4f}")
        
        print(f"    📊 任务损失对比:")
        print(f"        前期: {baseline_results['final_task_loss']:.4f}")
        print(f"        当前: {current_results['task_loss']:.4f}")
        
        # 改善评估
        total_improvement = (baseline_results['final_total_loss'] - current_results['total_loss']) / baseline_results['final_total_loss'] * 100
        task_improvement = (baseline_results['final_task_loss'] - current_results['task_loss']) / baseline_results['final_task_loss'] * 100
        
        print(f"    📈 总损失改善: {total_improvement:.2f}%")
        print(f"    📈 任务损失改善: {task_improvement:.2f}%")
        
        if task_improvement > 10:
            print("    🎉 实验改进非常成功!")
        elif task_improvement > 5:
            print("    ✅ 实验改进成功!")
        elif task_improvement > 0:
            print("    ⚠️  实验有轻微改进")
        else:
            print("    ❌ 实验改进效果不明显")
    
    def generate_recommendations(self, results):
        """生成改进建议"""
        print("\n💡 后续改进建议:")
        
        epoch_results = results['epoch_results']
        final_metrics = epoch_results[-1]['metrics']
        
        # 基于结果的建议
        if final_metrics['task_loss'] > 1.5:
            print("    🎯 任务损失仍较高，建议:")
            print("        - 进一步提高任务权重 (0.6 → 0.7)")
            print("        - 延长预训练阶段 (5 → 8 epochs)")
            print("        - 调整学习率策略")
        
        if final_metrics['distill_loss'] > 50:
            print("    🎯 蒸馏损失较高，建议:")
            print("        - 调整温度参数范围")
            print("        - 优化特征对齐策略")
            print("        - 考虑更匹配的师生架构")
        
        print("    🚀 通用改进方向:")
        print("        - 引入真实数据集验证")
        print("        - 添加更多评估指标 (mIoU, Accuracy)")
        print("        - 实施早停策略")
        print("        - 尝试不同的蒸馏策略组合")

# 🎯 主实验函数
def run_improved_distillation_experiment():
    """运行改进版知识蒸馏实验"""
    print("🎯 MapSage V5 - 改进版知识蒸馏实验")
    print("=" * 80)
    
    # 创建实验配置
    config = ExperimentConfig()
    config.save_config()
    
    # 实验前分析
    print("\n📋 实验前分析:")
    compare_with_previous_experiment()
    
    # 创建训练器
    trainer = ImprovedDistillationTrainer(config)
    
    # 开始训练
    best_metrics = trainer.train()
    
    # 加载结果进行分析
    result_path = config.result_dir / 'experiment_results.json'
    with open(result_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 结果分析
    analyzer = ExperimentAnalyzer(config)
    analyzer.analyze_training_progress(results)
    analyzer.compare_with_baseline(results)
    analyzer.generate_recommendations(results)
    
    print(f"\n🎉 改进版知识蒸馏实验完成!")
    print(f"📊 最佳损失: {best_metrics['best_loss']:.4f}")
    print(f"📁 实验结果保存在: {config.output_dir}")
    
    return results, config

if __name__ == "__main__":
    # 运行实验
    results, config = run_improved_distillation_experiment()