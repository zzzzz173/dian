import os
import matplotlib.font_manager as fm

# 在代码最开始添加环境变量设置
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import json
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
from torch.nn.utils import clip_grad_norm_
from collections import Counter

# 定义数据集类
class TextRatingDataset(Dataset):
    def __init__(self, texts, ratings, tokenizer, max_length=512):
        self.texts = texts
        self.ratings = ratings
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        rating = self.ratings[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'rating': torch.tensor(rating, dtype=torch.float)
        }


class RatingPredictor:
    def __init__(self, model_name='bert-base-chinese', device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            problem_type="regression"
        ).to(self.device)

        # 训练参数
        self.batch_size = 8
        self.learning_rate = 1e-5
        self.num_epochs = 20
        self.max_grad_norm = 1.0

        self.train_mean = None
        self.train_std = None

    def preprocess_text(self, text):

        # 基础清理
        text = text.strip()
        text = ' '.join(text.split())

        # 保留更多有意义的标点
        valid_chars = set('。，！？：；（）【】《》""、')
        text = ''.join(char for char in text if
                       '\u4e00' <= char <= '\u9fff' or  # 中文
                       char.isalnum() or  # 数字和字母
                       char in valid_chars)  # 标点

        # 移除重复字符
        text = ''.join(char for i, char in enumerate(text)
                       if i == 0 or char != text[i - 1] or char in '。！？')

        return text

    def balance_dataset(self, texts, ratings, bins=10):
        #平衡数据集
        # 将评分分成bins个桶
        ratings_np = np.array(ratings)
        hist, bin_edges = np.histogram(ratings_np, bins=bins)
        max_samples = int(np.median(hist) * 1.5)  # 允许每个桶最多样本数

        balanced_texts = []
        balanced_ratings = []

        for i in range(len(bin_edges) - 1):
            mask = (ratings_np >= bin_edges[i]) & (ratings_np < bin_edges[i + 1])
            bin_texts = np.array(texts)[mask]
            bin_ratings = ratings_np[mask]

            if len(bin_ratings) > max_samples:
                # 随机下采样
                indices = np.random.choice(len(bin_ratings), max_samples, replace=False)
                bin_texts = bin_texts[indices]
                bin_ratings = bin_ratings[indices]
            elif len(bin_ratings) < max_samples / 2:
                # 上采样
                if len(bin_ratings) > 0:
                    n_samples = min(max_samples // 2 - len(bin_ratings), len(bin_ratings))
                    indices = np.random.choice(len(bin_ratings), n_samples, replace=True)
                    bin_texts = np.append(bin_texts, bin_texts[indices])
                    bin_ratings = np.append(bin_ratings, bin_ratings[indices])

            balanced_texts.extend(bin_texts)
            balanced_ratings.extend(bin_ratings)

        return balanced_texts, balanced_ratings

    def prepare_data(self, train_file, val_file):
        #数据准备

        def load_and_process(file_path):
            texts, ratings = [], []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    text = self.preprocess_text(data['text'])

                    if len(text) < 20:  # 增加最小长度要求
                        continue

                    rating = float(data['point'] if 'point' in data else data['label'])
                    rating = max(0, min(10, rating))

                    texts.append(text)
                    ratings.append(rating / 10.0)
            return texts, ratings

        # 加载数据
        train_texts, train_ratings = load_and_process(train_file)
        val_texts, val_ratings = load_and_process(val_file)

        # 平衡训练集
        train_texts, train_ratings = self.balance_dataset(train_texts, train_ratings)

        # 计算并应用标准化
        self.train_mean = np.mean(train_ratings)
        self.train_std = np.std(train_ratings)

        train_ratings = [(r - self.train_mean) / self.train_std for r in train_ratings]
        val_ratings = [(r - self.train_mean) / self.train_std for r in val_ratings]

        # 创建数据加载器
        train_dataset = TextRatingDataset(train_texts, train_ratings, self.tokenizer)
        val_dataset = TextRatingDataset(val_texts, val_ratings, self.tokenizer)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=2
        )

        return train_loader, val_loader, train_texts, val_texts, train_ratings, val_ratings

    def custom_loss(self, outputs, labels):
        #损失函数
        mse_loss = torch.nn.MSELoss()(outputs.squeeze(), labels)
        l1_loss = torch.nn.L1Loss()(outputs.squeeze(), labels)
        huber_loss = torch.nn.SmoothL1Loss()(outputs.squeeze(), labels)

        # 组合损失
        return 0.4 * mse_loss + 0.3 * l1_loss + 0.3 * huber_loss

    def train(self, train_loader, val_loader, train_ratings, val_ratings):
        #训练过程
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        total_steps = len(train_loader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )

        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0

        train_losses = []
        val_losses = []
        metrics_history = []

        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            # 训练阶段
            self.model.train()
            total_train_loss = 0

            for batch in train_loader:
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                ratings = batch['rating'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                loss = self.custom_loss(outputs.logits, ratings)
                total_train_loss += loss.item()

                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                optimizer.step()
                scheduler.step()

            avg_train_loss = total_train_loss / len(train_loader)

            # 验证阶段
            val_loss, metrics = self.evaluate(val_loader)

            # 打印详细的评估结果
            print(f'训练损失: {avg_train_loss:.4f}')
            print(f'验证损失: {val_loss:.4f}')
            print(f'R² 分数: {metrics["r2"]:.4f}')
            print(f'MAE: {metrics["mae"]:.4f}')
            print(f'MSE: {metrics["mse"]:.4f}')

            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model('./best_model')
                print("保存最佳模型")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("\n早停：验证损失没有改善")
                    break

            # 收集指标
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            metrics_history.append(metrics)

        # 绘制训练过程的可视化图表
        self.plot_training_metrics(train_losses, val_losses, metrics_history)
        self.plot_rating_distribution(train_ratings, val_ratings)

        # 获取最终预测结果
        final_predictions = self.get_predictions(val_loader)
        self.plot_prediction_scatter(val_ratings, final_predictions)

    def evaluate(self, val_loader):
        #评估过程
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_ratings = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                ratings = batch['rating'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                loss = self.custom_loss(outputs.logits, ratings)
                total_loss += loss.item()

                # 反标准化预测结果
                predictions = outputs.logits.cpu().numpy()
                predictions = predictions * self.train_std + self.train_mean
                ratings = ratings.cpu().numpy() * self.train_std + self.train_mean

                all_predictions.extend(predictions)
                all_ratings.extend(ratings)

        avg_loss = total_loss / len(val_loader)

        metrics = {
            'r2': r2_score(all_ratings, all_predictions),
            'mae': mean_absolute_error(all_ratings, all_predictions),
            'mse': mean_squared_error(all_ratings, all_predictions)
        }

        return avg_loss, metrics

    def save_model(self, path):
        """保存模型和tokenizer"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def plot_training_metrics(self, train_losses, val_losses, metrics_history):
        #绘制训练过程中的各项指标
        plt.style.use('default')

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Process Monitor', fontsize=16)

        # 损失曲线
        axes[0, 0].plot(train_losses, label='Train Loss', marker='o')
        axes[0, 0].plot(val_losses, label='Val Loss', marker='o')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # R²分数
        axes[0, 1].plot([m['r2'] for m in metrics_history], marker='o', color='green')
        axes[0, 1].set_title('R² Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('R²')
        axes[0, 1].grid(True, alpha=0.3)

        # MAE
        axes[1, 0].plot([m['mae'] for m in metrics_history], marker='o', color='orange')
        axes[1, 0].set_title('Mean Absolute Error (MAE)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].grid(True, alpha=0.3)

        # MSE
        axes[1, 1].plot([m['mse'] for m in metrics_history], marker='o', color='red')
        axes[1, 1].set_title('Mean Squared Error (MSE)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MSE')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_rating_distribution(self, train_ratings, val_ratings):
        """绘制评分分布"""
        plt.figure(figsize=(12, 6))

        train_ratings = np.array(train_ratings) * self.train_std + self.train_mean
        val_ratings = np.array(val_ratings) * self.train_std + self.train_mean

        plt.hist(train_ratings * 10, bins=20, alpha=0.6, label='Training Set', color='blue')
        plt.hist(val_ratings * 10, bins=20, alpha=0.6, label='Validation Set', color='orange')

        plt.title('Rating Distribution')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('rating_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_prediction_scatter(self, true_ratings, predictions):
        #绘制预测值与真实值的散点图
        plt.figure(figsize=(10, 6))

        true_ratings = np.array(true_ratings) * self.train_std + self.train_mean
        predictions = np.array(predictions) * self.train_std + self.train_mean

        plt.scatter(true_ratings * 10, predictions * 10, alpha=0.5)

        min_val = min(true_ratings.min(), predictions.min()) * 10
        max_val = max(true_ratings.max(), predictions.max()) * 10
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Prediction')

        plt.title('Predictions vs Ground Truth')
        plt.xlabel('True Rating')
        plt.ylabel('Predicted Rating')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('prediction_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()

    def get_predictions(self, val_loader):
        #获取验证集的预测结果
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                batch_predictions = outputs.logits.cpu().numpy()
                predictions.extend(batch_predictions)

        return predictions


def main():
    predictor = RatingPredictor()

    # 准备数据
    train_loader, val_loader, train_texts, val_texts, train_ratings, val_ratings = \
        predictor.prepare_data('comments_and_ratings.jsonl', 'test.jsonl')

    # 打印数据统计
    print("\n数据集统计：")
    print(f"训练集样本数: {len(train_texts)}")
    print(f"验证集样本数: {len(val_texts)}")
    print(f"训练集评分均值: {predictor.train_mean * 10:.2f}")
    print(f"训练集评分标准差: {predictor.train_std * 10:.2f}")

    # 训练模型（添加参数）
    predictor.train(train_loader, val_loader, train_ratings, val_ratings)
if __name__ == '__main__':
    main()