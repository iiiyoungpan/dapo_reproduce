import matplotlib.pyplot as plt
import numpy as np

# 读取准确率数据
accuracy_data = []
with open('docs/accuracy_losses.txt', 'r') as f:
    for line in f:
        if line.strip():
            step, accuracy = line.strip().split(',')
            accuracy_data.append((int(step), float(accuracy)))

# 读取训练损失数据
loss_data = []
with open('docs/training_losses.txt', 'r') as f:
    for line in f:
        if line.strip():
            step, loss = line.strip().split(',')
            loss_data.append((int(step), float(loss)))

# 创建图形
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))

# 绘制准确率图
steps_acc, accuracies = zip(*accuracy_data)
ax1.plot(steps_acc, accuracies, 'b-', linewidth=2, marker='o', markersize=4)
ax1.set_xlabel('Training Steps')
ax1.set_ylabel('Accuracy')
ax1.set_title('Training Steps vs Accuracy')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, max(accuracies) * 1.1)

# 绘制损失函数图
steps_loss, losses = zip(*loss_data)
ax2.plot(steps_loss, losses, 'r-', linewidth=2, marker='s', markersize=4)
ax2.set_xlabel('Training Steps')
ax2.set_ylabel('Loss')
ax2.set_title('Training Steps vs Loss Function')
ax2.grid(True, alpha=0.3)

# 自动调整损失图的y轴范围
loss_min, loss_max = min(losses), max(losses)
loss_range = loss_max - loss_min
ax2.set_ylim(loss_min - 0.1 * loss_range, loss_max + 0.1 * loss_range)

plt.tight_layout()
plt.savefig('./docs/training_plots.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Accuracy data points: {len(accuracy_data)}")
print(f"Loss data points: {len(loss_data)}")
print("Charts saved as training_plots_en.png")