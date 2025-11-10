import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('extracted_epochs_clean.csv')

epochs = df['epoch'].values
alex = df['alexnet_acc'].values
dense = df['densenet_acc'].values
fenet = df['fenet_acc'].values
ours = df['Ours'].values

plt.figure(figsize=(6.4,4.2))

# 标准折线图：硬折线、圆标记便于看到拐点
plt.plot(epochs, alex, '-o', linewidth=2, markersize=4, label='AlexNet')
plt.plot(epochs, dense, '--o', linewidth=2, markersize=4, label='DenseNet')
plt.plot(epochs, fenet, '-.o', linewidth=2, markersize=4, label='FENet-1D')
plt.plot(epochs, ours, '-.o', linewidth=2, markersize=4, label='Ours')

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Test Accuracy (%)', fontsize=12)
plt.title('AIS Dataset — 50 classes', fontsize=14)
plt.xlim(0,35)
plt.ylim(0,100)
plt.grid(which='both', linestyle=':', linewidth=0.6)
plt.legend(loc='lower right', fontsize=11)
plt.tight_layout()

plt.savefig('hardline_convergence_plot.pdf', dpi=300)
plt.show()
