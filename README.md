# Cell-Clustering-with-PyTorch-MLP
# 简介
细胞聚类是生物信息学和医学研究中的关键问题之一。通过对细胞数据进行聚类，我们可以识别潜在的生物学群体，并更好地理解细胞之间的关系。本项目旨在通过构建一个基于PyTorch的MLP模型，实现对细胞数据的自动聚类。
数据
共计148301条数据，特征如下。

```python
Index(['cell_id', 'x', 'y', 'DAPI-01', 'CD20', 'CXCR5', 'IL21', 'PAX5',
       'PMTOR', 'LAG3', 'KI67', 'BCA1', 'FOXP3', 'STAT3', 'TBET', 'CD4',
       'IRF4', 'CD45RO', 'CD3E', 'VIMENTIN', 'CD38', 'PD1', 'PAN CYTO', 'CD21',
       'ICOS', 'MAC2/GAL3', 'CD8', 'PDL1', 'S100B', 'GZMB', 'TCF1', 'IRF7',
       'IFNG', 'CD19', 'STAT1', 'TOX', 'CD11C', 'CD45', 'HLA DR', 'CD39',
       'CXCR3', 'IL21R', 'Cluster', 'Cell Cluster'],
      dtype='object')
```


# 结果展示
![\[图片\]](https://img-blog.csdnimg.cn/direct/76c2114b9c434116b93a5eb47fc9fd2d.png)

![\[图片\]](https://img-blog.csdnimg.cn/direct/99cc9966f4584e96b2353582966e6204.png)

![\[图片\]](https://img-blog.csdnimg.cn/direct/680bfad3259d487bb15d3e55f16163da.png)

# 项目逻辑
1. 数据预处理：
  - 读取和加载细胞数据集
  - 删除不必要的列，提取特征和目标变量
  - 对目标变量进行编码，以便于神经网络处理
  - 划分数据集为训练集和测试集
2. MLP模型构建：
  - 使用PyTorch构建一个多层感知机（MLP）
  - 设计合适的网络结构，包括输入层、隐藏层和输出层
  - 初始化模型参数，设置损失函数和优化器
3. 模型训练与评估：
  - 训练MLP模型并记录损失值
  - 绘制训练损失曲线，以便于直观了解模型的学习过程
  - 通过在测试集上进行预测，评估模型的性能
  - 绘制混淆矩阵，详细了解模型的分类效果
4. 新数据预测：
  - 读取新的细胞数据集
  - 对新数据进行与训练数据相似的预处理步骤
  - 加载已保存的MLP模型参数
  - 使用模型进行新数据的聚类预测
  - 解码预测结果，可视化细胞分类信息
## 数据预处理

```python

# 读取并加载数据集
data = pd.read_csv('data.csv')

# 删除不需要的列
data = data.drop(['cell_id', 'x', 'y', 'Cell Cluster'], axis=1)

# 提取特征和目标变量
X = data.drop('Cluster', axis=1)
y = data['Cluster']

# 对目标变量进行编码
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# 将数据转换为PyTorch张量
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 打印数据预处理后的信息
print("数据预处理完成，部分数据信息如下：")
print("X_train_tensor shape:", X_train_tensor.shape)
print("y_train_tensor shape:", y_train_tensor.shape)
print("X_test_tensor shape:", X_test_tensor.shape)
print("y_test_tensor shape:", y_test_tensor.shape)

```

上述代码首先使用pandas库读取CSV文件，然后删除不需要的列。接着，提取特征（X）和目标变量（y）。对目标变量进行了编码，使用LabelEncoder将类别标签转换为数字。之后，使用train_test_split将数据集划分为训练集和测试集。最后，将数据转换为PyTorch张量，以便后续在PyTorch中进行处理。
## 构建多层感知机（MLP）模型
下面是构建多层感知机（MLP）模型的代码，包括定义模型结构、初始化参数、以及定义损失函数和优化器：

```python

# 定义MLP模型的结构
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, len(label_encoder.classes_))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 初始化模型
input_size = X_train.shape[1]
model = MLP(input_size=input_size)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 打印模型结构
print("MLP模型结构：")
print(model)

```

在这段代码中，首先定义了一个简单的MLP模型结构，包括三个全连接层（nn.Linear）。在forward方法中定义了每个层之间的流动方式，使用ReLU作为激活函数。
接着，通过MLP(input_size=input_size)初始化了一个MLP模型，其中input_size是输入特征的维度，即X_train的列数。
然后，定义了损失函数为交叉熵损失（nn.CrossEntropyLoss()），并选择了Adam优化器（optim.Adam）来更新模型参数，学习率设置为0.01。
最后，通过print(model)打印了MLP模型的结构，以便检查模型的配置是否正确。
## 训练模型
下面是训练模型、记录损失值和绘制训练损失曲线的代码：

```python
# 用于记录每个epoch的损失值
losses = []

# 训练模型
num_epochs = 100for epoch in range(num_epochs):# 梯度归零
    optimizer.zero_grad()
    # 前向传播
    outputs = model(X_train_tensor)
    # 计算损失
    loss = criterion(outputs, y_train_tensor)
    # 反向传播
    loss.backward()
    # 更新参数
    optimizer.step()
# 记录损失值
    losses.append(loss.item())
# 打印每个epoch的损失值print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 绘制损失曲线
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

```
在这段代码中，首先定义了一个空列表losses，用于存储每个epoch的损失值。然后，进入训练循环，循环num_epochs次。
在每个epoch内，模型的梯度被归零（optimizer.zero_grad()），然后进行前向传播，计算损失，反向传播，最后通过优化器更新模型参数。
每个epoch的损失值被记录到losses列表中。在每个epoch结束后，通过print语句打印当前epoch的损失值。
最后，使用matplotlib绘制了训练损失曲线，以便可视化损失值的变化趋势。
## 保存和加载模型
下面是保存和加载模型的代码：

```python
# 保存模型参数
torch.save(model.state_dict(), 'mlp_model.pth')

# 加载模型
loaded_model = MLP(input_size=X_train.shape[1])
loaded_model.load_state_dict(torch.load('mlp_model.pth'))
loaded_model.eval()

# 可选：打印加载的模型结构print("Loaded Model Structure:")
print(loaded_model)
```
在这段代码中，首先使用torch.save保存了训练好的模型参数。model.state_dict()返回一个包含模型所有参数的字典，将其保存为名为mlp_model.pth的文件。
然后，使用MLP(input_size=X_train.shape[1])创建了一个新的MLP模型（loaded_model），并使用load_state_dict方法加载保存的模型参数。
最后，通过loaded_model.eval()将模型设置为评估模式。在PyTorch中，加载模型参数后需要将模型设置为eval模式，以确保在推理时不会影响到Batch Normalization和Dropout等层的行为。
## 模型评估
下面是模型评估的代码，包括在测试集上进行预测、计算模型准确率以及绘制混淆矩阵：

```python

# 在测试集上进行预测with torch.no_grad():
    outputs = loaded_model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)

# 计算模型准确率
accuracy = accuracy_score(predicted.numpy(), y_test)
print(f'模型在测试集上的准确率: {accuracy}')

# 计算混淆矩阵
cm = confusion_matrix(y_test, predicted)

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

```

在这段代码中，首先使用加载的模型在测试集上进行预测。通过torch.no_grad()上下文管理器，确保在预测过程中不计算梯度，因为在评估阶段不再需要反向传播。
然后，使用accuracy_score计算模型的准确率，将模型在测试集上的预测结果与实际标签进行比较。
最后，使用confusion_matrix计算混淆矩阵，该矩阵提供了模型在每个类别上的分类性能。通过seaborn和matplotlib库绘制混淆矩阵，以便更详细地了解模型的分类效果。
## 新数据预测

```python
# 读取新数据集
testData = pd.read_csv('testData.csv')

# Drop不需要的列，与训练数据预处理方式相同
new_data = testData.drop(['cell_id', 'x', 'y', 'Cell Cluster', 'Cluster'], axis=1)

# Load保存的模型
loaded_model = MLP(input_size=new_data.shape[1])
loaded_model.load_state_dict(torch.load('mlp_model.pth'))
loaded_model.eval()

# 将新数据转换为PyTorch张量
new_data_tensor = torch.tensor(new_data.values, dtype=torch.float32)

# 使用模型进行新数据的预测
with torch.no_grad():
    new_outputs = loaded_model(new_data_tensor)
    _, new_predicted = torch.max(new_outputs, 1)

# Decode预测的标签
decoded_predictions = label_encoder.inverse_transform(new_predicted.numpy())

# 打印或使用预测结果
print("Predictions on new data:")
print(decoded_predictions)

```

在这段代码中，首先通过pd.read_csv读取了新的数据集。然后，与训练数据相同的方式，通过drop删除不需要的列，进行必要的预处理。
接着，通过创建一个新的MLP模型，并加载已保存的模型参数，将模型设置为评估模式。
将新数据转换为PyTorch张量，使用加载的模型进行预测。通过torch.no_grad()确保在预测过程中不计算梯度。
最后，使用label_encoder.inverse_transform将预测的标签反向转换，以获取可读的标签。打印或使用这些预测结果，具体取决于你的需求。
# 项目链接
CSDN：https://blog.csdn.net/m0_46573428/article/details/136087419
# 后记
如果觉得有帮助的话，求 关注、收藏、点赞、星星 哦！
