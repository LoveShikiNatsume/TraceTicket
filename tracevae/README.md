# TraceVAE
这是"通过图变分自编码器实现微服务追踪数据无监督异常检测"的源代码。

使用说明
安装依赖包：pip3 install -r requirements.txt

数据预处理（将原始数据转换为模型输入格式）：

text
python3 -m tracegnn.cli.data_process preprocess -i [输入路径] -o [数据集路径]
示例数据集位于sample_dataset目录（注：此示例数据集仅用于演示数据格式，不能用于评估模型性能，请替换为您自己的数据集）

预处理命令示例：

text
python3 -m tracegnn.cli.data_process preprocess -i sample_dataset -o sample_dataset
python3 -m tracegnn.cli.data_process preprocess -i /home/fuxian/tracevae/TT_Dataset/TT_Dataset/convert_data -o /home/fuxian/tracevae/TT_Dataset/TT_Dataset/convert_data_time_corrected
模型训练：

text
bash train.sh [数据集路径]
训练示例：

text
bash train.sh sample_dataset
bash train.sh /home/fuxian/tracevae/TT_Dataset/TT_Dataset/convert_data_time_corrected
模型评估：

text
bash test.sh [模型路径] [数据集路径]
默认模型路径为results/train/models/final.pt

评估示例：

text
bash test.sh results/train/models/final.pt sample_dataset
bash test.sh results/train/models/final.pt /home/fuxian/tracevae/TT_Dataset/TT_Dataset/convert_data_time_corrected
bash test.sh results/train/models/final.pt /home/fuxian/TraceTicket/tracevae/TT_Dataset/TT_Dataset/convert_data_time_corrected