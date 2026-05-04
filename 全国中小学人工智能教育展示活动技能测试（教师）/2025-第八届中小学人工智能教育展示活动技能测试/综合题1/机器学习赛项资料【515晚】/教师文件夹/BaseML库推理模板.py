from XEdu.hub import Workflow as wf
import pandas as pd


# 推理函数参数：模型文件名，功能：推理测试集A，生成一个推理结果csv；
# 上交当前python文件后，路径会被替换为测试集B，得到最终结果。
def BodyFat_inference(model_name='cmodel.pkl'):
    baseml = wf(task='baseml', checkpoint=model_name) 
    test_A_path = "data/测试集A.csv"
    test_data = pd.read_csv(test_A_path)
    #如有数据预处理函数，请添加在下方
    

    
    X_test = test_data.iloc[:, :-1].values  # 除最后一列外的所有列作为特征
    y_pred = baseml.inference(data=X_test)
    print('测试集A的推理结果为：',y_pred)
    pred_df = pd.DataFrame(y_pred, columns=['预测值'])
    pred_csv_path = "data/预测结果.csv"
    pred_df.to_csv(pred_csv_path, index=False)

BodyFat_inference()