# 运行本代码，需要在云端下载模型
from PIL import Image
import numpy as np
from XEdu.hub import Workflow as wf # 导入库
def get_img(img_path,bbox):
    print(bbox)
    # 处理不同类型的边界框输入
    if isinstance(bbox, str):
        coords = bbox.strip().split()
        x1, y1, x2, y2 = map(float, coords)
    elif isinstance(bbox, (np.ndarray, list, tuple)):
        x1, y1, x2, y2 = map(float, bbox)
    else:
        print(f"错误：不支持的边界框类型 - {type(bbox)}")
        return None
    
    # 打开图像
    try:
        img = Image.open(img_path)
    except FileNotFoundError:
        print(f"错误：找不到图像文件 '{img_path}'")
        return None
    except Exception as e:
        print(f"错误：打开图像时出错 - {e}")
        return None
    
    # 裁剪图像
    cropped_img = img.crop((x1, y1, x2, y2))
    return cropped_img

det  = wf(task='det_coco_l') # 实例化模型
cls = wf(task='cls_imagenet') # 实例化模型
img_path = 'img.jpg' # 指定进行推理的图片路径
bboxs,new_img = det.inference(data=img_path,img_type='pil') # 进行目标检测推理
# print(bboxs)
# det.show(new_img)
for i in bboxs:
    test = get_img(img_path,i)
    result =cls.inference(data=test) # 进行关键点检测推理
    print(result)
