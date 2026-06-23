# 3D-step_mask2former
3d模型转化为图片进行特征识别
3d step → 2d png → 3d step
自带数据集标注器，在step上直接标注，一件生成多视角（12、24）掩码图像和带RGB编码器的图像作为模型输入传入mask2former训练
2d→3d，将2d图像采用颜色编码回传到3d模型上
