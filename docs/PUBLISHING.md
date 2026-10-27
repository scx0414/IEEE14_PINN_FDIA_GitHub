# 上传GitHub

本文件夹是独立的发布副本，研究原目录未移动或删除。

## 发布内容

源代码、配置、测试、说明文档、小型实验汇总、示例网络矩阵，以及带许可的xLSTM核心。未包含原始季度数据、检查点、预测数组、论文/PPT、环境缓存、历史日志和其他Git仓库。

`.gitignore`默认排除`data/`、`artifacts/`、`runs/`、`local/`和常见二进制/文档产物。如以后单独共享数据或权重，可使用Release或外部存储，并记录哈希与对应配置；不要让演示数据指标冒充正式实验。

## 上传步骤

在当前项目文件夹运行：

```bash
git init
git add .
git status --short
git commit -m "Organize AC-PINN state estimation and FDIA detection pipeline"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

也可在GitHub新建空仓库后上传当前文件夹内容。此包没有自动创建远程仓库、提交或推送。

## 许可证

尚未替项目原创代码选择许可证。你可以在发布时按自己的共享意愿添加根目录LICENSE；本整理没有擅自决定授权范围。第三方xLSTM代码已保留其Apache2.0许可证和版权声明，不受你为原创部分选择的许可证覆盖。

不要将第三方论文或许可不明确的参考仓库原文件一起上传。本包只保留研究方法链接，原论文及DDET-MTD源码没有放入发布目录。
