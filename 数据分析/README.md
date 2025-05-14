# Nginx日志分析工具

这是一个用于分析Nginx访问日志的可视化工具，能够生成丰富的图表和HTML报告。

## 功能特点

- 按月份、日期、小时统计访问量
- 分析热门访问路径
- HTTP状态码分布统计
- 客户端浏览器和操作系统分布
- 访问量最大的IP统计
- 生成美观的HTML报告

## 使用方法

1. 将Nginx日志文件(.log)放在`数据分析`目录下
2. 运行启动脚本：
   ```
   python run_analysis.py
   ```
3. 程序会自动检查并安装必要的依赖包
4. 分析完成后，打开`数据分析/分析结果/report.html`查看完整报告

## 依赖项

程序会自动检查并安装以下依赖：
- pandas
- matplotlib
- seaborn

## 支持的日志格式

本工具支持标准的Nginx访问日志格式，例如：
```
10.8.66.187 - - [17/May/2024:19:35:26 +0800] "GET /admin/dict/item/tree/creditor_pawn_type HTTP/1.1" 200 4757 "http://100.142.80.36/" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
```

## 生成的图表

分析结果将包含以下图表：
- 月访问量统计
- 每日访问热力图
- 每小时访问趋势
- 热门访问路径TOP10
- HTTP状态码分布
- 浏览器分布
- 操作系统分布
- 访问量最大的IP TOP10

## 定制

如需定制分析维度，请修改`nginx_log_analysis.py`文件。 