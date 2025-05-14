#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib
from collections import defaultdict, Counter
import seaborn as sns

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 定义日志目录
LOG_DIR = "."  # 使用当前目录

def parse_nginx_log(log_file):
    """解析Nginx日志文件，提取时间、IP地址、请求路径、状态码等信息"""
    data = []
    
    # 正则表达式解析Nginx日志格式
    pattern = r'(\d+\.\d+\.\d+\.\d+) - - \[(.*?)\] "(.*?)" (\d+) (\d+) "(.*?)" "(.*?)"'
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.match(pattern, line)
            if match:
                ip, time_str, request, status, size, referer, user_agent = match.groups()
                
                # 解析请求方法、路径和协议
                request_parts = request.split()
                if len(request_parts) >= 2:
                    method = request_parts[0]
                    path = request_parts[1]
                else:
                    method = request
                    path = ""
                
                # 解析时间
                time_pattern = r'(\d+)/(\w+)/(\d+):(\d+):(\d+):(\d+)'
                time_match = re.match(time_pattern, time_str)
                if time_match:
                    day, month, year, hour, minute, second = time_match.groups()
                    # 将月份英文转换为数字
                    month_dict = {
                        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                    }
                    month_num = month_dict.get(month, 1)
                    date_str = f"{year}-{month_num:02d}-{day:0>2s}"
                    time = f"{hour}:{minute}:{second}"
                    
                    # 提取客户端操作系统和浏览器信息
                    browser = "其他"
                    os_name = "其他"
                    
                    if "Windows" in user_agent:
                        os_name = "Windows"
                    elif "Mac OS" in user_agent:
                        os_name = "Mac OS"
                    elif "Linux" in user_agent:
                        os_name = "Linux"
                    elif "Android" in user_agent:
                        os_name = "Android"
                    elif "iOS" in user_agent:
                        os_name = "iOS"
                    
                    if "Chrome" in user_agent:
                        browser = "Chrome"
                    elif "Firefox" in user_agent:
                        browser = "Firefox"
                    elif "Safari" in user_agent and "Chrome" not in user_agent:
                        browser = "Safari"
                    elif "MSIE" in user_agent or "Trident" in user_agent:
                        browser = "IE"
                    elif "Edge" in user_agent:
                        browser = "Edge"
                    
                    data.append({
                        'ip': ip,
                        'date': date_str,
                        'time': time,
                        'method': method,
                        'path': path,
                        'status': status,
                        'size': size,
                        'referer': referer,
                        'user_agent': user_agent,
                        'browser': browser,
                        'os': os_name
                    })
    
    return pd.DataFrame(data)

def analyze_logs():
    """分析所有日志文件并生成图表"""
    all_data = []
    
    # 遍历日志目录下的所有.log文件
    for filename in os.listdir(LOG_DIR):
        if filename.endswith('.log'):
            file_path = os.path.join(LOG_DIR, filename)
            print(f"正在分析日志文件: {file_path}")
            df = parse_nginx_log(file_path)
            all_data.append(df)
    
    # 合并所有数据
    if not all_data:
        print("未找到任何日志文件！")
        return
    
    df = pd.concat(all_data, ignore_index=True)
    
    # 转换日期列为datetime类型
    df['datetime'] = pd.to_datetime(df['date'])
    df['year_month'] = df['datetime'].dt.strftime('%Y-%m')
    
    # 基本统计分析
    
    # 1. 按月统计访问量
    monthly_visits = df['year_month'].value_counts().sort_index()
    
    # 2. 按路径统计访问量
    path_visits = df['path'].value_counts().head(10)
    
    # 3. 按状态码统计
    status_counts = df['status'].value_counts()
    
    # 4. 按小时统计访问量
    df['hour'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.hour
    hourly_visits = df['hour'].value_counts().sort_index()
    
    # 5. 按浏览器和操作系统统计
    browser_counts = df['browser'].value_counts()
    os_counts = df['os'].value_counts()
    
    # 6. IP地址访问量统计
    ip_counts = df['ip'].value_counts().head(10)
    
    # 创建输出目录
    output_dir = os.path.join(LOG_DIR, "分析结果")
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置图表样式
    plt.style.use('ggplot')
    
    # 绘制图表
    
    # 1. 月访问量图表
    plt.figure(figsize=(12, 6))
    monthly_visits.plot(kind='bar', color='skyblue')
    plt.title('月访问量统计', fontsize=16)
    plt.xlabel('月份', fontsize=14)
    plt.ylabel('访问次数', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'monthly_visits.png'), dpi=300)
    
    # 2. 热门路径图表
    plt.figure(figsize=(14, 8))
    path_visits.plot(kind='barh', color='lightgreen')
    plt.title('热门访问路径 Top 10', fontsize=16)
    plt.xlabel('访问次数', fontsize=14)
    plt.ylabel('路径', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_paths.png'), dpi=300)
    
    # 3. 状态码饼图
    plt.figure(figsize=(10, 10))
    plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', 
            colors=sns.color_palette('pastel'), startangle=90)
    plt.title('HTTP状态码分布', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'status_codes.png'), dpi=300)
    
    # 4. 小时访问趋势图
    plt.figure(figsize=(12, 6))
    hourly_visits.plot(kind='line', marker='o', color='coral', linewidth=2)
    plt.title('每小时访问趋势', fontsize=16)
    plt.xlabel('小时', fontsize=14)
    plt.ylabel('访问次数', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hourly_visits.png'), dpi=300)
    
    # 5. 浏览器分布图
    plt.figure(figsize=(10, 10))
    plt.pie(browser_counts, labels=browser_counts.index, autopct='%1.1f%%', 
            colors=sns.color_palette('Set3'), startangle=90)
    plt.title('浏览器分布', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'browser_distribution.png'), dpi=300)
    
    # 6. 操作系统分布图
    plt.figure(figsize=(10, 10))
    plt.pie(os_counts, labels=os_counts.index, autopct='%1.1f%%', 
            colors=sns.color_palette('Paired'), startangle=90)
    plt.title('操作系统分布', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'os_distribution.png'), dpi=300)
    
    # 7. 热门IP图表
    plt.figure(figsize=(14, 8))
    ip_counts.plot(kind='barh', color='lightcoral')
    plt.title('访问量最大的IP Top 10', fontsize=16)
    plt.xlabel('访问次数', fontsize=14)
    plt.ylabel('IP地址', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_ips.png'), dpi=300)
    
    # 8. 生成每日访问热力图
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    
    daily_counts = df.groupby(['month', 'day']).size().reset_index(name='count')
    daily_pivot = daily_counts.pivot(index='day', columns='month', values='count')
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(daily_pivot, annot=True, fmt='.0f', cmap='YlGnBu')
    plt.title('日访问量热力图 (月/日)', fontsize=16)
    plt.xlabel('月份', fontsize=14)
    plt.ylabel('日期', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'daily_heatmap.png'), dpi=300)
    
    # 生成HTML报告，使用更简单的方式生成
    start_date = df['date'].min()
    end_date = df['date'].max()
    total_visits = len(df)
    unique_ips = df['ip'].nunique()
    avg_daily_visits = round(len(df) / df['date'].nunique(), 2)
    success_rate = round(len(df[df['status'].str.startswith('2')]) / len(df) * 100, 2)
    report_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Nginx日志分析报告</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            border-radius: 5px;
        }}
        h1, h2 {{
            color: #2c3e50;
        }}
        h1 {{
            text-align: center;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .chart-container {{
            margin: 20px 0;
            text-align: center;
        }}
        .chart {{
            max-width: 100%;
            box-shadow: 0 0 5px rgba(0,0,0,0.2);
            border-radius: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .summary {{
            background: #eaf6ff;
            padding: 15px;
            border-radius: 5px;
            font-size: 18px;
            line-height: 1.8;
        }}
        .footer {{
            text-align: center;
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Nginx日志分析报告</h1>
        
        <div class="section summary">
            <h2>数据概览</h2>
            <p>分析时间段: {start_date} 至 {end_date}</p>
            <p>总访问量: {total_visits}次</p>
            <p>独立IP数: {unique_ips}个</p>
            <p>平均每日访问量: {avg_daily_visits}次</p>
            <p>成功请求比例(2xx状态码): {success_rate}%</p>
        </div>
        
        <div class="section">
            <h2>月访问量趋势</h2>
            <div class="chart-container">
                <img src="monthly_visits.png" alt="月访问量统计" class="chart">
            </div>
        </div>
        
        <div class="section">
            <h2>每日访问热力图</h2>
            <div class="chart-container">
                <img src="daily_heatmap.png" alt="日访问量热力图" class="chart">
            </div>
        </div>
        
        <div class="section">
            <h2>每小时访问趋势</h2>
            <div class="chart-container">
                <img src="hourly_visits.png" alt="每小时访问趋势" class="chart">
            </div>
        </div>
        
        <div class="section">
            <h2>热门访问路径</h2>
            <div class="chart-container">
                <img src="top_paths.png" alt="热门访问路径" class="chart">
            </div>
        </div>
        
        <div class="section">
            <h2>HTTP状态码分布</h2>
            <div class="chart-container">
                <img src="status_codes.png" alt="HTTP状态码分布" class="chart">
            </div>
        </div>
        
        <div class="section">
            <h2>浏览器分布</h2>
            <div class="chart-container">
                <img src="browser_distribution.png" alt="浏览器分布" class="chart">
            </div>
        </div>
        
        <div class="section">
            <h2>操作系统分布</h2>
            <div class="chart-container">
                <img src="os_distribution.png" alt="操作系统分布" class="chart">
            </div>
        </div>
        
        <div class="section">
            <h2>访问量最大的IP</h2>
            <div class="chart-container">
                <img src="top_ips.png" alt="访问量最大的IP" class="chart">
            </div>
        </div>
        
        <div class="footer">
            <p>报告生成时间: {report_time}</p>
        </div>
    </div>
</body>
</html>'''
    
    with open(os.path.join(output_dir, 'report.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"分析完成! 报告已保存到 {output_dir} 目录")

if __name__ == "__main__":
    analyze_logs() 