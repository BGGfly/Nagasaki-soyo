import requests
import os
from tqdm import tqdm
import time


class CWRUDownloader:
    def __init__(self):
        self.base_url = "https://engineering.case.edu/sites/default/files/"
        self.download_folder = "CWRU_Bearing_Data"

        # 推荐下载的文件列表（16个核心文件）
        self.recommended_files = [
            # 正常数据 (4个转速)
            "97.mat", "98.mat", "99.mat", "100.mat",

            # 内圈故障 0.007英寸 (4个转速)
            "105.mat", "106.mat", "107.mat", "108.mat",

            # 外圈故障 0.007英寸 (4个转速)
            "169.mat", "170.mat", "171.mat", "172.mat",

            # 滚动体故障 0.007英寸 (4个转速)
            "118.mat", "119.mat", "120.mat", "121.mat"
        ]

        # 文件信息说明
        self.file_info = {
            "97.mat": "正常轴承, 1797 RPM, 0 HP",
            "98.mat": "正常轴承, 1772 RPM, 0 HP",
            "99.mat": "正常轴承, 1750 RPM, 0 HP",
            "100.mat": "正常轴承, 1730 RPM, 0 HP",
            "105.mat": "内圈故障(0.007英寸), 1797 RPM, 0 HP",
            "106.mat": "内圈故障(0.007英寸), 1772 RPM, 0 HP",
            "107.mat": "内圈故障(0.007英寸), 1750 RPM, 0 HP",
            "108.mat": "内圈故障(0.007英寸), 1730 RPM, 0 HP",
            "169.mat": "外圈故障(0.007英寸), 1797 RPM, 0 HP",
            "170.mat": "外圈故障(0.007英寸), 1772 RPM, 0 HP",
            "171.mat": "外圈故障(0.007英寸), 1750 RPM, 0 HP",
            "172.mat": "外圈故障(0.007英寸), 1730 RPM, 0 HP",
            "118.mat": "滚动体故障(0.007英寸), 1797 RPM, 0 HP",
            "119.mat": "滚动体故障(0.007英寸), 1772 RPM, 0 HP",
            "120.mat": "滚动体故障(0.007英寸), 1750 RPM, 0 HP",
            "121.mat": "滚动体故障(0.007英寸), 1730 RPM, 0 HP"
        }

    def download_file(self, filename, max_retries=3):
        """下载单个文件"""
        file_url = self.base_url + filename
        save_path = os.path.join(self.download_folder, filename)

        # 如果文件已存在，跳过下载
        if os.path.exists(save_path):
            file_size = os.path.getsize(save_path)
            print(f"✓ 文件已存在: {filename} ({file_size / 1024 / 1024:.1f} MB) - {self.file_info[filename]}")
            return True

        for attempt in range(max_retries):
            try:
                print(f"↓ 下载中 ({attempt + 1}/{max_retries}): {filename} - {self.file_info[filename]}")

                response = requests.get(file_url, stream=True, timeout=60)
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))

                with open(save_path, 'wb') as f, tqdm(
                        desc=f"  {filename}",
                        total=total_size,
                        unit='iB',
                        unit_scale=True,
                        unit_divisor=1024,
                        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

                print(f"✓ 下载完成: {filename} ({total_size / 1024 / 1024:.1f} MB)")
                return True

            except requests.exceptions.RequestException as e:
                print(f"✗ 下载失败 {filename} (尝试 {attempt + 1}): {e}")
                time.sleep(2)  # 等待2秒后重试
            except Exception as e:
                print(f"✗ 错误 {filename}: {e}")
                break

        return False

    def download_all_files(self, file_list=None):
        """下载所有推荐的文件"""
        if file_list is None:
            file_list = self.recommended_files

        # 创建下载目录
        os.makedirs(self.download_folder, exist_ok=True)
        print(f"📁 下载目录: {os.path.abspath(self.download_folder)}")
        print(f"📊 总共要下载 {len(file_list)} 个文件")
        print("=" * 60)

        # 使用不同的变量名避免冲突
        downloaded_count = 0
        failed_list1 = []

        for i, filename in enumerate(file_list, 1):
            print(f"\n[{i}/{len(file_list)}] ", end="")
            if self.download_file(filename):
                downloaded_count += 1
            else:
                failed_list1.append(filename)

            # 稍微延迟一下，避免请求过于频繁
            time.sleep(0.5)

        # 打印下载总结
        print("\n" + "=" * 60)
        print("📋 下载总结:")
        print(f"✅ 成功下载: {downloaded_count} 个文件")
        print(f"❌ 失败: {len(failed_list1)} 个文件")

        if failed_list1:
            print("失败的文件:")
            for f in failed_list1:
                print(f"  - {f}")

        total_size = self.calculate_total_size()
        print(f"💾 总数据量: {total_size / 1024 / 1024:.1f} MB")
        print(f"📁 文件位置: {os.path.abspath(self.download_folder)}")

        return downloaded_count, failed_list

    def calculate_total_size(self):
        """计算已下载文件的总大小"""
        total_size = 0
        for filename in os.listdir(self.download_folder):
            if filename.endswith('.mat'):
                file_path = os.path.join(self.download_folder, filename)
                total_size += os.path.getsize(file_path)
        return total_size

    def check_downloaded_files(self):
        """检查已下载的文件"""
        print("\n🔍 检查已下载文件...")
        downloaded_files1 = [f for f in os.listdir(self.download_folder) if f.endswith('.mat')]

        print(f"找到 {len(downloaded_files1)} 个.mat文件:")
        for filename in downloaded_files1:
            file_path = os.path.join(self.download_folder, filename)
            file_size = os.path.getsize(file_path) / 1024 / 1024
            file_info = self.file_info.get(filename, "未知文件")
            print(f"  ✓ {filename} ({file_size:.1f} MB) - {file_info}")

        return downloaded_files1

    def create_readme_file(self):
        """创建说明文件"""
        readme_path = os.path.join(self.download_folder, "README.txt")

        readme_content = """CWRU轴承故障数据集说明

数据集来源: Case Western Reserve University Bearing Data Center
网址: https://engineering.case.edu/bearingdatacenter

包含文件:
==========

正常轴承:
  97.mat - 1797 RPM, 0 HP
  98.mat - 1772 RPM, 0 HP  
  99.mat - 1750 RPM, 0 HP
  100.mat - 1730 RPM, 0 HP

内圈故障 (0.007英寸):
  105.mat - 1797 RPM, 0 HP
  106.mat - 1772 RPM, 0 HP
  107.mat - 1750 RPM, 0 HP
  108.mat - 1730 RPM, 0 HP

外圈故障 (0.007英寸):
  169.mat - 1797 RPM, 0 HP
  170.mat - 1772 RPM, 0 HP
  171.mat - 1750 RPM, 0 HP
  172.mat - 1730 RPM, 0 HP

滚动体故障 (0.007英寸):
  118.mat - 1797 RPM, 0 HP
  119.mat - 1772 RPM, 0 HP
  120.mat - 1750 RPM, 0 HP
  121.mat - 1730 RPM, 0 HP

数据格式:
每个.mat文件包含多个变量，主要振动数据变量命名格式:
  X{文件编号}_DE_time - 驱动端振动信号
  X{文件编号}_FE_time - 风扇端振动信号
  X{文件编号}_BA_time - 基座振动信号

采样频率: 12 kHz
信号长度: 121,000 个采样点

建议使用驱动端(DE)数据进行故障诊断分析。
"""

        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)

        print(f"📝 已创建说明文件: {readme_path}")


# 执行下载
if __name__ == "__main__":
    print("开始下载CWRU轴承故障数据集...")
    print("=" * 60)

    downloader = CWRUDownloader()

    # 下载所有推荐文件
    success_count, failed_list = downloader.download_all_files()

    # 检查下载的文件
    downloaded_files = downloader.check_downloaded_files()

    # 创建说明文件
    downloader.create_readme_file()

    print("\n🎉 下载完成！")
    print(f"您可以在 '{os.path.abspath(downloader.download_folder)}' 文件夹中找到所有数据")

    if failed_list:
        print("\n⚠️  注意：有些文件下载失败，您可以稍后重试下载")