import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import platform

class LibreOfficeConverter:
    def __init__(self, libreoffice_path=None):
        """
        初始化转换器
        
        Args:
            libreoffice_path: LibreOffice可执行文件路径，如果为None则自动检测
        """
        self.libreoffice_path = libreoffice_path or self._detect_libreoffice()
        if not self.libreoffice_path:
            raise Exception("未找到LibreOffice，请确保已安装LibreOffice")
        
        self.supported_formats = {'.txt', '.doc', '.docx', '.ppt', '.pptx', '.odt', '.ods', '.odp'}

        print(f"使用LibreOffice路径: {self.libreoffice_path}")
    
    def _detect_libreoffice(self):
        """自动检测LibreOffice安装路径"""
        system = platform.system().lower()
        
        if system == "windows":
            # 常见安装路径
            possible_paths = [
                r"C:\Program Files\LibreOffice\program\soffice.exe",
                r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
                r"C:\Program Files\LibreOffice\program\soffice.com",  # 某些版本使用.com
            ]
        elif system == "darwin":  # macOS
            possible_paths = [
                "/Applications/LibreOffice.app/Contents/MacOS/soffice",
                "/Applications/LibreOffice.app/Contents/MacOS/soffice.bin",
            ]
        else:  # Linux和其他Unix-like系统
            possible_paths = [
                "/usr/bin/soffice",
                "/usr/local/bin/soffice",
                "/opt/libreoffice/program/soffice",
                "/snap/bin/libreoffice",
            ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # 尝试在PATH中查找
        which_cmd = "where" if system == "windows" else "which"
        try:
            result = subprocess.run([which_cmd, "soffice"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        
        return None
    
    def convert_to_pdf(self, input_path, output_dir=None, timeout=60):
        """
        将文档转换为PDF
        
        Args:
            input_path: 输入文件路径
            output_dir: 输出目录，如果为None则使用输入文件所在目录
            timeout: 转换超时时间（秒）
        
        Returns:
            生成的PDF文件路径
        """
        input_path = Path(input_path).resolve()
        
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")
        
        if output_dir is None:
            output_dir = input_path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # 支持的输入格式
        if input_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"不支持的格式: {input_path.suffix}，支持格式: {', '.join(self.supported_formats)}")
        
        try:
            # 构建LibreOffice命令
            cmd = [
                self.libreoffice_path,
                '--headless',  # 无界面模式
                '--convert-to', 'pdf',
                '--outdir', str(output_dir),
                # '--infilter', 'Text (encoded):UTF8',  # 根据实际编码调整
                str(input_path)
            ]
            
            # 执行转换
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout
            )
            
            if result.returncode != 0:
                error_msg = result.stderr.decode('utf-8', errors='ignore')
                raise Exception(f"LibreOffice转换失败: {error_msg}")
            
            # 检查输出文件
            pdf_path = output_dir / f"{input_path.stem}.pdf"
            if not pdf_path.exists():
                # 有时LibreOffice会使用不同的命名规则
                possible_names = [
                    f"{input_path.stem}.pdf",
                    f"{input_path.name.replace(input_path.suffix, '.pdf')}",
                ]
                
                for name in possible_names:
                    possible_path = output_dir / name
                    if possible_path.exists():
                        pdf_path = possible_path
                        break
                else:
                    raise FileNotFoundError("未找到生成的PDF文件")
            
            print(f"转换成功: {input_path} -> {pdf_path}")
            return str(pdf_path)
            
        except subprocess.TimeoutExpired:
            raise Exception(f"转换超时（{timeout}秒）")
        except Exception as e:
            raise Exception(f"转换过程出错: {str(e)}")
    
    def batch_convert(self, input_files, output_dir=None, timeout=60):
        """
        批量转换多个文件
        
        Args:
            input_files: 输入文件路径列表
            output_dir: 输出目录
            timeout: 每个文件的转换超时时间
        
        Returns:
            成功转换的文件列表
        """
        successful_conversions = []
        
        for input_file in input_files:
            try:
                pdf_path = self.convert_to_pdf(input_file, output_dir, timeout)
                successful_conversions.append(pdf_path)
            except Exception as e:
                print(f"转换失败 {input_file}: {str(e)}")
        
        return successful_conversions
    
    def is_available(self):
        """检查LibreOffice是否可用"""
        try:
            result = subprocess.run(
                [self.libreoffice_path, '--version'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10
            )
            return result.returncode == 0
        except:
            return False


def any2pdf(input_dir: str, output_dir: str = None):
    try:
        # 创建转换器实例
        converter = LibreOfficeConverter()
        
        # 检查LibreOffice是否可用
        if not converter.is_available():
            print("警告: LibreOffice可能未正确安装或配置")
        else:
            print("LibreOffice可用")

        # 创建输出目录
        output_dir = output_dir if output_dir else os.path.join(input_dir, "pdfs")
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # 待转化的文件列表
        file_list = [os.path.join(input_dir, file) for file in os.listdir(input_dir)]
        # 本就是pdf的文件
        pdf_files = [f for f in file_list if Path(f).exists() and Path(f).suffix.lower() == ".pdf"]

        # 批量转换
        successful = converter.batch_convert(
            [f for f in file_list if Path(f).exists() and Path(f).suffix.lower() in converter.supported_formats],
            output_dir=output_dir
        )
        
        # 将pdf移动到output_dir
        for file in pdf_files:
            shutil.move(file, output_dir)
        print(f"\n成功转换了 {len(successful)} 个文件:")
        for file in successful:
            print(f"  - {file}")
            
    except Exception as e:
        print(f"错误: {str(e)}")
        print("\n请确保已安装LibreOffice:")
        print("Windows: 从 https://libreoffice.org/download 下载安装")
        print("macOS: brew install --cask libreoffice")
        print("Linux: sudo apt install libreoffice (Ubuntu/Debian) 或使用包管理器")
    return output_dir

# 使用示例和测试函数
def main():
    """主函数 - 使用示例"""
    try:
        # 创建转换器实例
        converter = LibreOfficeConverter()
        
        # 检查LibreOffice是否可用
        if not converter.is_available():
            print("警告: LibreOffice可能未正确安装或配置")
        else:
            print("LibreOffice可用")
        
        # 测试文件列表
        test_files = [
            "./test.txt",  # Word文档
            # "example.doc",   # 旧版Word文档
            # "example.pptx",  # PowerPoint文档
            # "example.txt",   # 文本文件
        ]
        
        # 创建输出目录
        output_dir = Path("converted_pdfs")
        output_dir.mkdir(exist_ok=True)
        
        # 批量转换
        successful = converter.batch_convert(
            [f for f in test_files if Path(f).exists()],
            output_dir=output_dir
        )
        
        print(f"\n成功转换了 {len(successful)} 个文件:")
        for file in successful:
            print(f"  - {file}")
            
    except Exception as e:
        print(f"错误: {str(e)}")
        print("\n请确保已安装LibreOffice:")
        print("Windows: 从 https://libreoffice.org/download 下载安装")
        print("macOS: brew install --cask libreoffice")
        print("Linux: sudo apt install libreoffice (Ubuntu/Debian) 或使用包管理器")


def convert_single_file(input_file, output_dir=None):
    """转换单个文件的便捷函数"""
    converter = LibreOfficeConverter()
    return converter.convert_to_pdf(input_file, output_dir)


# if __name__ == "__main__":
#     main()