#!/usr/bin/env python3
"""
TopsRider安装包分析脚本
分析解压后的TopsRider安装包，提取ECCL、DTU、GCU相关的组件和配置信息
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any

class TopsRiderAnalyzer:
    def __init__(self, package_path: str):
        self.package_path = Path(package_path)
        self.analysis_result = {
            "package_info": {},
            "eccl_components": {},
            "gcu_components": {},
            "dtu_components": {},
            "torch_gcu_packages": {},
            "sdk_components": {},
            "distributed_tools": {},
            "container_tools": {},
            "k8s_components": {}
        }
    
    def analyze_package_structure(self):
        """分析包的基本结构"""
        print("🔍 分析TopsRider包结构...")
        
        if not self.package_path.exists():
            print(f"❌ 包路径不存在: {self.package_path}")
            return
        
        # 基本信息
        self.analysis_result["package_info"] = {
            "path": str(self.package_path),
            "size": self._get_directory_size(self.package_path),
            "main_directories": [d.name for d in self.package_path.iterdir() if d.is_dir()],
            "main_files": [f.name for f in self.package_path.iterdir() if f.is_file()]
        }
        
        print(f"✓ 包路径: {self.package_path}")
        print(f"✓ 包大小: {self.analysis_result['package_info']['size']}")
        print(f"✓ 主要目录: {self.analysis_result['package_info']['main_directories']}")
    
    def analyze_eccl_components(self):
        """分析ECCL相关组件"""
        print("\n🔍 分析ECCL组件...")
        
        eccl_files = []
        eccl_configs = []
        
        # 查找ECCL相关文件
        for root, dirs, files in os.walk(self.package_path):
            for file in files:
                file_path = Path(root) / file
                if any(keyword in file.lower() for keyword in ['eccl', 'tops-eccl']):
                    eccl_files.append({
                        "name": file,
                        "path": str(file_path.relative_to(self.package_path)),
                        "size": file_path.stat().st_size if file_path.exists() else 0,
                        "type": self._get_file_type(file)
                    })
        
        # 查找ECCL配置和环境变量
        for root, dirs, files in os.walk(self.package_path):
            for file in files:
                if file.endswith(('.sh', '.py', '.json', '.yaml', '.yml')):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            if 'ECCL' in content or 'eccl' in content:
                                eccl_configs.append({
                                    "file": str(file_path.relative_to(self.package_path)),
                                    "eccl_vars": self._extract_eccl_variables(content)
                                })
                    except:
                        continue
        
        self.analysis_result["eccl_components"] = {
            "files": eccl_files,
            "configurations": eccl_configs,
            "deb_packages": [f for f in eccl_files if f["name"].endswith('.deb')],
            "libraries": [f for f in eccl_files if f["name"].endswith('.so')],
            "headers": [f for f in eccl_files if f["name"].endswith('.h')]
        }
        
        print(f"✓ 找到ECCL文件: {len(eccl_files)}个")
        print(f"✓ 找到ECCL配置: {len(eccl_configs)}个")
        
        # 显示重要的ECCL文件
        for file_info in eccl_files[:5]:  # 显示前5个
            print(f"  - {file_info['name']} ({file_info['type']}) - {file_info['path']}")
    
    def analyze_gcu_components(self):
        """分析GCU相关组件"""
        print("\n🔍 分析GCU组件...")
        
        gcu_files = []
        gcu_configs = []
        
        # 查找GCU相关文件
        for root, dirs, files in os.walk(self.package_path):
            for file in files:
                file_path = Path(root) / file
                if any(keyword in file.lower() for keyword in ['gcu', 'torch_gcu']):
                    gcu_files.append({
                        "name": file,
                        "path": str(file_path.relative_to(self.package_path)),
                        "size": file_path.stat().st_size if file_path.exists() else 0,
                        "type": self._get_file_type(file)
                    })
        
        # 查找GCU配置
        for root, dirs, files in os.walk(self.package_path):
            for file in files:
                if file.endswith(('.sh', '.py', '.json', '.yaml', '.yml')):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            if 'GCU' in content or 'gcu' in content:
                                gcu_configs.append({
                                    "file": str(file_path.relative_to(self.package_path)),
                                    "gcu_vars": self._extract_gcu_variables(content)
                                })
                    except:
                        continue
        
        self.analysis_result["gcu_components"] = {
            "files": gcu_files,
            "configurations": gcu_configs,
            "wheel_packages": [f for f in gcu_files if f["name"].endswith('.whl')],
            "libraries": [f for f in gcu_files if f["name"].endswith('.so')],
            "scripts": [f for f in gcu_files if f["name"].endswith('.sh')]
        }
        
        print(f"✓ 找到GCU文件: {len(gcu_files)}个")
        print(f"✓ 找到GCU配置: {len(gcu_configs)}个")
    
    def analyze_torch_gcu_packages(self):
        """分析torch_gcu相关包"""
        print("\n🔍 分析torch_gcu包...")
        
        torch_packages = []
        
        # 查找torch_gcu wheel文件
        for root, dirs, files in os.walk(self.package_path):
            for file in files:
                if 'torch_gcu' in file and file.endswith('.whl'):
                    file_path = Path(root) / file
                    torch_packages.append({
                        "name": file,
                        "path": str(file_path.relative_to(self.package_path)),
                        "size": file_path.stat().st_size,
                        "version": self._extract_version_from_filename(file),
                        "python_version": self._extract_python_version(file)
                    })
        
        self.analysis_result["torch_gcu_packages"] = {
            "packages": torch_packages,
            "versions": list(set([pkg["version"] for pkg in torch_packages if pkg["version"]])),
            "python_versions": list(set([pkg["python_version"] for pkg in torch_packages if pkg["python_version"]]))
        }
        
        print(f"✓ 找到torch_gcu包: {len(torch_packages)}个")
        for pkg in torch_packages:
            print(f"  - {pkg['name']} (版本: {pkg['version']}, Python: {pkg['python_version']})")
    
    def analyze_sdk_components(self):
        """分析SDK组件"""
        print("\n🔍 分析SDK组件...")
        
        sdk_files = []
        
        # 查找SDK目录和文件
        sdk_dir = self.package_path / "sdk"
        if sdk_dir.exists():
            for file in sdk_dir.iterdir():
                if file.is_file():
                    sdk_files.append({
                        "name": file.name,
                        "path": str(file.relative_to(self.package_path)),
                        "size": file.stat().st_size,
                        "type": self._get_file_type(file.name)
                    })
        
        self.analysis_result["sdk_components"] = {
            "files": sdk_files,
            "deb_packages": [f for f in sdk_files if f["name"].endswith('.deb')]
        }
        
        print(f"✓ 找到SDK文件: {len(sdk_files)}个")
    
    def analyze_distributed_tools(self):
        """分析分布式训练工具"""
        print("\n🔍 分析分布式训练工具...")
        
        distributed_files = []
        
        # 查找distributed目录
        distributed_dir = self.package_path / "distributed"
        if distributed_dir.exists():
            for root, dirs, files in os.walk(distributed_dir):
                for file in files:
                    file_path = Path(root) / file
                    distributed_files.append({
                        "name": file,
                        "path": str(file_path.relative_to(self.package_path)),
                        "size": file_path.stat().st_size,
                        "type": self._get_file_type(file)
                    })
        
        # 查找AI开发工具包中的分布式工具
        ai_toolkit_dir = self.package_path / "ai_development_toolkit" / "distributed"
        if ai_toolkit_dir.exists():
            for root, dirs, files in os.walk(ai_toolkit_dir):
                for file in files:
                    file_path = Path(root) / file
                    distributed_files.append({
                        "name": file,
                        "path": str(file_path.relative_to(self.package_path)),
                        "size": file_path.stat().st_size,
                        "type": self._get_file_type(file)
                    })
        
        self.analysis_result["distributed_tools"] = {
            "files": distributed_files,
            "wheel_packages": [f for f in distributed_files if f["name"].endswith('.whl')],
            "scripts": [f for f in distributed_files if f["name"].endswith('.sh')],
            "documents": [f for f in distributed_files if f["name"].endswith('.md')]
        }
        
        print(f"✓ 找到分布式工具文件: {len(distributed_files)}个")
    
    def analyze_container_tools(self):
        """分析容器工具"""
        print("\n🔍 分析容器工具...")
        
        container_files = []
        
        # 查找容器相关目录
        for subdir in ["deployment", "dockerfile"]:
            container_dir = self.package_path / subdir
            if container_dir.exists():
                for root, dirs, files in os.walk(container_dir):
                    for file in files:
                        file_path = Path(root) / file
                        container_files.append({
                            "name": file,
                            "path": str(file_path.relative_to(self.package_path)),
                            "size": file_path.stat().st_size,
                            "type": self._get_file_type(file)
                        })
        
        self.analysis_result["container_tools"] = {
            "files": container_files,
            "dockerfiles": [f for f in container_files if 'dockerfile' in f["name"].lower()],
            "yaml_configs": [f for f in container_files if f["name"].endswith(('.yaml', '.yml'))],
            "shell_scripts": [f for f in container_files if f["name"].endswith('.sh')]
        }
        
        print(f"✓ 找到容器工具文件: {len(container_files)}个")
    
    def _get_directory_size(self, path: Path) -> str:
        """获取目录大小"""
        try:
            result = subprocess.run(['du', '-sh', str(path)], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.split()[0]
        except:
            pass
        return "未知"
    
    def _get_file_type(self, filename: str) -> str:
        """获取文件类型"""
        if filename.endswith('.deb'):
            return 'Debian包'
        elif filename.endswith('.whl'):
            return 'Python Wheel包'
        elif filename.endswith('.so'):
            return '动态库'
        elif filename.endswith('.h'):
            return '头文件'
        elif filename.endswith('.sh'):
            return 'Shell脚本'
        elif filename.endswith('.py'):
            return 'Python脚本'
        elif filename.endswith(('.yaml', '.yml')):
            return 'YAML配置'
        elif filename.endswith('.json'):
            return 'JSON配置'
        elif filename.endswith('.md'):
            return 'Markdown文档'
        else:
            return '其他'
    
    def _extract_eccl_variables(self, content: str) -> List[str]:
        """提取ECCL相关环境变量"""
        eccl_vars = []
        lines = content.split('\n')
        for line in lines:
            if 'ECCL' in line and ('export' in line or 'set' in line):
                eccl_vars.append(line.strip())
        return eccl_vars
    
    def _extract_gcu_variables(self, content: str) -> List[str]:
        """提取GCU相关环境变量"""
        gcu_vars = []
        lines = content.split('\n')
        for line in lines:
            if 'GCU' in line and ('export' in line or 'set' in line):
                gcu_vars.append(line.strip())
        return gcu_vars
    
    def _extract_version_from_filename(self, filename: str) -> str:
        """从文件名提取版本号"""
        import re
        version_pattern = r'(\d+\.\d+\.\d+[^\-]*)'
        match = re.search(version_pattern, filename)
        return match.group(1) if match else "未知"
    
    def _extract_python_version(self, filename: str) -> str:
        """从文件名提取Python版本"""
        if 'cp36' in filename:
            return '3.6'
        elif 'cp38' in filename:
            return '3.8'
        elif 'py3' in filename:
            return '3.x'
        return "未知"
    
    def generate_report(self):
        """生成分析报告"""
        print("\n📊 生成分析报告...")
        
        report_file = Path("topsrider_analysis_report.json")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_result, f, ensure_ascii=False, indent=2)
        
        # 生成可读性报告
        readable_report = self._generate_readable_report()
        readable_file = Path("topsrider_analysis_report.md")
        
        with open(readable_file, 'w', encoding='utf-8') as f:
            f.write(readable_report)
        
        print(f"✓ JSON报告已保存: {report_file}")
        print(f"✓ 可读报告已保存: {readable_file}")
        
        return self.analysis_result
    
    def _generate_readable_report(self) -> str:
        """生成可读性报告"""
        report = "# TopsRider安装包分析报告\n\n"
        
        # 基本信息
        report += "## 📦 包基本信息\n\n"
        info = self.analysis_result["package_info"]
        report += f"- **包路径**: {info['path']}\n"
        report += f"- **包大小**: {info['size']}\n"
        report += f"- **主要目录**: {', '.join(info['main_directories'])}\n\n"
        
        # ECCL组件
        report += "## 🔧 ECCL组件分析\n\n"
        eccl = self.analysis_result["eccl_components"]
        report += f"- **ECCL文件总数**: {len(eccl['files'])}\n"
        report += f"- **DEB包**: {len(eccl['deb_packages'])}\n"
        report += f"- **动态库**: {len(eccl['libraries'])}\n"
        report += f"- **头文件**: {len(eccl['headers'])}\n"
        report += f"- **配置文件**: {len(eccl['configurations'])}\n\n"
        
        if eccl['deb_packages']:
            report += "### ECCL DEB包\n"
            for pkg in eccl['deb_packages']:
                report += f"- {pkg['name']} ({pkg['path']})\n"
            report += "\n"
        
        # GCU组件
        report += "## 🎯 GCU组件分析\n\n"
        gcu = self.analysis_result["gcu_components"]
        report += f"- **GCU文件总数**: {len(gcu['files'])}\n"
        report += f"- **Wheel包**: {len(gcu['wheel_packages'])}\n"
        report += f"- **动态库**: {len(gcu['libraries'])}\n"
        report += f"- **脚本**: {len(gcu['scripts'])}\n\n"
        
        # torch_gcu包
        report += "## 🔥 torch_gcu包分析\n\n"
        torch_gcu = self.analysis_result["torch_gcu_packages"]
        report += f"- **torch_gcu包数量**: {len(torch_gcu['packages'])}\n"
        report += f"- **支持版本**: {', '.join(torch_gcu['versions'])}\n"
        report += f"- **Python版本**: {', '.join(torch_gcu['python_versions'])}\n\n"
        
        if torch_gcu['packages']:
            report += "### torch_gcu包详情\n"
            for pkg in torch_gcu['packages']:
                report += f"- **{pkg['name']}**\n"
                report += f"  - 版本: {pkg['version']}\n"
                report += f"  - Python: {pkg['python_version']}\n"
                report += f"  - 大小: {pkg['size']} bytes\n"
                report += f"  - 路径: {pkg['path']}\n\n"
        
        # SDK组件
        report += "## 🛠️ SDK组件分析\n\n"
        sdk = self.analysis_result["sdk_components"]
        report += f"- **SDK文件总数**: {len(sdk['files'])}\n"
        report += f"- **DEB包**: {len(sdk['deb_packages'])}\n\n"
        
        # 分布式工具
        report += "## 🌐 分布式训练工具\n\n"
        dist = self.analysis_result["distributed_tools"]
        report += f"- **分布式工具文件**: {len(dist['files'])}\n"
        report += f"- **Wheel包**: {len(dist['wheel_packages'])}\n"
        report += f"- **脚本**: {len(dist['scripts'])}\n"
        report += f"- **文档**: {len(dist['documents'])}\n\n"
        
        # 容器工具
        report += "## 🐳 容器工具\n\n"
        container = self.analysis_result["container_tools"]
        report += f"- **容器工具文件**: {len(container['files'])}\n"
        report += f"- **Dockerfile**: {len(container['dockerfiles'])}\n"
        report += f"- **YAML配置**: {len(container['yaml_configs'])}\n"
        report += f"- **Shell脚本**: {len(container['shell_scripts'])}\n\n"
        
        # 重要发现
        report += "## 🎯 重要发现\n\n"
        
        # 检查是否有ECCL DEB包
        if eccl['deb_packages']:
            report += "✅ **发现ECCL DEB安装包**\n"
            for pkg in eccl['deb_packages']:
                report += f"   - {pkg['name']}\n"
            report += "\n"
        
        # 检查torch_gcu版本
        if torch_gcu['packages']:
            report += "✅ **发现torch_gcu包**\n"
            for version in torch_gcu['versions']:
                report += f"   - 版本: {version}\n"
            report += "\n"
        
        # 检查SDK包
        if sdk['deb_packages']:
            report += "✅ **发现SDK DEB包**\n"
            for pkg in sdk['deb_packages']:
                report += f"   - {pkg['name']}\n"
            report += "\n"
        
        report += "## 📋 建议\n\n"
        report += "1. **安装ECCL**: 使用找到的tops-eccl DEB包进行安装\n"
        report += "2. **安装SDK**: 安装tops-sdk和topsfactor DEB包\n"
        report += "3. **安装torch_gcu**: 选择合适的Python版本对应的torch_gcu wheel包\n"
        report += "4. **配置环境**: 参考分布式训练脚本中的ECCL环境变量配置\n"
        report += "5. **测试验证**: 使用提供的测试脚本验证安装结果\n\n"
        
        return report
    
    def run_full_analysis(self):
        """运行完整分析"""
        print("🚀 开始TopsRider包分析...")
        
        self.analyze_package_structure()
        self.analyze_eccl_components()
        self.analyze_gcu_components()
        self.analyze_torch_gcu_packages()
        self.analyze_sdk_components()
        self.analyze_distributed_tools()
        self.analyze_container_tools()
        
        result = self.generate_report()
        
        print("\n✅ 分析完成！")
        return result

def main():
    # TopsRider包路径
    package_path = "/Users/barryzhang/myDev3/MapSage_V5/test/topsrider_extracted/TopsRider_t2x_2.5.136_deb_amd64"
    
    if not os.path.exists(package_path):
        print(f"❌ 包路径不存在: {package_path}")
        return
    
    # 创建分析器并运行分析
    analyzer = TopsRiderAnalyzer(package_path)
    result = analyzer.run_full_analysis()
    
    # 显示关键信息摘要
    print("\n" + "="*60)
    print("📊 关键信息摘要")
    print("="*60)
    
    eccl_files = len(result["eccl_components"]["files"])
    gcu_files = len(result["gcu_components"]["files"])
    torch_packages = len(result["torch_gcu_packages"]["packages"])
    sdk_files = len(result["sdk_components"]["files"])
    
    print(f"🔧 ECCL相关文件: {eccl_files}个")
    print(f"🎯 GCU相关文件: {gcu_files}个")
    print(f"🔥 torch_gcu包: {torch_packages}个")
    print(f"🛠️ SDK文件: {sdk_files}个")
    
    if result["torch_gcu_packages"]["versions"]:
        print(f"📦 torch_gcu版本: {', '.join(result['torch_gcu_packages']['versions'])}")
    
    print("\n📄 详细报告已生成:")
    print("  - topsrider_analysis_report.json (JSON格式)")
    print("  - topsrider_analysis_report.md (Markdown格式)")

if __name__ == "__main__":
    main()