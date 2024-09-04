# How to Release?
当前，PyTorch的发布支持3种模式，包括有**source**，**wheel**和**docker**。该文档用于指导怎么快速使用和构建发布包。
## 文件目录
为了提高自动化和高效率，主要使用shell脚本进行构建。目录结构主要有：
```bash
├── build.property		# 依赖包版本指定
├── torch_mlu_trim_files.sh››   # 删除git，jenkins以及其他信息
├── env_pytorch.sh		# 设置编译以及运行所需的环境变量，由independent_build.sh调用
├── gen_cambricon_neuware_lib.sh›   ›   # 一键安装cntoolkit, cnnl, cncl, magicmind
├── independent_build.sh	# shell脚本，用于制作source/wheel/docker
├── json_parser.py		# 用于解析build.property
├── make_base_image.sh		# shell脚本用于制作base基础镜像
└──  README.md
```
## 脚本
1. make_base_image.sh
	- 命令
		```bash
		make_base_image.sh [-f DOCKERFILE] [-n REPOSITORY] [-t TAG]
		```
	- 用途
		- 指定base文件下的dockerfile，**制作基础镜像**
	- 参数
		- -f : 指定base dockerfile
		- -n : 传入REPOSITORY名称
		- -t : 传入TAG名称
2. independent_build.sh
	- 命令
		```bash
		independent_build.sh ...
		```
	- 用途
		- 用于生成3种发布包
	- 参数
		- -r: 指定3种发布形式，支持的字符包括`src`, `wheel`和`docker`
		- -b: 指定pytorch分支
		- -c: 指定torch_mlu分支
		- -p: 指定pytorch_models分支
		- -o: 指定操作系统的名称，支持的字符包括`ubuntu`，`debian`和`centos`，均为小写
		- -v: 指定操作系统具体的版本，ubuntu对应`16.04`和`18.04`，debian对应`9`，centos对用`7.4`和`7.6`
		- -w: 传入REPOSITORY名称
		- -d: 传入REPOSITORY名称
		- -t: 传入TAG名称
		- -f: 指定base dockerfile
    - -e: 指定python版本

3. gen_cambricon_neuware_lib.sh
    - 命令示例
      ```bash
      ./gen_cambricon_neuware_lib.sh -o 0 -p CambriconNeuware_v2.2.1_CentOS7
      ```
    - 用途
      - 一键安装cntoolkit, cnnl, cncl, magicmind。
    - 参数
      - -o: 指定操作系统，0表示Centos，1表示Ubuntu
      - -p: 指定安装包存放路径
## 生成base
之所以有base image这一步，有两个原因：
1. 编译pytorch所需要的依赖库过多，每次从头开始构建wheel或docker耗时严重；
2. 当前发布需要统一gcc-7、python3.6和openmpi，所以需要对原始镜像进行改动，以适应发布的要求
注意：gcc的安装统一使用的是yum或者apt安装，未使用源码编译；python3.6统一使用源码编译；openmpi统一使用源码编译，版本为4.0.5

```bash
cd torch_mlu根目录
bash script/release/make_base_image.sh -f docker/base/manylinux2014_x86_64-base.Dockerfile -n yellow.hub.cambricon.com/framework/base_images/manylinux -t manylinux2014_x86_64_torch_py36_gcc7 -p 3.6
```
在执行结束之后会在本地生成一个`REPOSITORY`为yellow.hub.cambricon.com/framework/base_images/manylinux, `TAG`为manylinux2014_x86_64_torch_py36_gcc7的镜像

## Source
```bash
cd torch_mlu根目录
bash script/release/independent_build.sh -r src
```
会在本地生成一个压缩包`Cambricon-PyTorch-ABC.tar.gz`

## Wheel
```bash
cd torch_mlu根目录
bash script/release/independent_build.sh -r wheel
```
manylinux中会进行`pytorch`、`torch_mlu`代码编译，编译完成之后会在本地创建一个文件夹，包含上述组件的wheel包。

## Docker
```bash
cd torch_mlu根目录
bash script/release/independent_build.sh -r docker -d pytorch_centos7.4 -t beta -f docker/dockerfile.base_centos7.4.framework .
```
执行以下几个步骤：
1. 会将本地的wheel包拷贝到镜像；
2. 安装wheel包到虚拟环境中；
3. 下载`pytorch`、`torch_mlu`、`pytorch_models`源码并且删除git信息
4. 所有的发布文件均放在/torch目录下，包含有Neuware依赖包，venv3虚拟环境，src源码包等
完成以上步骤，会在本地生成一个`REPOSITORY`为pytorch_centos7.4，`TAG`为beta的镜像
