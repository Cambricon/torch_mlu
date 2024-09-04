# Usage
该工具中主要包含两个脚本，环境检查脚本和环境设置脚本。
## 环境检查脚本
使用该脚本来进行网络性能测试的前置检查，其中包括检查数据集是否挂载，打印CPU kernel数，检查CPU是否为性能模式，检查CPU是否独占，irqbalance检查和ACS(Access Control Services)开关检查。
## 环境设置脚本
使用 set_env.sh脚本，可以启动CPU性能模式，开启irqbalance服务并且可以关闭ACS，该工具需要sudo权限。

# Dependencies
当前工具需要在环境中安装以下软件：sudo，pciutils，irqbalance和linux-tools。其中linux-tools的版本需要和系统内核版本一致，安装指令如下（ubunt系统）：
```
apt-get install sudo pciutils irqbalance linux-tools-$(uname -r) 
```
# Limitation
当前该工具仅支持Ubuntu 20.04，ubuntu 22.04 和Debian10.11操作系统，其他操作系统需要自行支持。
