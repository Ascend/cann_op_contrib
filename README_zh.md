# cann_op_contrib

[View English](./README.md)

<!-- TOC -->

- [生态算子仓介绍](#生态算子仓介绍)
- [目录结构简介](#目录结构简介)
- [开发环境要求](#开发环境要求)
- [算子编译打包](#算子编译打包)
- [算子包部署](#算子包部署)
- [算子开发文档](#算子开发文档)
- [贡献](#贡献)
- [许可证](#许可证)


<!-- /TOC -->
## 生态算子仓介绍
生态算子仓提供生态算子供开发者参考使用，开发者可基于该代码仓进行算子开发，编译构建生态算子包并部署使用。

## 目录结构简介
源码目录结构如下所示。
<details open><summary><b> cann_op_contrib</b></summary><blockquote>
<details><summary><b> cmake</b></summary><blockquote>
<details><summary><b> external</b></summary><blockquote>
<b>存放依赖的第三方库编译文件</b><br>
</blockquote></details>
<details><summary><b> util</b></summary><blockquote>
<b>存放工具类编译文件</b><br>
</blockquote></details>
<b>dependencies.cmake</b><br>
</blockquote></details>
<details open><summary><b> community</b></summary><blockquote>
<details><summary><b> common</b></summary><blockquote>
<details><summary><b> inc</b></summary><blockquote>
<b>主要存放算子原型工具类头文件</b><br>
</blockquote></details>
<details><summary><b> src</b></summary><blockquote>
<b>主要存放算子原型工具类实现</b><br>
</blockquote></details>
<details><summary><b> utils</b></summary><blockquote>
<b>主要存放算子实现工具类</b><br>
</blockquote></details>
</blockquote></details>

<details open><summary><b> ops</b></summary><blockquote>
<details open><summary><b> add(以add算子为例)</b></summary><blockquote>
<details><summary><b> ai_core</b></summary><blockquote>
<details><summary><b> impl</b></summary><blockquote>
<b>add.py 算子实现代码</b><br>
</blockquote></details>
<details><summary><b> op_info_cfg</b></summary><blockquote>
<details><summary><b> ascend310</b></summary><blockquote>
<b>add.ini 算子信息配置文件</b><br>
</blockquote></details>
<details><summary><b> ascend310p</b></summary><blockquote>
<b>add.ini 算子信息配置文件</b><br>
</blockquote></details>
<details><summary><b> ascend910</b></summary><blockquote>
<b>add.ini 算子信息配置文件</b><br>
</blockquote></details>
<details><summary><b> ascend910b</b></summary><blockquote>
<b>add.ini 算子信息配置文件</b><br>
</blockquote></details>
</blockquote></details>

<details><summary><b> op_tiling</b></summary><blockquote>
<b>add_tiling.cc 算子tiling文件</b><br>
</blockquote></details>
</blockquote></details>
<details><summary><b> ai_cpu</b></summary><blockquote>
<details><summary><b> impl</b></summary><blockquote>
<b>add.cc  算子实现代码</b><br>
<b>add.h  算子实现头文件</b><br>
</blockquote></details>
<details><summary><b> op_info_cfg</b></summary><blockquote>
<b>add.ini 算子信息配置文件</b><br>
</blockquote></details>
</blockquote></details>
<details><summary><b> framework</b></summary><blockquote>
<details><summary><b> onnx</b></summary><blockquote>
<b>add_plugin.cc  算子适配onnx框架插件代码</b><br>
</blockquote></details>
<details><summary><b> tf</b></summary><blockquote>
<b>add_plugin.cc  算子适配tf框架插件代码</b><br>
</blockquote></details>
<details><summary><b> caffe</b></summary><blockquote>
<b>add_plugin.cc  算子适配caffe框架插件代码</b><br>
</blockquote></details>
</blockquote></details>
<details><summary><b> op_proto</b></summary><blockquote>

<details><summary><b> inc</b></summary><blockquote>
<b>add_op.h 算子原型IR注册文件</b><br>
</blockquote></details>

<b>add_proto.cc 算子原型实现文件</b><br>
</blockquote></details>
</blockquote></details>
</blockquote></details>
<details><summary><b> tests</b></summary><blockquote>
<details><summary><b> add(以add算子为例)</b></summary><blockquote>
<b> ut</b><br>
</blockquote></details>
</blockquote></details>
</blockquote></details>
<details><summary><b> scripts</b></summary><blockquote>
<b>ai_core_parse_ini.py  tbe算子信息配置解析脚本</b><br>
<b>CANN_OP_CONTRIB_install.sh  生态仓算子部署脚本</b><br>
<b>install_run.sh  生态仓run包安装脚本</b><br>
<b>parse_ini.py  aicpu算子信息配置解析脚本</b><br>

</blockquote></details>
<b>CMakeLists.txt 生态仓cmake编译脚本</b><br>
<b>build.sh 编译脚本，用于触发生态仓代码编译及编译出的文件路径变更</b><br>
<b>pack.sh 生态仓run包打包脚本</b><br>
</blockquote></details>
</blockquote></details>

## 开发环境要求
使用源码进行编译构建之前，需安装社区版本的对应平台开发套件软件包(toolkit)及配套版本的算子开发工具包(communitysdk)

[软件包下载地址及用户手册](#https://www.hiascend.com/software/cann/community)

其中communitysdk包需解压到toolkit安装路径(如~/Ascend/ascend-toolkit/latest/)

## 算子编译打包
算子开发指导请参见[算子开发指南](#https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/63RC1alpha001/operatordevelopment/opdevg/atlasopdev_10_0001.html)，完成所需交付件的实现，存放要求请参见[目录结构简介](#目录结构简介)

命令行下执行编译打包脚本。       
**注：环境需要配置ASCEND_AICPU_PATH环境变量，否则aicpu算子会编译失败。**
```
./pack.sh
```

若只希望执行编译选项，可单独执行编译脚本
```
./build.sh
```
> **说明**：build.sh还可执行UT测试功能，具体方法可执行帮助命令进行查看。
    ```
    ./build.sh -h
    ```

## 算子包部署
执行算子包安装命令进行安装。   
**注：安装前需保证环境存在ASCEND_OPP_PATH环境变量，否则会安装失败。**
```
./CANN_OP_CONTRIB_linux-x86_64.run --install
```
> **说明** ：安装包支持参数可通过-h查看
    ```
    ./CANN_OP_CONTRIB_linux-x86_64.run -h
    ```

## 算子开发文档
有关开发指南、教程和API的更多详细信息，请参阅[学习向导](#https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/63RC1alpha001/operatordevelopment/opdevg/atlasopdev_10_0001.html)。

## 贡献
欢迎参与贡献。更多详情，请参阅我们的[贡献者Wiki](#https://gitee.com/ascend/cann_op_contrib/blob/master/CONTRIBUTING_CN.md)。

## 许可证
[Apache License 2.0](https://gitee.com/ascend/cann_op_contrib/blob/master/LICENSE)
