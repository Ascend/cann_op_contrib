# cann_op_contrib

#### 介绍
算子开源仓

#### 目录结构
仓库目录结构如下所示。

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
<details><summary><b> framework</b></summary><blockquote>
<details><summary><b> onnx</b></summary><blockquote>
<b>add_plugin.cc  算子适配onnx框架插件代码</b><br>
<b>CMakeLists.txt  算子适配插件编译文件</b><br>
</blockquote></details>
<details><summary><b> tf</b></summary><blockquote>
<b>add_plugin.cc  算子适配tf框架插件代码</b><br>
<b>CMakeLists.txt  算子适配插件编译文件</b><br>
</blockquote></details>
</blockquote></details>
<details open><summary><b> ops</b></summary><blockquote>
<details open><summary><b> add(以add算子为例)</b></summary><blockquote>
<details><summary><b> ai_core</b></summary><blockquote>
<details><summary><b> cust_impl</b></summary><blockquote>
<b>add.py  算子实现代码</b><br>
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

<b> op_tiling</b><br>
</blockquote></details>
<details><summary><b> ai_cpu</b></summary><blockquote>
<details><summary><b> impl</b></summary><blockquote>
<b>add.cc  算子实现代码</b><br>
<b>add.h  算子代码头文件</b><br>
</blockquote></details>
<details><summary><b> op_info_cfg</b></summary><blockquote>
<b>add.ini 算子信息配置文件</b><br>
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
<details open><summary><b> tests</b></summary><blockquote>
<details><summary><b> add(以add算子为例)</b></summary><blockquote>
<b> ut</b><br>
</blockquote></details>
</blockquote></details>
</blockquote></details>

<details><summary><b> scripts</b></summary><blockquote>
<b>ai_core_parse_ini.py  tbe算子信息配置解析脚本</b><br>
<b>CANN_OP_CONTRIB_install.sh  生态仓算子部署脚本</b><br>
<b>install_run.sh  生态仓run包安装脚本</b><br>
<b>parse_ini.py  AiCpu算子信息配置解析脚本</b><br>
</blockquote></details>
<details><summary><b> third_party</b></summary><blockquote>
<b>metadef metadef仓代码</b><br>

</blockquote></details>
<b>CMakeLists.txt 生态仓总编译脚本</b><br>
<b>build.sh 编译脚本，用于触发生态仓代码编译及编译出的文件路径变更</b><br>

<b>pack.sh 生态仓run包打包脚本</b><br>
</blockquote></details>

#### 环境要求
使用此仓前，需要安装CANN-toolkit包，完整搭建开发环境。

#### 算子开发
请参见算子开发指南，完成哪些交付件的实现，存放要求，参见“目录结构”

#### 工程打包
命令行下执行打包脚本。    
**注1：首次编译可能会出现报错，重新再执行一次打包脚本即可。**    
**注2：环境需要配置ASCEND_AICPU_PATH环境变量，否则aicpu算子会编译失败。**
```
bash pack.sh
```

#### 算子包部署
执行安装命令进行安装。   
**注：安装前需保证环境存在ASCEND_OPP_PATH环境变量，否则会安装失败。**
```
./CANN_OP_CONTRIB_linux-x86_64.run --install
```

#### 参与贡献
1.  Fork 本仓库
2.  提交代码
3.  新建 Pull Request

