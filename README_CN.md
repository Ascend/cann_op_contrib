# cann_op_contrib

#### 介绍
算子开源仓

#### 目录结构
仓库目录结构如下所示。

<details open><summary><b> cann_op_contrib</b></summary><blockquote>
<details open><summary><b> community</b></summary><blockquote>
<details><summary><b> common</b></summary><blockquote>
<b> inc</b><br>
<b> src</b><br>
<b> utils</b><br>
</blockquote></details>
<details><summary><b> framework</b></summary><blockquote>
<b> onnx</b><br>
<b> tf</b><br>
</blockquote></details>
<details open><summary><b> ops</b></summary><blockquote>
<details open><summary><b> add(以add算子为例)</b></summary><blockquote>
<details><summary><b> ai_core</b></summary><blockquote>
<b> cust_impl</b><br>
<b> op_info_cfg</b><br>
<b> op_tiling</b><br>
</blockquote></details>
<details><summary><b> ai_cpu</b></summary><blockquote>
<b> impl</b><br>
<b> op_info_cfg</b><br>
</blockquote></details>
<details><summary><b> op_proto</b></summary><blockquote>
<b> inc</b><br>
</blockquote></details>
</blockquote></details>
</blockquote></details>
<details open><summary><b> tests</b></summary><blockquote>
<details><summary><b> add(以add算子为例)</b></summary><blockquote>
<b> ut</b><br>
</blockquote></details>
</blockquote></details>
</blockquote></details>
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

