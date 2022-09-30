#!/bin/bash

readonly PACKAGE_LOG_NAME="CANN_OP_CONTRIB"
log_file_name="cann_op_contrib_operate.log" # log文件名字
log_file=""                                 # log文件带路径
operate_path=""                             # cann_op_contirb操作路径
PACKAGE_NAME=""                             # run包名称
script_path="$( cd "$(dirname $BASH_SOURCE)" ; pwd -P)"

# 用户配置初始化
if [ "$UID" = "0" ]; then
    LOG_PATH="/var/log/ascend_seclog/"
    log_file="${LOG_PATH}${log_file_name}"
else
    LOG_PATH="${HOME}/var/log/ascend_seclog/"
    log_file="${LOG_PATH}${log_file_name}"
fi

####################################
####  公用函数 #####
# 打印提示信息
function print_usage() {
    echo "Please input this command for more help: ./${PACKAGE_NAME} --help"
}

# 将信息输出到日志文件中
function log() {
    if [ x$log_file = x ] || [ ! -f $log_file ]; then
        echo -e "[${PACKAGE_LOG_NAME}] [$(date +%Y%m%d-%H:%M:%S)] [$1] $2"
    elif [ -f $log_file ]; then
        echo -e "[${PACKAGE_LOG_NAME}] [$(date +%Y%m%d-%H:%M:%S)] [$1] $2" >>$log_file
    fi
}

# 将关键信息打印到屏幕上
function print() {
    if [ x$log_file = x ] || [ ! -f $log_file ]; then
        echo -e "[${PACKAGE_LOG_NAME}] [$(date +%Y%m%d-%H:%M:%S)] [$1] $2"
    else
        echo -e "[${PACKAGE_LOG_NAME}] [$(date +%Y%m%d-%H:%M:%S)] [$1] $2" | tee -a $log_file
    fi
}

# 安全删除文件夹
function fn_del_dir() {
    local dir_path=$1
    local is_empty=$2
    # 判断变量不为空且不是系统根盘
    if [ -n "${dir_path}" ] && [[ ! "${dir_path}" =~ ^/+$ ]]; then
        # 判断是否是目录
        if [ -d "${dir_path}" ]; then
            # 判断是否需要判断目录为空不删除
            if [ x"${is_empty}" == x ] || [ "$(ls -A ${dir_path})" = "" ]; then
                chmod -R 700 "${dir_path}" 2>/dev/null
                rm -rf "${dir_path}"
                log "INFO" " delete directory ${dir_path} successfully."
                return 0
            else
                log "INFO" " delete operation, the directory ${dir_path} is not empty."
                return 1
            fi
        else
            log "WARNING" " delete operation, the ${dir_path} is not exist or not directory."
            return 1
        fi
    else
        log "WARNING" " delete operation, directory parameter invalid."
        return 2
    fi
}

# 设置权限掩码
function change_umask() {
    if [ ${UID} -eq 0 ] && [ $(umask) != "0022" ]; then
        log "INFO" "change umask 0022"
        umask 0022
    elif [ ${UID} -ne 0 ] && [ $(umask) != "0002" ]; then
        log "INFO" "change umask 0002"
        umask 0002
    fi
}

# 创建文件夹
function make_dir() {
    change_umask
    log "INFO" "mkdir ${1}"
    mkdir -p ${1} 2>/dev/null
    if [ $? -ne 0 ]; then
        print "ERROR" "create $1 fail !"
        return 1
    fi
    return 0
}

# 拷贝文件
function copy_file() {
    change_umask
    log "INFO" "copy ${1} to ${2}"
    cp -r ${1} ${2} 2>/dev/null
    if [ $? -ne 0 ]; then
        print "ERROR" "copy to ${2} fail !"
        return 1
    fi
    return 0
}

# 创建文件
function make_file() {
    change_umask
    log "INFO" "touch ${1}"
    touch ${1} 2>/dev/null
    if [ $? -ne 0 ]; then
        print "ERROR" "create $1 fail !"
        return 1
    fi
    return 0
}

####################################
####  日志模块初始化 #####
function log_init() {
    # 判断输入的安装路径路径是否存在，不存在则创建
    if [ ! -d $LOG_PATH ]; then
        make_dir "$LOG_PATH"
	if [ $? -ne 0 ];then
            return 1
	fi
    fi
    # 判断日志文件是否存在，不存在则创建；存在则判断是否大于50M
    if [ ! -f $log_file ]; then
        make_file "$log_file"
	if [ $? -ne 0 ];then
            return 1
	fi
        chmod 640 ${log_file}
    else
        local filesize=$(ls -l $log_file | awk '{ print $5 }')
        local maxsize=$((1024 * 1024 * 50))
        if [ $filesize -gt $maxsize ]; then
            local log_file_move_name="cann_op_contrib_operate_bak.log"
            mv -f ${log_file} ${LOG_PATH}${log_file_move_name}
            chmod 440 ${LOG_PATH}${log_file_move_name}
            make_file "$log_file"
	    if [ $? -ne 0 ];then
                return 1;
	    fi
            chmod 640 ${log_file}
            log "INFO" "log file > 50M, move ${log_file} to ${LOG_PATH}${log_file_move_name}."
        fi
    fi
    print "INFO" "LogFile:$log_file"
    return 0
}

###################################
####  程序运行 #####
# 设置操作路径
function set_operate_path() {
    operate_path=${ASCEND_OPP_PATH}
    if [ x${operate_path} = x ];then
        print "ERROR" "Please set environment variables ASCEND_OPP_PATH."
	return 1
    fi
    if [ ! -d ${operate_path} ];then
        print "ERROR" "The path ${operate_path} does not exist, Please check."
	return 1
    fi
    PACKAGE_NAME=$1
    return 0
}

# 安装
function install_process() {
    print "INFO" "install start."
    print "INFO" "The installation path is ${operate_path}."
    # 检查是否已经安装，如果已经安装，则退出
    if [ -d ${operate_path}/framework/vendor ] || [ -d ${operate_path}/op_proto/vendor ] || [ -d ${operate_path}/op_impl/vendor ]; then
        print "ERROR" "run package is already installed, install failed."
	return 1
    fi
    # 安装
    chmod +w ${operate_path}/framework  2>/dev/null
    chmod +w ${operate_path}/op_proto 2>/dev/null
    chmod +w ${operate_path}/op_impl 2>/dev/null
    make_dir ${operate_path}/framework/vendor
    if [ $? -ne 0 ];then
        return 1;
    fi
    make_dir ${operate_path}/op_proto/vendor
    if [ $? -ne 0 ];then
        return 1;
    fi
    make_dir ${operate_path}/op_impl/vendor
    if [ $? -ne 0 ];then
        return 1;
    fi
    copy_file ${script_path}/../framework/vendor/* ${operate_path}/framework/vendor
    if [ $? -ne 0 ];then
        return 1;
    fi
    copy_file ${script_path}/../op_proto/vendor/* ${operate_path}/op_proto/vendor
    if [ $? -ne 0 ];then
        return 1;
    fi
    copy_file ${script_path}/../op_impl/vendor/* ${operate_path}/op_impl/vendor
    if [ $? -ne 0 ];then
        return 1;
    fi
    chmod -w ${operate_path}/framework 2>/dev/null
    chmod -w ${operate_path}/op_proto 2>/dev/null
    chmod -w ${operate_path}/op_impl 2>/dev/null
    print "INFO" "install success."
    return 0
}

# 更新操作
function upgrade_process() {
    print "INFO" "upgrade start."
    print "INFO" "The upgrade path is ${operate_path}."
    # 检查是否已经安装，如果没有安装，则退出
    if [ ! -d ${operate_path}/framework/vendor ] || [ ! -d ${operate_path}/op_proto/vendor ] || [ ! -d ${operate_path}/op_impl/vendor ]; then
        print "ERROR" "run package is not installed on path ${install_path}, upgrade failed !"
	return 1
    fi
    # 升级
    chmod +w ${operate_path}/framework  2>/dev/null
    chmod +w ${operate_path}/op_proto 2>/dev/null
    chmod +w ${operate_path}/op_impl 2>/dev/null
    fn_del_dir ${operate_path}/framework/vendor 
    fn_del_dir ${operate_path}/op_proto/vendor 
    fn_del_dir ${operate_path}/op_impl/vendor 
    copy_file ${script_path}/../framework/vendor ${operate_path}/framework
    if [ $? -ne 0 ];then
        return 1;
    fi
    copy_file ${script_path}/../op_proto/vendor ${operate_path}/op_proto
    if [ $? -ne 0 ];then
        return 1;
    fi
    copy_file ${script_path}/../op_impl/vendor ${operate_path}/op_impl
    if [ $? -ne 0 ];then
        return 1;
    fi
    chmod -w ${operate_path}/framework 2>/dev/null
    chmod -w ${operate_path}/op_proto 2>/dev/null
    chmod -w ${operate_path}/op_impl 2>/dev/null
    print "INFO" "upgrade success."
    return 0
}

# 卸载操作
function uninstall_process() {
    print "INFO" "uninstall start"
    # 卸载
    chmod +w ${operate_path}/framework  2>/dev/null
    chmod +w ${operate_path}/op_proto 2>/dev/null
    chmod +w ${operate_path}/op_impl 2>/dev/null
    if [ -d ${operate_path}/framework/vendor ];then
        fn_del_dir ${operate_path}/framework/vendor
    fi
    if [ -d ${operate_path}/op_proto/vendor ];then
        fn_del_dir ${operate_path}/op_proto/vendor
    fi
    if [ -d ${operate_path}/op_impl/vendor ];then
        fn_del_dir ${operate_path}/op_impl/vendor
    fi
    chmod -w ${operate_path}/framework 2>/dev/null
    chmod -w ${operate_path}/op_proto 2>/dev/null
    chmod -w ${operate_path}/op_impl 2>/dev/null
    print "INFO" "uninstall success."
}

###################################
#### 程序开始 ####
function main() {
    log_init
    if [ $? -ne 0 ];then
        return 1;
    fi
    set_operate_path $*
    if [ $? -ne 0 ];then
        return 1;
    fi
    if [ x$2 = "xinstall" ];then
        install_process
	if [ $? -ne 0 ];then
            return 1;
        fi
    elif [ x$2 = "xupgrade" ];then
        upgrade_process
	if [ $? -ne 0 ];then
            return 1;
        fi
    elif [ x$2 = "xuninstall" ];then
        uninstall_process
    else
        print_usage
	return 1;
    fi
    return 0
}

main $*
