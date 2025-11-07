FROM ubuntu:22.04                                                                                                                                                                              
                                                                                                                                                                                                 
COPY res /tmp                                                                                                                                                                                  
                                                                                                                                                                                                 
SHELL ["/bin/bash", "-c"]                                                                                                                                                                      
                                                                                                                                                                                                 
# 变量设置                                                                                                                                                                                     
ARG CANN_VER="8.0.0"                                                                                                                                                                           
ARG PYTHON_VER="3.10.15"                                                                                                                                                                       
ARG PYTHON_MAIN_VER="3.10"                                                                                                                                                                     
ARG CPYTHON_VER="cp310"                                                                                                                                                                        
ARG TORCH_NPU_PKG="torch_npu-2.1.0.post10-${CPYTHON_VER}-${CPYTHON_VER}-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"                                                                      
ARG TOOLKIT_PKG="Ascend-cann-toolkit_${CANN_VER}_linux-aarch64.run"                                                                                                                            
ARG KERNEL_PKG="Ascend-cann-kernels-910b_${CANN_VER}_linux-aarch64.run"                                                                                                                        
ARG NNAL_PKG="Ascend-cann-nnal_${CANN_VER}_linux-aarch64.run"                                                                                                                                  
                                                                                                                                                                                                 
ENV DEBIAN_FRONTEND=noninteractive                                                                                                                                                             
ENV GIT_SSL_NO_VERIFY=1                                                                                                                                                                        
ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest                                                                                                                                
ENV LD_LIBRARY_PATH=/usr/local/python${PYTHON_VER}/lib:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/add-ons:/usr/local/Ascend/driver/tools/hccn
_tool:$LD_LIBRARY_PATH                                                                                                                                                                         
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/lib64:${ASCEND_TOOLKIT_HOME}/lib64/plugin/opskernel:${ASCEND_TOOLKIT_HOME}/lib64/plugin/nnengine:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/
tbe/op_tiling/lib/linux/$(arch):$LD_LIBRARY_PATH                                                                                                                                               
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/tools/aml/lib64:${ASCEND_TOOLKIT_HOME}/tools/aml/lib64/plugin:$LD_LIBRARY_PATH                                                                      
ENV PYTHONPATH=${ASCEND_TOOLKIT_HOME}/python/site-packages:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe:/usr/local/python${PYTHON_VER}/lib/python${PYTHON_MAIN_VER}/site-packages:$PY
THONPATH                                                                                                                                                                                       
ENV PATH=/usr/local/python${PYTHON_VER}/bin:/usr/local/python${PYTHON_VER}/lib/python${PYTHON_MAIN_VER}/site-packages/torch/bin:${ASCEND_TOOLKIT_HOME}/bin:${ASCEND_TOOLKIT_HOME}/compiler/ccec_
compiler/bin:${ASCEND_TOOLKIT_HOME}/tools/ccec_compiler/bin:$PATH                                                                                                                              
ENV ASCEND_AICPU_PATH=${ASCEND_TOOLKIT_HOME}                                                                                                                                                   
ENV ASCEND_OPP_PATH=${ASCEND_TOOLKIT_HOME}/opp                                                                                                                                                 
ENV TOOLCHAIN_HOME=${ASCEND_TOOLKIT_HOME}/toolkit                                                                                                                                              
ENV ASCEND_HOME_PATH=${ASCEND_TOOLKIT_HOME}                                                                                                                                                    
                                                                                                                                                                                                 
RUN rm -f /etc/apt/sources.list.d/* \                                                                                                                                                          
    && sed -i -r 's@(archive|security|ports).ubuntu.com@mirrors.aliyun.com@g' /etc/apt/sources.list \                                                                                          
    && apt -o "Acquire::https::Verify-Peer=false" update \                                                                                                                                     
    && apt -o "Acquire::https::Verify-Peer=false" install -y --no-install-recommends ca-certificates \                                                                                         
    && apt install -y sudo curl gcc g++ make git cmake zlib1g zlib1g-dev openssh-server openssl libssl-dev libsqlite3-dev libffi-dev unzip \                                                   
       pciutils net-tools libblas-dev gfortran libblas3 unzip vim git wget dos2unix lzma libgl1-mesa-glx libglib2.0-dev python3-dev \                                                                      
       pkg-config liblzma-dev libbz2-dev bash-completion \                                                                                                                                                     
    && apt clean && rm -rf /var/lib/apt/lists/* \                                                                                                                                              
    && rm -f /var/lib/dpkg/statoverride \                                                                                                                                                      
    && touch /var/lib/dpkg/statoverride                                                                                                                                                        
                                                                                                                                                                                                 
RUN ssh-keygen -A \                                                                                                                                                                            
    && chmod 666 /etc/ssh/* \                                                                                                                                                                  
    && sed -i 's/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g' /etc/ssh/ssh_config \                                                                                                             
    && echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config \                                                                                                                        
    && sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config                                                                                                                              
                                                                                                                                                                                                 
RUN cd /tmp \                                                                                                                                                                                  
    && wget -O Python-${PYTHON_VER}.tgz "https://www.python.org/ftp/python/${PYTHON_VER}/Python-${PYTHON_VER}.tgz" \                                                                           
    && tar -xzf Python-${PYTHON_VER}.tgz \                                                                                                                                                     
    && cd Python-${PYTHON_VER} \                                                                                                                                                               
    && ./configure --prefix=/usr/local/python${PYTHON_VER} --enable-loadable-sqlite-extensions --enable-shared \                                                                               
    && make -j 4 && make install \                                                                                                                                                             
    && cd /tmp && rm -rf Python-${PYTHON_VER} Python-${PYTHON_VER}.tgz \                                                                                                                       
    && ln -sf /usr/local/python${PYTHON_VER}/bin/python3 /usr/bin/python \                                                                                                                     
    && ln -sf /usr/local/python${PYTHON_VER}/bin/python3 /usr/bin/python3 \                                                                                                                    
    && ln -sf /usr/local/python${PYTHON_VER}/bin/pip3 /usr/bin/pip \                                                                                                                           
    && ln -sf /usr/local/python${PYTHON_VER}/bin/pip3 /usr/bin/pip3                                                                                                                            
                                                                                                                                                                                                 
RUN groupadd  HwHiAiUser -g 1000 \                                                                                                                                                             
    && useradd -d /home/HwHiAiUser -u 1000 -g 1000 -m -s /bin/bash HwHiAiUser \                                                                                                                
    && mkdir -p /home/HwHiAiUser \                                                                                                                                                             
    && chown HwHiAiUser:HwHiAiUser /home/HwHiAiUser                                                                                                                                            
                                                                                                                                                                                                 
# 安装pip包                                                                                                                                                                                    
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir --upgrade pip \                                                                                                    
    && pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir numpy==1.26.0 \                                                                                                 
    && pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir decorator \                                                                                                     
    && pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir scipy \                                                                                                         
    && pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir einops \                                                                                                        
    && pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir datasets \                                                                                                      
    && pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir six \                                                                                                           
    && pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir torch==2.1.0 \                                                                                                  
    && pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir torchvision==0.16.0 \                                                                                           
    && pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir transformers==4.38.2  \                                                                                         
    && pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir protobuf \                                                                                                      
    && pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir regex \                                                                                                         
    && pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir psutil \                                                                                                        
    && pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir pyyaml \                                                                                                        
    && pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir accelerate \                                                                                                    
    && pip3 install --no-cache-dir /tmp/pkg/apex-0.1+ascend-${CPYTHON_VER}-${CPYTHON_VER}-linux_aarch64.whl                                                                                    
                                                                                                                                                                                                 
# 安装toolkit/kernel/                                                                                                                                                                          
RUN cd /tmp \                                                                                                                                                                                  
    && wget -O "/tmp/pkg/$TORCH_NPU_PKG" "https://gitee.com/ascend/pytorch/releases/download/v6.0.0-pytorch2.1.0/${TORCH_NPU_PKG}" \                                                           
    && wget -O "/tmp/pkg/$TOOLKIT_PKG" "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%20${CANN_VER}/${TOOLKIT_PKG}?response-content-type=application/octet-stream" \           
    && wget -O "/tmp/pkg/$KERNEL_PKG" "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%20${CANN_VER}/${KERNEL_PKG}?response-content-type=application/octet-stream" \             
    && wget -O "/tmp/pkg/$NNAL_PKG" "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%20${CANN_VER}/${NNAL_PKG}?response-content-type=application/octet-stream" \                 
    && chmod +x "/tmp/pkg/$TOOLKIT_PKG" "/tmp/pkg/$KERNEL_PKG" "/tmp/pkg/$NNAL_PKG" \                                                                                                          
    && pip3 install --no-cache-dir "/tmp/pkg/$TORCH_NPU_PKG" \                                                                                                                                 
    && umask 0022 \                                                                                                                                                                            
    && ./pkg/$TOOLKIT_PKG --install-path=/usr/local/Ascend/ --install --quiet --install-for-all \                                                                                              
    && ./pkg/$KERNEL_PKG --install-path=/usr/local/Ascend/ --install --quiet --install-for-all \                                                                                               
    && ./pkg/$NNAL_PKG --install-path=/usr/local/Ascend/ --install --quiet --install-for-all \                                                                                                 
    && rm -rf /tmp/*                                                                                                                                                                           
                                                                                                                                                                                                 
# 安装MindSpeed for all                                                                                                                                                                        
RUN cd / \                                                                                                                                                                                     
    && git clone https://gitee.com/ascend/MindSpeed.git \                                                                                                                                      
    && cd MindSpeed \                                                                                                                                                                          
    && git checkout 4ea42a23 \                                                                                                                                                                 
    && chmod -R 777 /MindSpeed \                                                                                                                                                               
    && pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt \                                                                                                          
    && pip3 install -e . 


RUN cd ps-slm \
  && pip3 install -r requirements.txt
